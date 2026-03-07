"""
CP-Refine Pipeline - Multi-agent iterative refinement for highest quality

This pipeline uses LangGraph for multi-agent orchestration with Extractor,
Reviewer, and Finalizer agents to iteratively refine responses.
"""

from typing import List, Dict, Any, TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from CopyPasteLLM.pipelines.base import PipelineBase
from CopyPasteLLM.utils.text import create_standard_response_structure


class RAGState(TypedDict):
    """State for CP-Refine pipeline"""
    context: str
    query: str
    response: str
    messages: Annotated[List, add_messages]
    extractiveness_score: float
    extractiveness_coverage: float
    extractiveness_density: float
    iteration_count: int
    final_response: str
    response_history: List[Dict[str, Any]]
    best_response: str
    best_extractiveness_score: float


class CPRefinePipeline(PipelineBase):
    """
    CP-Refine Pipeline - Multi-agent iterative refinement.

    Algorithm:
    1. Extractor Agent: Generate high-extractiveness response
    2. Reviewer Agent: Evaluate quality and provide feedback
    3. Finalizer Agent: Select best response based on quality threshold
    4. Iterate until quality threshold met or max iterations reached

    Args:
        verbose: Whether to print verbose output
        llm_backend: Optional pre-initialized LLMBackend
        max_iterations: Maximum refinement iterations
        min_extractiveness: Minimum extractiveness threshold (default: 0.99)
        **llm_kwargs: Additional arguments for LLMBackend
    """

    def __init__(
        self,
        verbose: bool = False,
        llm_backend=None,
        max_iterations: int = 5,
        min_extractiveness: float = 0.99,
        **llm_kwargs
    ):
        super().__init__(
            pipeline_name="CP-Refine",
            verbose=verbose,
            llm_backend=llm_backend,
            **llm_kwargs
        )

        self.max_iterations = max_iterations
        self.min_extractiveness = min_extractiveness

        # Create state graph
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create LangGraph state graph"""
        graph = StateGraph(RAGState)

        # Add nodes
        graph.add_node("extractor", self._extractor_agent)
        graph.add_node("reviewer", self._reviewer_agent)
        graph.add_node("finalizer", self._finalizer_agent)

        # Set graph structure
        graph.add_edge(START, "extractor")
        graph.add_conditional_edges(
            "extractor",
            self._should_review,
            {
                "review": "reviewer",
                "finalize": "finalizer"
            }
        )
        graph.add_conditional_edges(
            "reviewer",
            self._should_continue_iteration,
            {
                "continue": "extractor",
                "finalize": "finalizer"
            }
        )
        graph.add_edge("finalizer", END)

        # Compile graph
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    def _extractor_agent(self, state: RAGState) -> RAGState:
        """Extractor agent: Generate high-extractiveness response"""
        iteration = state.get("iteration_count", 0)
        self._print_verbose(f"Extractor agent executing (iteration {iteration + 1})", "Extractor")

        context = state["context"]
        query = state["query"]

        # Build extractor prompt
        if iteration == 0:
            requirements = """
1. RELEVANT CONTEXT REUSE: Incorporate relevant text from the context
2. MINIMAL ORIGINAL CONTENT: Limit additions to essential connections only
3. PRESERVE EXACT WORDING: Keep original phrases and expressions from context
4. CONTEXT-ONLY INFORMATION: Use only facts explicitly in the context
5. NO INFERENCE: Do not add any inferred content"""
        else:
            # If refining, incorporate feedback
            feedback = state.get("feedback", "")
            requirements = f"""
1. RELEVANT CONTEXT REUSE: Incorporate relevant text from the context
2. MINIMAL ORIGINAL CONTENT: Limit additions to essential connections only
3. PRESERVE EXACT WORDING: Keep original phrases and expressions from context
4. CONTEXT-ONLY INFORMATION: Use only facts explicitly in the context
5. NO INFERENCE: Do not add any inferred content
6. ADDRESS FEEDBACK: {feedback}"""

        extractor_prompt = f"""You are an expert at generating highly extractive responses that faithfully represent information from the given context.

## Context
{context}

## Question
{query}

## Requirements
{requirements}

Generate a comprehensive response that prioritizes information extraction from the context:"""

        try:
            response_content = self._call_llm(
                extractor_prompt,
                agent_name="Extractor",
                temperature=0.1
            )

            # Calculate extractiveness
            metrics = self._calculate_extractiveness(context, response_content)

            # Update state
            state["response"] = response_content
            state["extractiveness_score"] = metrics["score"]
            state["extractiveness_coverage"] = metrics["coverage"]
            state["extractiveness_density"] = metrics["density"]
            state["iteration_count"] = iteration + 1

            # Track best response
            if metrics["score"] > state.get("best_extractiveness_score", 0):
                state["best_response"] = response_content
                state["best_extractiveness_score"] = metrics["score"]

            # Add to history
            if "response_history" not in state:
                state["response_history"] = []

            state["response_history"].append({
                "iteration": iteration + 1,
                "response": response_content,
                "extractiveness_score": metrics["score"],
                "extractiveness_coverage": metrics["coverage"],
                "extractiveness_density": metrics["density"]
            })

            self._print_verbose(
                f"Extractor generated response (extractiveness: {metrics['score']:.3f})"
            )

        except Exception as e:
            self._print_verbose(f"Extractor failed: {str(e)}", "ERROR")
            state["response"] = state.get("best_response", "")
            state["extractiveness_score"] = state.get("best_extractiveness_score", 0.0)

        return state

    def _reviewer_agent(self, state: RAGState) -> RAGState:
        """Reviewer agent: Evaluate response quality and provide feedback"""
        self._print_verbose("Reviewer agent evaluating response", "Reviewer")

        response = state["response"]
        context = state["context"]
        query = state["query"]
        extractiveness = state["extractiveness_score"]

        reviewer_prompt = f"""You are an expert evaluator of RAG (Retrieval-Augmented Generation) responses. Your task is to evaluate the quality of a generated response and provide feedback for improvement.

## Context
{context}

## Question
{query}

## Generated Response
{response}

## Current Extractiveness Score
{extractiveness:.3f}

## Evaluation Criteria
1. Extractiveness: How much of the response comes from the context (0-1)
2. Contextual Faithfulness: Does the response stay faithful to the context?
3. Query Relevance: Does the response directly address the question?

## Instructions
Evaluate the response and provide:
1. A score for each criterion (0-1)
2. Specific feedback for improvement if extractiveness < {self.min_extractiveness}
3. A recommendation: "CONTINUE" if improvement is needed, "ACCEPT" if quality is sufficient

## Output Format
EXTRACTIVENESS_EVAL: [your score]
FAITHFULNESS_EVAL: [your score]
RELEVANCE_EVAL: [your score]
FEEDBACK: [specific feedback for improvement]
RECOMMENDATION: [CONTINUE or ACCEPT]

Evaluate the response:"""

        try:
            review_content = self._call_llm(
                reviewer_prompt,
                agent_name="Reviewer",
                temperature=0.1
            )

            # Parse review
            feedback = ""
            recommendation = "CONTINUE"

            lines = review_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('FEEDBACK:'):
                    feedback = line.replace('FEEDBACK:', '').strip()
                elif line.startswith('RECOMMENDATION:'):
                    recommendation = line.replace('RECOMMENDATION:', '').strip()

            state["feedback"] = feedback
            state["recommendation"] = recommendation

            self._print_verbose(
                f"Reviewer evaluation: {recommendation} "
                f"(extractiveness: {extractiveness:.3f})"
            )

        except Exception as e:
            self._print_verbose(f"Reviewer failed: {str(e)}", "ERROR")
            state["feedback"] = ""
            state["recommendation"] = "ACCEPT"

        return state

    def _finalizer_agent(self, state: RAGState) -> RAGState:
        """Finalizer agent: Select and return best response"""
        self._print_verbose("Finalizer agent selecting best response", "Finalizer")

        # Use best response from history
        best_response = state.get("best_response", state["response"])
        best_score = state.get("best_extractiveness_score", state["extractiveness_score"])

        state["final_response"] = best_response
        state["extractiveness_score"] = best_score

        self._print_verbose(
            f"Finalizer selected best response (extractiveness: {best_score:.3f})"
        )

        return state

    def _should_review(self, state: RAGState) -> str:
        """Decide whether to review or finalize"""
        iteration = state.get("iteration_count", 0)
        extractiveness = state.get("extractiveness_score", 0.0)

        # If quality threshold met, can finalize
        if extractiveness >= self.min_extractiveness:
            return "finalize"

        # If max iterations reached, finalize
        if iteration >= self.max_iterations:
            return "finalize"

        # Otherwise, review
        return "review"

    def _should_continue_iteration(self, state: RAGState) -> str:
        """Decide whether to continue iteration or finalize"""
        recommendation = state.get("recommendation", "ACCEPT")
        iteration = state.get("iteration_count", 0)

        if recommendation == "ACCEPT" or iteration >= self.max_iterations:
            return "finalize"
        else:
            return "continue"

    def process(self, context: str, query: str) -> Dict[str, Any]:
        """
        Process a RAG request with CP-Refine pipeline.

        Args:
            context: Context text
            query: User query

        Returns:
            Dictionary with response and metrics
        """
        self.printer.print_separator(f"Start {self.pipeline_name} Pipeline")

        # Initialize state
        initial_state = RAGState(
            context=context,
            query=query,
            response="",
            messages=[],
            extractiveness_score=0.0,
            extractiveness_coverage=0.0,
            extractiveness_density=0.0,
            iteration_count=0,
            final_response="",
            response_history=[],
            best_response="",
            best_extractiveness_score=0.0
        )

        try:
            # Run graph
            config = {"configurable": {"thread_id": "copypastellm-refine"}}
            final_state = self.graph.invoke(initial_state, config)

            final_response = final_state.get("final_response", final_state.get("response", ""))
            coverage = final_state.get("extractiveness_coverage", 0.0)
            density = final_state.get("extractiveness_density", 0.0)
            score = final_state.get("extractiveness_score", 0.0)
            iteration_count = final_state.get("iteration_count", 0)
            response_history = final_state.get("response_history", [])

            return create_standard_response_structure(
                output=final_response,
                extractiveness_coverage=coverage,
                extractiveness_density=density,
                extractiveness_score=score,
                extra={
                    "iteration_count": iteration_count,
                    "response_history": response_history,
                    "num_iterations": len(response_history)
                }
            )

        except Exception as e:
            self._print_verbose(f"CP-Refine pipeline failed: {str(e)}", "ERROR")
            return create_standard_response_structure(
                output="Unable to answer based on the given passages."
            )
