#!/usr/bin/env python3
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
import warnings
warnings.filterwarnings("ignore")

def load_model_and_tokenizer():
    """Load model and tokenizer"""
    print("Loading model and tokenizer...")
    
    # Base model configuration
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_model_name = "wingchiuloong/CopyPasteLLM-L3-8B"
    
    # Quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load LoRA weights
    print("Loading LoRA weights...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_model_name)
    
    return model, tokenizer

def create_pipeline(model, tokenizer):
    """Create inference pipeline"""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )

def format_prompt(user_input):
    """Format input to Llama-3 format"""
    system_message = "You are a helpful AI assistant."
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

def main():
    """Main function - One-time inference"""
    print("CopyPasteLLM inference Demo")
    print("=" * 50)
    
    # Set test question
    context = "Galileo Galilei, renowned as one of the most influential figures in the history of science, made numerous contributions that revolutionized our understanding of physics and astronomy. His meticulous work with telescopes led to groundbreaking discoveries about the moons of Jupiter and the phases of Venus. Beyond the realm of astronomy, his observations and experiments laid the foundation for classical mechanics. One of Galileo’s lesser-known achievements is his development of the Three Laws of Motion, which were critical in advancing the study of kinematics and dynamics. These laws articulate the principles of inertia, the relationship between force and motion, and the law of action and reaction, providing a comprehensive framework for understanding moving bodies. His work on pendulums also contributed substantially to timekeeping and horology, as he discovered that pendulums of different lengths oscillate at predictable periods, a principle still applied in modern clocks. Galileo’s interdisciplinary approach enabled him to synthesize knowledge from various fields, which allowed new theories to emerge, reshaping the scientific landscape of his time and beyond. Notably, his support of the heliocentric model of the solar system earned him both acclaim and censure, highlighting the tension between scientific inquiry and established doctrine. In contrast to Galen’s biological studies and Newton’s later contributions, Galileo’s articulation of the Three Laws of Motion was pivotal in the transition from Aristotelian physics to Newtonian mechanics. His contributions remain a testament to the interplay of observation, theory, and experimentation in scientific progress."
    question = "Which law was Galileo Galilei responsible for describing?"
    test_question = f"{context}\nQ: {question}\nA:"
    print(f"{test_question}")
    print("-" * 50)
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer()
        
        # Create pipeline
        print("Creating inference pipeline...")
        pipe = create_pipeline(model, tokenizer)
        
        print("Model loaded! Starting inference...")
        
        # Format input
        prompt = format_prompt(test_question)
        
        # Generate reply
        print("Generating reply...")
        outputs = pipe(
            prompt,
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Output result
        response = outputs[0]['generated_text']
        print(f"\nCopyPasteLLM:\n{response}")
                
    except Exception as e:
        print(f"Inference failed: {e}")
        print("Please ensure the necessary dependencies are installed: pip install transformers peft bitsandbytes accelerate")

if __name__ == "__main__":
    main()
