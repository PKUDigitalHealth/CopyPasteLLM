"""
Extractiveness Metrics for Copy-Paste RAG Pipelines

This module provides metrics and visualization tools for measuring how much
content in a response is extracted from the context, helping to identify
hallucinations and verify factual groundedness.

Implementation based on: Grusky et al., NEWSROOM: A Dataset of 1.3 Million Summaries (2018)
"""

import re
from typing import List, Optional, Tuple

# Optional dependencies
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    np = None


# Color utility functions (inline fallback)
def _get_color_from_gradient(fragment_length: int, max_length: int = 50) -> str:
    """Get ANSI color code based on fragment length"""
    if fragment_length == 0:
        return '\033[91m'  # Red for hallucinations
    elif fragment_length < 3:
        return '\033[93m'  # Yellow for short extracts
    elif fragment_length < 8:
        return '\033[94m'  # Blue for medium extracts
    else:
        return '\033[92m'  # Green for long extracts


def _generate_gradient_legend() -> str:
    """Generate color legend for heatmap"""
    return (
        "█ \033[91mNon-Copy-Paste\033[0m  "
        "█ \033[93mShort Copy-Paste (1-2)\033[0m  "
        "█ \033[94mMedium Copy-Paste (3-7)\033[0m  "
        "█ \033[92mLong Copy-Paste (8+)\033[0m"
    )


class ExtractivenessMetrics:
    """
    Calculate extractiveness metrics (Coverage and Density) for text.

    Supports both Chinese and English text with automatic language detection.

    Args:
        tokenizer: Optional custom tokenizer. If None, uses auto-detection.
        case_sensitive: Whether matching should be case-sensitive (default: False).
        language: Language mode ('auto', 'zh', 'en'). Default: 'auto'.
        verbose: Whether to print detailed information (default: False).
    """

    def __init__(
        self,
        tokenizer=None,
        case_sensitive: bool = False,
        language: str = 'auto',
        verbose: bool = False
    ):
        self.case_sensitive = case_sensitive
        self.language = language
        self.verbose = verbose

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self._get_default_tokenizer()

    def _contains_chinese(self, text: str) -> bool:
        """Detect if text contains Chinese characters"""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(chinese_pattern.search(text))

    def _get_default_tokenizer(self):
        """Get default tokenizer based on language settings"""

        def smart_tokenizer(text: str) -> List[str]:
            # Auto-detect or use specified language
            if self.language == 'auto':
                has_chinese = self._contains_chinese(text)
            elif self.language == 'zh':
                has_chinese = True
            else:  # self.language == 'en'
                has_chinese = False

            if has_chinese:
                if JIEBA_AVAILABLE:
                    # Use jieba for Chinese
                    return list(jieba.cut(text, cut_all=False))
                else:
                    # Fallback to character-level for Chinese
                    tokens = []
                    current_word = ""
                    for char in text:
                        if self._contains_chinese(char):
                            if current_word:
                                tokens.append(current_word)
                                current_word = ""
                            tokens.append(char)
                        elif char.isspace():
                            if current_word:
                                tokens.append(current_word)
                                current_word = ""
                        else:
                            current_word += char
                    if current_word:
                        tokens.append(current_word)
                    return [t for t in tokens if t.strip()]
            else:
                # Space-based tokenization for English and other languages
                return text.split()

        return smart_tokenizer

    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Normalize tokens (handle case sensitivity)"""
        if self.case_sensitive:
            return tokens
        else:
            return [token.lower() for token in tokens]

    def _find_extractive_fragments(
        self,
        context_tokens: List[str],
        response_tokens: List[str]
    ) -> List[List[str]]:
        """
        Identify extractive fragments using greedy longest-match algorithm.

        Args:
            context_tokens: Tokenized context text
            response_tokens: Tokenized response text

        Returns:
            List of fragments (each fragment is a list of tokens)
        """
        # Normalize tokens
        context_tokens_norm = self._normalize_tokens(context_tokens)
        response_tokens_norm = self._normalize_tokens(response_tokens)

        fragments = []
        i = 0

        while i < len(response_tokens_norm):
            best_match_length = 0
            current_token = response_tokens_norm[i]

            # Find all positions in context that match current token
            positions = [
                p for p, token in enumerate(context_tokens_norm)
                if token == current_token
            ]

            # Check continuous match length for each position
            for pos in positions:
                match_length = 0
                while (i + match_length < len(response_tokens_norm) and
                       pos + match_length < len(context_tokens_norm) and
                       response_tokens_norm[i + match_length] == context_tokens_norm[pos + match_length]):
                    match_length += 1

                if match_length > best_match_length:
                    best_match_length = match_length

            # Record fragment and skip matched tokens
            if best_match_length > 0:
                fragment = response_tokens[i:i + best_match_length]
                fragments.append(fragment)
                i += best_match_length
            else:
                i += 1

        # Remove duplicate fragments
        unique_fragments = []
        seen_fragments = set()

        for fragment in fragments:
            fragment_norm = self._normalize_tokens(fragment)
            fragment_tuple = tuple(fragment_norm)

            if fragment_tuple not in seen_fragments:
                seen_fragments.add(fragment_tuple)
                unique_fragments.append(fragment)

        return unique_fragments

    def _build_token_char_mapping(
        self,
        text: str,
        tokens: List[str]
    ) -> dict:
        """Build character position to token index mapping"""
        char_to_token = {}
        text_position = 0

        for token_index, token in enumerate(tokens):
            token_start = text.find(token, text_position)

            if token_start != -1:
                for i in range(len(token)):
                    char_to_token[token_start + i] = token_index
                text_position = token_start + len(token)
            else:
                # Character-level fallback for Chinese
                for char in token:
                    char_pos = text.find(char, text_position)
                    if char_pos != -1:
                        char_to_token[char_pos] = token_index
                        text_position = char_pos + 1

        return char_to_token

    def calculate_metrics(
        self,
        context: str,
        response: str
    ) -> Tuple[float, float]:
        """
        Calculate Coverage and Density metrics.

        Args:
            context: Original context text
            response: Response text to evaluate

        Returns:
            Tuple of (coverage, density) where both are floats between 0-1
        """
        context_tokens = self.tokenizer(context)
        response_tokens = self.tokenizer(response)

        if self.verbose:
            print("=" * 60)
            print("Tokenization Results")
            print("=" * 60)
            print(f"Context tokens: {len(context_tokens)}")
            print(f"Response tokens: {len(response_tokens)}")
            print("-" * 60)

        if len(response_tokens) == 0:
            if self.verbose:
                print("Empty response, returning 0 for both metrics")
            return 0.0, 0.0

        fragments = self._find_extractive_fragments(context_tokens, response_tokens)

        if self.verbose:
            print("Extracted Fragments")
            print("=" * 60)
            if fragments:
                for i, fragment in enumerate(fragments, 1):
                    fragment_text = " ".join(fragment)
                    print(f"Fragment {i}: '{fragment_text}' (len={len(fragment)})")
                print(f"Total fragments: {len(fragments)}")
                print(f"Total extracted tokens: {sum(len(f) for f in fragments)}")
            else:
                print("No fragments extracted")
            print("-" * 60)

        # Calculate coverage
        covered_count = sum(len(frag) for frag in fragments)
        coverage = covered_count / len(response_tokens)

        # Calculate density
        density = sum(len(frag)**2 for frag in fragments) / len(response_tokens)

        if self.verbose:
            print("Metrics Results")
            print("=" * 60)
            print(f"Coverage: {coverage:.4f} ({covered_count}/{len(response_tokens)})")
            print(f"Density: {density:.4f}")
            print("=" * 60)

        return coverage, density

    def calculate_extractiveness_score(
        self,
        context: str,
        response: str
    ) -> Tuple[float, float, float]:
        """
        Calculate overall extractiveness score.

        Args:
            context: Original context text
            response: Response text to evaluate

        Returns:
            Tuple of (coverage, density, score) where score combines both metrics
        """
        coverage, density = self.calculate_metrics(context, response)
        score = coverage * 0.6 + min(density**0.25 / 4, 0.4)
        return coverage, density, score

    def get_extractive_fragments(
        self,
        context: str,
        response: str,
        descending: Optional[bool] = None,
        include_min_len: int = 1
    ) -> Tuple[List[str], float]:
        """
        Get all extractive fragments from response.

        Args:
            context: Original context text
            response: Response text to analyze
            descending: Sort fragments by length (True=descending, False=ascending, None=no sort)
            include_min_len: Minimum fragment length to include in coverage calculation

        Returns:
            Tuple of (fragments_text, coverage_score) where fragments_text is a list
            of fragment strings and coverage_score is the effective coverage ratio
        """
        context_tokens = self.tokenizer(context)
        response_tokens = self.tokenizer(response)

        if len(response_tokens) == 0:
            if self.verbose:
                print("Empty response, returning empty fragments")
            return [], 0.0

        fragments = self._find_extractive_fragments(context_tokens, response_tokens)

        if descending is not None:
            fragments = sorted(fragments, key=len, reverse=descending)

        effective_fragments_num = 0
        effective_fragments_str = []

        for fragment in fragments:
            if len(fragment) >= include_min_len:
                effective_fragments_num += len(fragment)
                effective_fragments_str.append(" ".join(fragment))

        coverage = effective_fragments_num / len(response_tokens)
        return effective_fragments_str, coverage

    def render_extractiveness_heatmap(
        self,
        context: str,
        response: str,
        show_legend: bool = False
    ) -> str:
        """
        Render terminal heatmap with colored extractive fragments.

        Args:
            context: Original context text
            response: Response text to visualize
            show_legend: Whether to include color legend

        Returns:
            String with ANSI color codes for terminal display
        """
        context_tokens = self.tokenizer(context)
        response_tokens = self.tokenizer(response)

        if len(response_tokens) == 0:
            return "Empty response, cannot render heatmap."

        fragments = self._find_extractive_fragments(context_tokens, response_tokens)

        # Build fragment position map
        fragment_map = {}
        response_tokens_norm = self._normalize_tokens(response_tokens)
        used_positions = set()

        for fragment in fragments:
            fragment_norm = self._normalize_tokens(fragment)
            fragment_length = len(fragment)

            # Find first unused matching position in response
            for i in range(len(response_tokens_norm) - len(fragment_norm) + 1):
                if (i not in used_positions and
                    response_tokens_norm[i:i+len(fragment_norm)] == fragment_norm):

                    if all(j not in used_positions for j in range(i, i + len(fragment_norm))):
                        for j in range(len(fragment_norm)):
                            fragment_map[i + j] = fragment_length
                            used_positions.add(i + j)
                        break

        # Render character-by-character
        reset_code = '\033[0m'
        result_chars = []
        token_char_map = self._build_token_char_mapping(response, response_tokens)

        for char_index, char in enumerate(response):
            token_index = token_char_map.get(char_index, -1)

            if token_index >= 0 and token_index in fragment_map:
                fragment_length = fragment_map[token_index]
            else:
                fragment_length = 0

            color_code = _get_color_from_gradient(fragment_length)
            result_chars.append(f"{color_code}{char}{reset_code}")

        result = ''.join(result_chars)

        if show_legend:
            legend = "\n\nHeatmap Legend:\n"
            legend += _generate_gradient_legend()
            legend += "\n🌈 Longer extracts are more reliable (green=long, red=external knowledge)"
            result += legend

        return result

    def render_extractiveness_latex_heatmap(
        self,
        context: str,
        response: str,
        show_legend: bool = False,
        min_fragment_length: int = 2
    ) -> str:
        """
        Render LaTeX heatmap for academic papers.

        Rules:
        - \\hlr{} (red highlight): tokens never appearing in context (hallucinations)
        - \\hlg{} (green highlight): extractive fragments >= min_fragment_length

        Args:
            context: Original context text
            response: Response text to visualize
            show_legend: Whether to include LaTeX comments with legend
            min_fragment_length: Minimum fragment length for green highlighting

        Returns:
            LaTeX-formatted string with highlight commands
        """
        context_tokens = self.tokenizer(context)
        response_tokens = self.tokenizer(response)

        if len(response_tokens) == 0:
            return "Empty response, cannot render LaTeX heatmap."

        fragments = self._find_extractive_fragments(context_tokens, response_tokens)

        # Build fragment position map
        fragment_map = {}
        response_tokens_norm = self._normalize_tokens(response_tokens)
        context_tokens_norm = self._normalize_tokens(context_tokens)
        used_positions = set()

        for fragment in fragments:
            fragment_norm = self._normalize_tokens(fragment)
            fragment_length = len(fragment)

            for i in range(len(response_tokens_norm) - len(fragment_norm) + 1):
                if (i not in used_positions and
                    response_tokens_norm[i:i+len(fragment_norm)] == fragment_norm):

                    if all(j not in used_positions for j in range(i, i + len(fragment_norm))):
                        for j in range(len(fragment_norm)):
                            fragment_map[i + j] = fragment_length
                            used_positions.add(i + j)
                        break

        # Build token highlight map
        token_char_map = self._build_token_char_mapping(response, response_tokens)
        token_highlight_map = {}

        for i, token in enumerate(response_tokens):
            if i in fragment_map:
                fragment_length = fragment_map[i]
                token_highlight_map[i] = 'green' if fragment_length >= min_fragment_length else None
            else:
                token_norm = self._normalize_tokens([token])[0]
                should_highlight_red = token_norm not in context_tokens_norm
                token_highlight_map[i] = 'red' if should_highlight_red else None

        # Merge consecutive highlights
        result_chars = []
        current_highlight = None
        highlight_buffer = []

        for char_index, char in enumerate(response):
            token_index = token_char_map.get(char_index, -1)
            target_highlight = token_highlight_map.get(token_index, None) if token_index >= 0 else None

            if target_highlight != current_highlight:
                # Flush previous highlight
                if current_highlight == 'red' and highlight_buffer:
                    result_chars.append(f"\\hlr{{{''.join(highlight_buffer)}}}")
                    highlight_buffer = []
                elif current_highlight == 'green' and highlight_buffer:
                    result_chars.append(f"\\hlg{{{''.join(highlight_buffer)}}}")
                    highlight_buffer = []
                current_highlight = target_highlight

            if current_highlight is not None:
                highlight_buffer.append(char)
            else:
                result_chars.append(char)

        # Flush final highlight
        if current_highlight == 'red' and highlight_buffer:
            result_chars.append(f"\\hlr{{{''.join(highlight_buffer)}}}")
        elif current_highlight == 'green' and highlight_buffer:
            result_chars.append(f"\\hlg{{{''.join(highlight_buffer)}}}")

        result = ''.join(result_chars)

        # Merge consecutive same-type highlights
        def merge_highlights(result_text, highlight_type):
            pattern = rf'(\\{highlight_type}\{{[^}}]*\}})\s+(\\{highlight_type}\{{[^}}]*\}})'
            while re.search(pattern, result_text):
                result_text = re.sub(pattern, rf'\\{highlight_type}{{\1 \2}}', result_text)
                result_text = re.sub(
                    rf'\\{highlight_type}\{{\\{highlight_type}\{{([^}}]*)\}}\s+\\{highlight_type}\{{([^}}]*)\}}}}',
                    rf'\\{highlight_type}{{\1 \2}}',
                    result_text
                )
            return result_text

        result = merge_highlights(result, 'hlr')
        result = merge_highlights(result, 'hlg')

        if show_legend:
            legend = "\n\n% LaTeX Heatmap Legend\n"
            legend += "% Add these commands to your LaTeX preamble:\n"
            legend += "% \\newcommand{\\hlr}[1]{\\textcolor{red}{#1}}\n"
            legend += "% \\newcommand{\\hlg}[1]{\\textcolor{green}{#1}}\n"
            legend += f"%\n"
            legend += "% Highlight rules:\n"
            legend += "% - \\hlr{}: External knowledge (not in context)\n"
            legend += f"% - \\hlg{{}}: Reliable extracts (length>={min_fragment_length})\n"
            result += legend

        return result

    def calculate_extractiveness_score_with_heatmap(
        self,
        context: str,
        response: str,
        show_heatmap: bool = True
    ) -> Tuple[float, float]:
        """Calculate score and optionally render heatmap"""
        coverage, density, score = self.calculate_extractiveness_score(context, response)
        print(f"Coverage: {coverage:.3f}, Density: {density:.3f}, Score: {score:.3f}")
        if show_heatmap:
            print(self.render_extractiveness_heatmap(context, response, show_legend=True))
        return coverage, density

    def batch_calculate_metrics(
        self,
        data_pairs: List[Tuple[str, str]],
        show_progress: bool = True
    ) -> dict:
        """
        Batch calculate metrics for multiple (context, response) pairs.

        Args:
            data_pairs: List of (context, response) tuples
            show_progress: Whether to show progress information

        Returns:
            Dictionary with 'coverage', 'density', 'failed_indices', 'total_samples', 'success_count'
        """
        coverages = []
        densities = []
        failed_indices = []

        total = len(data_pairs)

        if show_progress:
            print(f"Starting batch calculation for {total} samples...")

        for i, (context, response) in enumerate(data_pairs):
            try:
                coverage, density = self.calculate_metrics(context, response)
                coverages.append(coverage)
                densities.append(density)

                if show_progress and (i + 1) % max(1, total // 10) == 0:
                    print(f"Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Sample {i} failed: {str(e)}")
                failed_indices.append(i)
                coverages.append(float('nan'))
                densities.append(float('nan'))

        if show_progress:
            success_count = total - len(failed_indices)
            print(f"Batch calculation complete! Success: {success_count}/{total}, Failed: {len(failed_indices)}")

        return {
            'coverage': coverages,
            'density': densities,
            'failed_indices': failed_indices,
            'total_samples': total,
            'success_count': total - len(failed_indices)
        }

    def plot_density_heatmap(
        self,
        coverage_data: List[float],
        density_data: List[float],
        title: str = "Extractiveness Density Heatmap",
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None,
        show_plot: bool = True,
        dataset_name: Optional[str] = None,
        colormap: str = 'Greens',
        levels: int = 20,
        fixed_xlim: Optional[Tuple[float, float]] = None,
        fixed_ylim: Optional[Tuple[float, float]] = None
    ):
        """
        Generate 2D kernel density estimation heatmap for coverage vs density.

        Requires matplotlib and scipy to be installed.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib and scipy are required for density heatmap generation.\n"
                "Install with: pip install matplotlib scipy"
            )

        if np is None:
            raise ImportError("numpy is required for density heatmap generation")

        # Filter NaN values
        valid_indices = [
            i for i in range(len(coverage_data))
            if not (np.isnan(coverage_data[i]) or np.isnan(density_data[i]))
        ]

        if len(valid_indices) == 0:
            raise ValueError("No valid data points available for plotting")

        coverage_clean = [coverage_data[i] for i in valid_indices]
        density_clean = [density_data[i] for i in valid_indices]

        coverage_array = np.array(coverage_clean)
        density_array = np.array(density_clean)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Set axis ranges
        if fixed_xlim is not None:
            x_min, x_max = fixed_xlim
        else:
            x_min, x_max = max(0, coverage_array.min() - 0.05), min(1, coverage_array.max() + 0.05)

        if fixed_ylim is not None:
            y_min, y_max = fixed_ylim
        else:
            y_min, y_max = max(0, density_array.min() - 0.1), density_array.max() + 0.1

        # Create grid
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        # Calculate KDE
        try:
            values = np.vstack([coverage_array, density_array])
            kernel = gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)

            contourf = ax.contourf(xx, yy, f, levels=levels, cmap=colormap, alpha=0.8)
            contour = ax.contour(xx, yy, f, levels=levels//2, colors='white', alpha=0.5, linewidths=0.5)
            cbar = plt.colorbar(contourf, ax=ax)
            cbar.set_label('Density', rotation=270, labelpad=20)

        except Exception as e:
            if self.verbose:
                print(f"KDE failed, falling back to scatter plot: {str(e)}")
            ax.scatter(coverage_array, density_array, alpha=0.6, s=10)

        # Labels and title
        ax.set_xlabel('Extractive Fragment Coverage', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

        # Add info text
        info_text = f"n = {len(coverage_clean):,}"
        if dataset_name:
            info_text = f"{dataset_name}\n{info_text}"

        mean_coverage = coverage_array.mean()
        mean_density = density_array.mean()
        if mean_density > 0:
            compression_ratio = mean_coverage / mean_density
            info_text += f"\nc = {compression_ratio:.1f}:1"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        if show_plot:
            plt.show()

        return fig

    def generate_batch_report(self, batch_results: dict, dataset_name: Optional[str] = None) -> dict:
        """Generate statistical report from batch results"""
        if np is None:
            raise ImportError("numpy is required for batch reporting")

        coverages = batch_results['coverage']
        densities = batch_results['density']

        valid_coverages = [c for c in coverages if not np.isnan(c)]
        valid_densities = [d for d in densities if not np.isnan(d)]

        if len(valid_coverages) == 0:
            return {"error": "No valid data for statistics"}

        coverage_array = np.array(valid_coverages)
        density_array = np.array(valid_densities)

        report = {
            'dataset_name': dataset_name or 'Unknown',
            'total_samples': batch_results['total_samples'],
            'valid_samples': len(valid_coverages),
            'failed_samples': len(batch_results['failed_indices']),
            'success_rate': len(valid_coverages) / batch_results['total_samples'],
            'coverage_stats': {
                'mean': float(coverage_array.mean()),
                'std': float(coverage_array.std()),
                'min': float(coverage_array.min()),
                'max': float(coverage_array.max()),
                'median': float(np.median(coverage_array)),
                'q25': float(np.percentile(coverage_array, 25)),
                'q75': float(np.percentile(coverage_array, 75))
            },
            'density_stats': {
                'mean': float(density_array.mean()),
                'std': float(density_array.std()),
                'min': float(density_array.min()),
                'max': float(density_array.max()),
                'median': float(np.median(density_array)),
                'q25': float(np.percentile(density_array, 25)),
                'q75': float(np.percentile(density_array, 75))
            }
        }

        if report['density_stats']['mean'] > 0:
            report['compression_ratio'] = report['coverage_stats']['mean'] / report['density_stats']['mean']

        return report
