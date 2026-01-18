"""
Test KV cache compression with all strategies, long prompts, and different batch sizes.
"""

import argparse
import time
from typing import List, Optional

from minisgl.core import SamplingParams
from minisgl.kvcache.press import SUPPORTED_PRESS
from minisgl.llm import LLM

# All supported compression strategies
ALL_STRATEGIES = list(SUPPORTED_PRESS.keys())

# Long document for generating very long prompts
LONG_DOCUMENT = """
The History and Future of Artificial Intelligence: A Comprehensive Overview

Chapter 1: The Origins of AI

The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of
artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of
modern AI were planted by philosophers who attempted to describe the process of human thinking
as the mechanical manipulation of symbols. This work culminated in the invention of the programmable
digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during
the summer of 1956. The attendees became the leaders of AI research for decades. Many of them
predicted that a machine as intelligent as a human being would exist in no more than a generation,
and they were given millions of dollars to make this vision come true.

Eventually, it became obvious that commercial developers and researchers had grossly underestimated
the difficulty of the project. In 1974, in response to the criticism from James Lighthill and
ongoing pressure from congress, the U.S. and British Governments stopped funding undirected
research into artificial intelligence, and the difficult years that followed would later be
known as an "AI winter".

Chapter 2: The Rise of Machine Learning

In the early 1980s, AI research was revived by the commercial success of expert systems, a form of
AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market
for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer
project inspired the U.S and British governments to restore funding for academic research.

However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into
disrepute, and a second, longer-lasting winter began. Many researchers began to doubt that the
symbolic approach would be able to imitate all the processes of human cognition, especially
perception, robotics, learning and pattern recognition.

The development of metal-oxide-semiconductor (MOS) very-large-scale integration (VLSI), in the form
of complementary MOS (CMOS) transistor technology, enabled the development of practical artificial
neural network (ANN) technology in the 1980s. A landmark publication in the field was the 1989 book
Analog VLSI Implementation of Neural Systems by Carver Mead and Mohammed Ismail.

Chapter 3: Deep Learning Revolution

In the late 2000s and early 2010s, breakthroughs in deep learning transformed the field. The
availability of large datasets, increased computational power through GPUs, and algorithmic
improvements led to significant advances in computer vision, speech recognition, and natural
language processing.

In 2012, a deep neural network achieved a dramatic improvement in image classification accuracy
in the ImageNet Large Scale Visual Recognition Challenge. This event, often referred to as the
"AlexNet moment," marked the beginning of the deep learning era. Subsequently, deep learning
methods achieved state-of-the-art results in various AI tasks.

The introduction of the Transformer architecture in 2017 revolutionized natural language processing.
Unlike previous sequence models that processed data sequentially, Transformers use self-attention
mechanisms to process entire sequences in parallel, enabling more efficient training on large
datasets. This architecture forms the foundation of modern large language models.

Chapter 4: Large Language Models

Large language models (LLMs) represent a significant advancement in AI capabilities. These models,
trained on vast amounts of text data, can generate human-like text, answer questions, write code,
and perform various language tasks with remarkable fluency.

The scaling of these models has revealed emergent capabilities that were not present in smaller
models. As models grow larger, they demonstrate improved reasoning abilities, better few-shot
learning, and more nuanced understanding of context. This has led to the development of models
with hundreds of billions of parameters.

The deployment of LLMs has raised important questions about safety, alignment, and societal impact.
Researchers are actively working on techniques to make these models more reliable, truthful, and
aligned with human values. This includes work on constitutional AI, reinforcement learning from
human feedback (RLHF), and various forms of model evaluation and red-teaming.

Chapter 5: Current Challenges in AI

Despite remarkable progress, AI systems face several fundamental challenges. One major challenge
is the brittleness of current systems - they can fail unexpectedly when faced with inputs that
differ from their training distribution. This lack of robustness limits their deployment in
safety-critical applications.

Another challenge is the computational cost of training and running large models. The energy
consumption and carbon footprint of AI systems have become significant concerns. Researchers
are exploring more efficient architectures, quantization techniques, and other methods to
reduce the computational requirements of AI systems.

The challenge of AI alignment - ensuring that AI systems pursue goals that are beneficial to
humanity - remains an active area of research. As AI systems become more capable, the importance
of solving this challenge grows. Various approaches are being explored, including value learning,
interpretability research, and formal verification methods.

Chapter 6: The Future of AI

Looking ahead, several trends are likely to shape the future of AI. Continued scaling of models
may unlock new capabilities, though the returns from scaling alone may diminish. Researchers are
exploring new architectures, training methods, and ways to incorporate external knowledge and
tools into AI systems.

Multimodal AI systems that can process and generate multiple types of data - text, images, audio,
video - are becoming increasingly sophisticated. These systems may lead to more general AI
capabilities and new applications in areas like robotics and human-computer interaction.

The integration of AI into scientific research is accelerating discoveries in fields ranging
from biology to physics. AI systems are helping to design new molecules, predict protein
structures, and analyze complex datasets. This trend is likely to continue and expand into
new domains.

Conclusion

The field of artificial intelligence has come a long way since its founding in 1956. From early
symbolic systems to modern deep learning and large language models, AI has repeatedly exceeded
expectations while also revealing new challenges. As we continue to develop more capable AI
systems, careful attention to safety, ethics, and societal impact will be essential to ensure
that AI benefits humanity.

The journey of AI is far from over. New breakthroughs are likely to come from unexpected
directions, and the ultimate potential of artificial intelligence remains to be seen. What is
certain is that AI will continue to be one of the most important and transformative technologies
of our time.
"""

# Technical content for variety
CODE_DOCUMENT = """
Advanced Data Structures and Algorithms: A Technical Deep Dive

Section 1: Tree-Based Data Structures

Binary search trees (BSTs) are fundamental data structures that maintain sorted data and allow
for efficient search, insertion, and deletion operations. In a balanced BST, these operations
take O(log n) time. However, in the worst case (when the tree becomes skewed), operations can
degrade to O(n).

Self-balancing trees like AVL trees and Red-Black trees address this issue by automatically
maintaining balance after insertions and deletions. AVL trees maintain a stricter balance
(the heights of the two child subtrees of any node differ by at most one), while Red-Black
trees use a more relaxed balance criterion that results in faster insertions and deletions.

B-trees and B+ trees are commonly used in database systems and file systems. These trees can
have more than two children per node, which reduces the height of the tree and minimizes disk
I/O operations. B+ trees store all data in leaf nodes and maintain a linked list between leaves,
enabling efficient range queries.

Here is an implementation of a Red-Black tree node:

```python
class RBNode:
    def __init__(self, key, color='red'):
        self.key = key
        self.color = color  # 'red' or 'black'
        self.left = None
        self.right = None
        self.parent = None

    def grandparent(self):
        if self.parent is None:
            return None
        return self.parent.parent

    def sibling(self):
        if self.parent is None:
            return None
        if self == self.parent.left:
            return self.parent.right
        return self.parent.left

    def uncle(self):
        if self.parent is None:
            return None
        return self.parent.sibling()
```

Section 2: Graph Algorithms

Graph algorithms are essential for solving problems involving networks, relationships, and
connectivity. Depth-first search (DFS) and breadth-first search (BFS) are foundational
algorithms that explore graphs systematically.

Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices in
a weighted graph with non-negative edge weights. The algorithm uses a priority queue and
runs in O((V + E) log V) time with a binary heap.

```python
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
```

The Bellman-Ford algorithm handles graphs with negative edge weights and can detect negative
cycles. Floyd-Warshall algorithm computes shortest paths between all pairs of vertices.

Section 3: Dynamic Programming

Dynamic programming is a method for solving complex problems by breaking them down into simpler
subproblems. It is applicable when subproblems overlap and have optimal substructure.

The key insight of dynamic programming is to store the results of subproblems to avoid redundant
computation. This can be done through memoization (top-down) or tabulation (bottom-up).

Consider the problem of finding the longest common subsequence (LCS) of two strings:

```python
def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

Section 4: String Algorithms

String matching algorithms are crucial for text processing, bioinformatics, and search engines.
The naive approach has O(nm) time complexity, where n is the text length and m is the pattern
length.

The Knuth-Morris-Pratt (KMP) algorithm improves this to O(n + m) by preprocessing the pattern
to create a failure function that allows the algorithm to skip characters when a mismatch occurs.

```python
def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    lps = compute_lps(pattern)
    matches = []

    i = j = 0
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return matches
```

Section 5: Advanced Topics

Suffix arrays and suffix trees enable efficient string operations like substring search,
longest repeated substring, and longest common prefix queries. These data structures are
fundamental in bioinformatics for genome analysis.

Bloom filters are probabilistic data structures that test whether an element is a member of
a set. They can have false positives but never false negatives, making them useful for
caching and distributed systems.

Skip lists are probabilistic data structures that allow O(log n) search, insertion, and
deletion operations. They are simpler to implement than balanced trees and are used in
systems like Redis and LevelDB.

Conclusion

Understanding these data structures and algorithms is essential for writing efficient software.
The choice of data structure can have a significant impact on program performance, and knowing
the trade-offs between different approaches allows developers to make informed decisions.
"""


def generate_long_prompt(target_tokens: int, prompt_id: int = 0) -> str:
    """Generate a prompt with approximately the target number of tokens."""
    # Approximate: 1 token ~= 4 characters for English text
    target_chars = target_tokens * 4

    documents = [LONG_DOCUMENT, CODE_DOCUMENT]
    base_doc = documents[prompt_id % len(documents)]

    # Repeat the document to reach target length
    repeated = base_doc
    while len(repeated) < target_chars:
        repeated += "\n\n" + base_doc

    # Truncate to target length
    content = repeated[:target_chars]

    # Add a question at the end
    questions = [
        "\n\nBased on the above content, please provide a brief summary of the main points.",
        "\n\nWhat are the key takeaways from the text above? Please be concise.",
        "\n\nPlease explain the most important concept mentioned in the text above.",
        "\n\nSummarize the technical content described above in 2-3 sentences.",
    ]

    return content + questions[prompt_id % len(questions)]


def create_prompts(num_prompts: int, tokens_per_prompt: int) -> List[str]:
    """Create a list of long prompts."""
    return [generate_long_prompt(tokens_per_prompt, i) for i in range(num_prompts)]


def test_compression_strategy(
    llm: LLM,
    prompts: List[str],
    strategy: Optional[str],
    compression_ratio: float,
    max_tokens: int = 50,
) -> dict:
    """Test a specific compression strategy and return timing info."""
    strategy_name = strategy if strategy else "none (baseline)"
    print(f"\n{'=' * 70}")
    print(f"Strategy: {strategy_name:<25} | Compression ratio: {compression_ratio:.2f}")
    print("=" * 70)

    sampling_params = [
        SamplingParams(
            temperature=0.7,
            ignore_eos=True,
            max_tokens=max_tokens,
            kv_press_method=strategy,
            kv_press_ratio=compression_ratio,
        )
        for _ in range(len(prompts))
    ]

    start_time = time.perf_counter()
    answers = llm.generate(prompts=prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - start_time

    # Print first response preview
    if answers:
        first_answer = answers[0]
        text = first_answer if isinstance(first_answer, str) else first_answer.get("text", "")
        preview = text[:150] + "..." if len(text) > 150 else text
        print(f"Sample output: {preview}")

    print(f"Time: {elapsed:.2f}s | Throughput: {len(prompts) / elapsed:.2f} req/s")

    return {
        "strategy": strategy_name,
        "compression_ratio": compression_ratio,
        "num_prompts": len(prompts),
        "elapsed_time": elapsed,
        "throughput": len(prompts) / elapsed,
    }


def run_benchmark(
    llm: LLM,
    prompt_length: int,
    batch_sizes: List[int],
    strategies: List[str],
    compression_ratios: List[float],
    max_output_tokens: int,
):
    """Run the full benchmark suite."""
    results = []

    for batch_size in batch_sizes:
        print(f"\n{'#' * 70}")
        print(f"# BATCH SIZE: {batch_size} | PROMPT LENGTH: ~{prompt_length} tokens")
        print("#" * 70)

        prompts = create_prompts(batch_size, prompt_length)
        print(f"Created {len(prompts)} prompts, each ~{len(prompts[0]) // 4} tokens")

        # Baseline (no compression)
        result = test_compression_strategy(llm, prompts, None, 1.0, max_output_tokens)
        result["batch_size"] = batch_size
        result["prompt_length"] = prompt_length
        results.append(result)

        # Test each strategy with each compression ratio
        for strategy in strategies:
            for ratio in compression_ratios:
                try:
                    result = test_compression_strategy(
                        llm, prompts, strategy, ratio, max_output_tokens
                    )
                    result["batch_size"] = batch_size
                    result["prompt_length"] = prompt_length
                    results.append(result)
                except Exception as e:
                    print(f"Error with {strategy} (ratio={ratio}): {e}")

    return results


def print_summary(results: List[dict]):
    """Print a summary table of results."""
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Strategy':<25} {'Ratio':>6} {'Batch':>6} {'Time(s)':>10} {'Throughput':>12}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['strategy']:<25} {r['compression_ratio']:>6.2f} {r['batch_size']:>6} "
            f"{r['elapsed_time']:>10.2f} {r['throughput']:>12.2f}"
        )

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Test KV cache compression strategies")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model to use")
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=2048,
        help="Approximate prompt length in tokens",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Strategies to test (default: all)",
    )
    parser.add_argument(
        "--compression-ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.3],
        help="Compression ratios to test",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=50,
        help="Maximum output tokens per request",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer strategies and batch sizes",
    )

    args = parser.parse_args()

    # Set strategies
    if args.strategies:
        strategies = args.strategies
    elif args.quick:
        # Quick test: only a few representative strategies
        strategies = ["streaming_llm", "l2_norm", "tova", "lag_kv", "compactor"]
    else:
        # Full test: all strategies
        strategies = ALL_STRATEGIES

    # Quick mode adjustments
    if args.quick:
        batch_sizes = [1, 4]
        compression_ratios = [0.5]
    else:
        batch_sizes = args.batch_sizes
        compression_ratios = args.compression_ratios

    print("=" * 70)
    print("KV CACHE COMPRESSION BENCHMARK")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Prompt length: ~{args.prompt_length} tokens")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Compression ratios: {compression_ratios}")
    print(f"Strategies to test ({len(strategies)}):")
    for i, s in enumerate(strategies):
        print(f"  {i + 1:2d}. {s}")
    print("=" * 70)

    print("\nLoading model...")
    llm = LLM(args.model)

    results = run_benchmark(
        llm=llm,
        prompt_length=args.prompt_length,
        batch_sizes=batch_sizes,
        strategies=strategies,
        compression_ratios=compression_ratios,
        max_output_tokens=args.max_output_tokens,
    )

    print_summary(results)

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
