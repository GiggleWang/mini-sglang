"""
Test KV cache compression with different strategies and longer prompts.
"""

# import os

# os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"

from minisgl.core import SamplingParams
from minisgl.llm import LLM

# Long prompts for better compression testing
LONG_PROMPTS = [
    # ~100 tokens
    """Please read the following passage and answer the question at the end.

The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of
artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of
modern AI were planted by philosophers who attempted to describe the process of human thinking
as the mechanical manipulation of symbols. This work culminated in the invention of the programmable
digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.

Question: When was the programmable digital computer invented?""",

    # ~80 tokens
    """You are an expert programmer. Please review the following code and explain what it does:

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

Explain the algorithm step by step.""",

    # ~60 tokens
    """The following is a conversation between a user and an AI assistant. The assistant is helpful,
creative, clever, and very friendly. The assistant always provides detailed and accurate information.

User: Can you explain the theory of relativity in simple terms that a high school student could understand?
Assistant:""",
]


def test_compression_strategy(llm: LLM, strategy: str, compression_ratio: float):
    """Test a specific compression strategy."""
    print(f"\n{'='*60}")
    print(f"Testing: {strategy} (compression_ratio={compression_ratio})")
    print("=" * 60)

    sampling_params = [
        SamplingParams(
            temperature=0.7,
            ignore_eos=True,
            max_tokens=50,  # Shorter output for faster testing
            kv_press_method=strategy,
            kv_press_ratio=compression_ratio,
        )
        for _ in range(len(LONG_PROMPTS))
    ]

    answers = llm.generate(prompts=LONG_PROMPTS, sampling_params=sampling_params)

    for i, answer in enumerate(answers):
        text = answer if isinstance(answer, str) else answer.get("text", "No text found")
        print(f"\n--- Response {i+1} (first 200 chars) ---")
        print(text[:200] + "..." if len(text) > 200 else text)


def test_no_compression(llm: LLM):
    """Baseline test without compression."""
    print(f"\n{'='*60}")
    print("Testing: No compression (baseline)")
    print("=" * 60)

    sampling_params = [
        SamplingParams(
            temperature=0.7,
            ignore_eos=True,
            max_tokens=50,
        )
        for _ in range(len(LONG_PROMPTS))
    ]

    answers = llm.generate(prompts=LONG_PROMPTS, sampling_params=sampling_params)

    for i, answer in enumerate(answers):
        text = answer if isinstance(answer, str) else answer.get("text", "No text found")
        print(f"\n--- Response {i+1} (first 200 chars) ---")
        print(text[:200] + "..." if len(text) > 200 else text)


def main():
    print("Loading model...")
    llm = LLM("Qwen/Qwen3-0.6B")

    # Test 1: No compression (baseline)
    test_no_compression(llm)

    # Test 2: StreamingLLM with different compression ratios
    test_compression_strategy(llm, "streaming_llm", compression_ratio=0.7)
    test_compression_strategy(llm, "streaming_llm", compression_ratio=0.5)
    test_compression_strategy(llm, "streaming_llm", compression_ratio=0.3)

    # Test 3: L2 Norm based compression
    test_compression_strategy(llm, "l2_norm", compression_ratio=0.5)

    # Test 4: Random compression (for comparison)
    test_compression_strategy(llm, "random", compression_ratio=0.5)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
