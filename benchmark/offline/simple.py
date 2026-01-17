import os

os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    llm = LLM("Qwen/Qwen3-0.6B")
    prompt_words = [
        "hello, introduce yourself!",
        "can you help me with my math problem?",
        "what is alibaba?",
    ]

    # KV cache compression: keep 50% of KV pairs using StreamingLLM strategy
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=100,
            kv_press_method="streaming_llm",  # Enable KV cache compression
            kv_press_ratio=0.5,
        )
        for _ in range(3)
    ]
    answers = llm.generate(prompts=prompt_words, sampling_params=sampling_params)
    for answer in answers:
        print(answer if isinstance(answer, str) else answer.get("text", "No text found"))
        print("\n\n\n")


if __name__ == "__main__":
    main()