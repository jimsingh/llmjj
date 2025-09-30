import argparse
from typing import Any

from .prompt_builder import PromptBuilder
from .text_generation import TextGenerator, build_text_generator

def rate_example(prompt_builder: PromptBuilder, example: dict[str, Any], generator: TextGenerator) -> dict[str, str]:
    prompt = prompt_builder.build(example)
    output = generator.generate(prompt, max_new_tokens=128, truncation=True)

    return {
        "query": example.get("query"),
        "document": example.get("document"),
        "prompt": prompt,
        "model_output": output,
    }


def run_eval(config_path: str, examples: list[dict[str, Any]], model_name: str = "gpt2") -> None:
    pb = PromptBuilder(config_path)

    generator = build_text_generator(model_name)

    results = []
    for ex in examples:
        res = rate_example(pb, ex, generator)
        results.append(res)

    for r in results:
        print("\n\n")
        print(f"Query: {r['query']}, Document: {r['document']}")
        print(f"Model output: {r['model_output']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/helpfulness.yaml")
    parser.add_argument("--model", default="mlx-community/gemma-3n-E4B-it-lm-4bit") #mlx-community/Qwen3-4B-Instruct-2507-DDWQ")
    args = parser.parse_args()

    examples = [
        {"query": "What is the population of New York City?", "document": "New York City's population is 8.4M people."},
        {"query": "Who wrote Hamlet?", "document": "Hamlet was written by William Shakespeare."},
        {"query": "Who wrote Hamlet?", "document": "King Lear was also a great play by Shakespeare."},
    ]

    run_eval(args.config, examples, args.model)
