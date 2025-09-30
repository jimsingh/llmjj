import argparse
import csv
from tqdm import tqdm
from typing import Any, Iterator

from .prompt_builder import PromptBuilder
from .text_generation import TextGenerator, build_text_generator
from .dataset import load_msmarco_dataset

def rate_example(prompt_builder: PromptBuilder, example: dict[str, str], generator: TextGenerator) -> dict[str, str]:
    prompt = prompt_builder.build(example)
    output = generator.generate(prompt, max_new_tokens=128, truncation=True)
    llm_reason, llm_label = prompt_builder.parse_response(output)

    return {
        "query": example.get("query", ""),
        "document": example.get("document", ""),
        "label": example.get("label", ""),
        "llm_reason": llm_reason,
        "llm_label": llm_label,
    }


def run_eval(config_path: str, items: list[dict[str, str]], model_name: str) -> Iterator[dict[str, str]]:
    pb = PromptBuilder(config_path)
    generator = build_text_generator(model_name)
    for item in items:
        res = rate_example(pb, item, generator)
        yield res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/helpfulness.yaml")
    parser.add_argument("--model", default="mlx-community/gemma-3n-E4B-it-lm-4bit")
    parser.add_argument("--dry-run", action="store_true", help="use hardcoded examples.")
    parser.add_argument("--num-examples", type=int, default=1000, help="number of ms marco examples to use")
    args = parser.parse_args()

    if args.dry_run:
        examples = [
            # Note: The dataset provides integer ratings, so I've updated these for consistency.
            {"query": "What is the population of New York City?", "document": "New York City's population is 8.4M people.", "human_rating": 5},
            {"query": "Who wrote Hamlet?", "document": "Hamlet was written by William Shakespeare.", "human_rating": 5},
            {"query": "Who wrote Hamlet?", "document": "King Lear was also a great play by Shakespeare.", "human_rating": 2},
        ]
    else:
        print(f"Processing {args.num_examples} examples from MS MARCO v2.1")
        ds = load_msmarco_dataset()
        
        num_to_select = min(args.num_examples, len(ds))
        examples = ds.select(range(num_to_select)).to_list()
    
    results_generator = run_eval(args.config, examples, args.model)

    fieldnames = ["query", "document", "label", "llm_reason", "llm_label"]

    with open('output.tsv', "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for r in tqdm(results_generator, total=len(examples), desc="Evaluating"):
            writer.writerow(r)
            print(
                f"{r['query']} | {r['document']} | {r['label']} | {r['llm_reason']} | {r['llm_label']}"
            )
        
