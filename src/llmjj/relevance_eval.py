import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llmjj.prompt_builder import build_prompt, load_prompt_bundle
from llmjj.evaluation_report import generate_report


def _prepare_run_paths(base: Path) -> tuple[Path, Path]:
    if base.suffix:
        raise ValueError("--out must be a directory path")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "eval_results.jsonl"
    return run_dir, results_path


def load_ms_marco_streaming(configs: Sequence[str]):
    if not configs:
        raise ValueError("At least one ms_marco config must be provided")
    last_err: Exception | None = None
    for idx, cfg in enumerate(configs):
        try:
            return load_dataset("ms_marco", cfg, split="validation", streaming=True)
        except Exception as err:  # pragma: no cover - fallback exercised in tests
            last_err = err
            if idx < len(configs) - 1:
                print(f"Failed to load ms_marco config {cfg}: {err}. Trying next config...")
            else:
                raise RuntimeError(
                    f"Failed to load ms_marco dataset for configs {tuple(configs)}. Last error: {err}"
                ) from err
    raise RuntimeError("No dataset configs provided for ms_marco")


def _resolve_hf_token(token: str | None) -> str | None:
    if token:
        return token
    for env_key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        env_val = os.getenv(env_key)
        if env_val:
            return env_val
    return None


def extract_json(response: str) -> dict:
    decoder = json.JSONDecoder()
    last_obj = None
    for match in re.finditer(r"{", response):
        try:
            obj, _ = decoder.raw_decode(response[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            last_obj = obj
    if last_obj is not None:
        return last_obj
    raise ValueError(f"No valid JSON found in model output: {response}")


def run_llm_eval(
    model,
    tokenizer,
    generation_cfg,
    prompt: str,
    *,
    max_new_tokens: int = 256,
) -> str:

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generation_kwargs = {
        "do_sample": getattr(generation_cfg, "do_sample", False),
        "num_beams": getattr(generation_cfg, "num_beams", None),
        "num_return_sequences": getattr(generation_cfg, "num_return_sequences", None),
        "early_stopping": getattr(generation_cfg, "early_stopping", None),
        "max_new_tokens": max_new_tokens,
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=generation_cfg,
            **{k: v for k, v in generation_kwargs.items() if v is not None},
        )

    generated_sequence = outputs[0]
    input_len = inputs["input_ids"].shape[-1]
    generated_part = generated_sequence[input_len:]
    return tokenizer.decode(generated_part, skip_special_tokens=True, skip_prompt=False).strip()


def run_eval(
    model,
    tokenizer,
    generation_cfg,
    ds,
    prompt_cfg,
    fragments,
    results_path: Path,
    n: int = 200,
    debug: bool = False,
    prompts_source: str | None = None,
):
    results = []
    processed = 0
    skipped_in_row = 0
    progress = tqdm(total=n, desc="Evaluating")

    try:
        for ex in ds:
            if processed >= n:
                break

            query = ex["query"]
            p_texts = ex["passages"]["passage_text"]
            p_labels = ex["passages"]["is_selected"]
            pos_idxs = [idx for idx, lab in enumerate(p_labels) if lab == 1]
            neg_idxs = [idx for idx, lab in enumerate(p_labels) if lab == 0]
            if not pos_idxs or len(neg_idxs) < 2:
                skipped_in_row += 1
                if skipped_in_row >= 10:
                    raise RuntimeError(
                        "Skipped 10 consecutive MS MARCO examples without 1 positive and 2 negatives."
                    )
                continue

            skipped_in_row = 0
            chosen_idxs = [pos_idxs[0]] + neg_idxs[:2]
            for ci in chosen_idxs:
                passage = p_texts[ci]
                rel_label = p_labels[ci]
                prompt = build_prompt(
                    prompt_cfg,
                    fragments,
                    query=query,
                    docs=[passage],
                    query_id=ex["query_id"],
                )
                if debug:
                    print(
                        f"\nPrompt for query {ex['query_id']} passage {ci} (label={rel_label}):\n{prompt}"
                    )
                raw_out = run_llm_eval(model, tokenizer, generation_cfg, prompt)
                format_error = False
                try:
                    parsed = extract_json(raw_out)
                except ValueError as err:
                    warn_msg = (
                        "Failed to parse JSON for query "
                        f"{ex['query_id']} passage index {ci}: {err}. Using fallback rating 0."
                    )
                    if debug:
                        print(f"Model output:\n{raw_out}\nParse warning: {warn_msg}")
                    else:
                        print(warn_msg)
                    parsed = {
                        "id": ex["query_id"],
                        "type": "score",
                        "rating": 0,
                        "reason": warn_msg,
                    }
                    format_error = True
                parsed.setdefault("id", ex["query_id"])
                parsed.setdefault("type", "score")
                if debug:
                    parsed_json = json.dumps(parsed, ensure_ascii=False)
                    print(f"Model output:\n{raw_out}\nParsed JSON:\n{parsed_json}")
                try:
                    model_score = int(parsed.get("rating", 0))
                except (TypeError, ValueError):
                    warn_msg = (
                        "Model rating is not an integer in parsed JSON for query "
                        f"{ex['query_id']} passage index {ci}: {parsed.get('rating')}."
                    )
                    print(warn_msg)
                    model_score = 0
                    format_error = True
                record = dict(
                    query_id=ex["query_id"],
                    query=query,
                    passage=passage,
                    ground_truth=int(rel_label),
                    model_score=model_score,
                    model_output=raw_out,
                    format_error=format_error,
                )
                if prompts_source:
                    record["prompts_path"] = prompts_source
                results.append(record)

            processed += 1
            progress.update(1)
    finally:
        progress.close()

    if processed == 0:
        raise RuntimeError("No MS MARCO examples satisfied sampling constraints.")
    if processed < n:
        print(f"Processed {processed} examples (requested {n}).")

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return pd.DataFrame(results)


def build_generation_config(model) -> GenerationConfig:
    base_cfg = getattr(model, "generation_config", None)
    if base_cfg is None:
        base_cfg = GenerationConfig.from_model_config(model.config)

    cfg_dict = base_cfg.to_dict()
    cfg_dict["do_sample"] = False
    cfg_dict["num_beams"] = 4
    cfg_dict["num_return_sequences"] = 1
    cfg_dict["max_new_tokens"] = 256
    cfg_dict["early_stopping"] = True
    cfg_dict["top_p"] = None
    cfg_dict["top_k"] = None
    cfg_dict["temperature"] = None

    generation_cfg = GenerationConfig.from_dict(cfg_dict)
    generation_cfg.cache_implementation = None
    return generation_cfg


def _print_full_prompt(prompts_path: str) -> None:
    prompt_cfg, fragments = load_prompt_bundle(prompts_path, "score")

    sample_query = str(prompt_cfg.get("sample_query") or "What does YAML stand for?")
    sample_query_id = str(prompt_cfg.get("sample_query_id") or "SAMPLE-Q1")
    sample_doc = fragments.get("sample_document") or (
        "YAML (YAML Ain't Markup Language) is a human-friendly format used for configuration files."
    )

    prompt = build_prompt(
        prompt_cfg,
        fragments,
        query=sample_query,
        docs=[str(sample_doc)],
        query_id=sample_query_id,
    )
    print(prompt.rstrip("\n"))

def evaluate(
    prompts_path: str,
    model_name: str,
    out_path: Path,
    n: int,
    debug: bool = False,
    allow_v1_fallback: bool = False,
    hf_token: str | None = None,
):
    run_dir, results_path = _prepare_run_paths(Path(out_path))

    print("Loading dataset (streaming)...")
    configs: Sequence[str] = ["v2.1", "v1.1"] if allow_v1_fallback else ["v2.1"]
    ds = load_ms_marco_streaming(configs)

    print(f"Loading LLM {model_name}")
    auth_token = _resolve_hf_token(hf_token)
    auth_kwargs: dict[str, str] = {}
    if auth_token:
        auth_kwargs["token"] = auth_token
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)
    except OSError as err:
        if "gated repo" in str(err).lower() or "401" in str(err):
            raise RuntimeError(
                "Access to the requested model is gated. Provide a Hugging Face token via --hf-token or"
                " the HF_TOKEN/HUGGINGFACEHUB_API_TOKEN environment variables."
            ) from err
        raise
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            **auth_kwargs,
        )
    else:
        print("CUDA not available; forcing model to CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            **auth_kwargs,
        )
        model.to("cpu")
    model.eval()

    base_gen_cfg = getattr(model, "generation_config", None)
    if base_gen_cfg is not None:
        base_gen_cfg.do_sample = False
        base_gen_cfg.num_beams = 4
        base_gen_cfg.num_return_sequences = 1
        base_gen_cfg.max_new_tokens = 256
        base_gen_cfg.early_stopping = True
        base_gen_cfg.top_p = None
        base_gen_cfg.top_k = None
        base_gen_cfg.temperature = None

    generation_cfg = build_generation_config(model)

    prompt_cfg, fragments = load_prompt_bundle(prompts_path, "score")

    df = run_eval(
        model,
        tokenizer,
        generation_cfg,
        ds,
        prompt_cfg,
        fragments,
        results_path,
        n=n,
        debug=debug,
        prompts_source=str(Path(prompts_path)),
    )

    _write_metrics_artifacts(df, run_dir, prompts_source=str(Path(prompts_path)))


def _load_results_dataframe(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return pd.DataFrame(records)


def _write_metrics_artifacts(
    df: pd.DataFrame,
    run_dir: Path,
    *,
    prompts_source: str | None = None,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = generate_report(df, plots_dir=run_dir)

    confusion = metrics.get("confusion_matrix")
    summary_path = run_dir / "metrics_summary.json"
    total_examples = int(len(df))
    format_error_count = 0
    if "format_error" in df.columns:
        try:
            format_error_count = int(df["format_error"].fillna(False).astype(bool).sum())
        except Exception:
            format_error_count = int(df["format_error"].fillna(0).astype(int).sum())

    summary_payload = {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "auc": float(metrics.get("auc", 0.0)),
        "quadratic_kappa": float(metrics.get("quadratic_kappa", 0.0)),
        "total_examples": total_examples,
        "format_error_count": format_error_count,
    }
    if prompts_source is None and "prompts_path" in df.columns:
        first_path = df["prompts_path"].dropna()
        if not first_path.empty:
            prompts_source = str(first_path.iloc[0])
    summary_payload["prompts_path"] = prompts_source
    if confusion is not None:
        summary_payload["confusion_matrix"] = getattr(confusion, "tolist", lambda: confusion)()
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def regenerate_plots(
    results_path: Path,
    plots_dir: Path | None = None,
    *,
    prompts_source: str | None = None,
) -> None:
    results_path = Path(results_path)
    df = _load_results_dataframe(results_path)
    target_dir = Path(plots_dir) if plots_dir else results_path.parent
    _write_metrics_artifacts(df, target_dir, prompts_source=prompts_source)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=str,
        default="conf/prompts/helpfulness_score.yaml",
        help="Path to prompts.yaml",
    )
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Directory for run artifacts (a timestamped subdir will be created)",
    )
    parser.add_argument("-n", type=int, default=200, help="Number of queries to evaluate")
    parser.add_argument("--debug", action="store_true", help="Print model prompts and outputs")
    parser.add_argument(
        "--print-template",
        action="store_true",
        help="Print the resolved prompt template and exit",
    )
    parser.add_argument(
        "--print-full-prompt",
        action="store_true",
        help="Print a fully rendered prompt using the configured sample inputs and exit",
    )
    parser.add_argument(
        "--allow-v1-fallback",
        action="store_true",
        help="Also try ms_marco v1.1 if v2.1 cannot be streamed",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face access token (falls back to HF_TOKEN/HUGGINGFACEHUB_API_TOKEN)",
    )
    parser.add_argument(
        "--plots-from",
        type=Path,
        help="Path to eval_results.jsonl to regenerate metrics and plots",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Directory to write regenerated plots/metrics (defaults to results parent)",
    )
    parser.add_argument(
        "--plots-prompts-path",
        type=str,
        help="Prompt YAML path to record when regenerating plots (optional)",
    )
    return parser


def main(args: tuple[str, ...] | None = None):
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)
    if parsed.print_template:
        prompt_cfg, _ = load_prompt_bundle(parsed.prompts, "score")
        template_text = prompt_cfg.get("template") or ""
        print(template_text.rstrip("\n"))
        return
    if parsed.print_full_prompt:
        _print_full_prompt(parsed.prompts)
        return
    if parsed.plots_from:
        regenerate_plots(
            parsed.plots_from,
            parsed.plots_dir,
            prompts_source=parsed.plots_prompts_path,
        )
        return
    evaluate(
        prompts_path=parsed.prompts,
        model_name=parsed.model,
        out_path=parsed.out,
        n=parsed.n,
        debug=parsed.debug,
        allow_v1_fallback=parsed.allow_v1_fallback,
        hf_token=parsed.hf_token,
    )


if __name__ == "__main__":
    main()
