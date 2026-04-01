"""LongBench-Summary evaluation for long-context models.

Evaluates models on summarization tasks from the LongBench benchmark using
ROUGE-L as the primary metric.

Dataset: ``THUDM/LongBench`` on HuggingFace (subsets ``multi_news``,
``gov_report``, ``qmsum``, ``multi_news_e``, ``gov_report_e``, ``vcsum``).

Usage example with TurboQuant KV-cache quantization::

    uv run -m mlx_vlm.evals.longbench_summary \
        --model "mlx-community/Qwen3.5-35B-A3B-4bit" \
        --kv-bits 3.5 \
        --kv-quant-scheme turboquant \
        --subsets gov_report multi_news qmsum
"""

import argparse
import csv
import json
import logging
import random
import re
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

from .utils import text_inference

# Summarization subsets in LongBench
SUMMARY_SUBSETS = [
    "multi_news",
    "gov_report",
    "qmsum",
    "multi_news_e",
    "gov_report_e",
    "vcsum",
]


# ---------------------------------------------------------------------------
# ROUGE-L helpers  (self-contained, no extra dependency required)
# ---------------------------------------------------------------------------


def _tokenize(text: str):
    """Simple whitespace + punctuation tokenization for ROUGE computation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return text.split()


def _lcs_length(x, y):
    """Length of the longest common subsequence between two token lists."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimised LCS.
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(prediction: str, reference: str) -> dict:
    """Compute ROUGE-L precision, recall, and F1."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SUBSET_PROMPTS = {
    "multi_news": (
        "You are given several news articles below. "
        "Write a concise summary that captures the key information.\n\n"
    ),
    "multi_news_e": (
        "You are given several news articles below. "
        "Write a concise summary that captures the key information.\n\n"
    ),
    "gov_report": (
        "You are given a government report below. "
        "Write a concise summary of the report.\n\n"
    ),
    "gov_report_e": (
        "You are given a government report below. "
        "Write a concise summary of the report.\n\n"
    ),
    "qmsum": (
        "You are given a meeting transcript below. "
        "Write a concise summary of the meeting.\n\n"
    ),
    "vcsum": (
        "下面是一段对话记录，请写一段简洁的摘要。\n\n"
    ),
}


def _build_prompt(sample: dict, subset: str) -> str:
    """Build the evaluation prompt for a single sample."""
    context = sample["context"]
    instruction = SUBSET_PROMPTS.get(
        subset,
        "Summarize the following text.\n\n",
    )
    question = sample.get("input", "")
    if question:
        instruction += f"Focus on: {question}\n\n"
    return f"{instruction}{context}\n\nSummary:"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="LongBench-Summary evaluation for long-context models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HuggingFace repo id for the model",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="THUDM/LongBench",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["gov_report", "multi_news", "qmsum"],
        help="Summarization subsets to evaluate",
    )
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per subset (for debugging)",
    )

    # KV cache quantization
    parser.add_argument("--kv-bits", type=float, default=None)
    parser.add_argument("--kv-quant-scheme", type=str, default="uniform")
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=5000)
    parser.add_argument("--prefill-step-size", type=int, default=2048)

    # Generation
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results/longbench_summary",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Loading model from {args.model}")
    model, processor = load(
        args.model, adapter_path=args.adapter_path, trust_remote_code=True
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kv_kwargs = {}
    if args.kv_bits is not None:
        kv_kwargs.update(
            {
                "kv_bits": args.kv_bits,
                "kv_quant_scheme": args.kv_quant_scheme,
                "kv_group_size": args.kv_group_size,
                "quantized_kv_start": args.quantized_kv_start,
            }
        )
    if args.prefill_step_size is not None:
        kv_kwargs["prefill_step_size"] = args.prefill_step_size

    all_results = []
    subset_scores = {}

    for subset in args.subsets:
        if subset not in SUMMARY_SUBSETS:
            logging.warning(f"Skipping unknown subset: {subset}")
            continue

        logging.info(f"Loading subset {subset}")
        try:
            dataset = load_dataset(
                args.dataset,
                subset,
                split="test",
                streaming=args.streaming,
                trust_remote_code=True,
            )
        except Exception as e:
            logging.error(f"Failed to load subset {subset}: {e}")
            continue

        if args.max_samples and not args.streaming:
            dataset = dataset.select(
                range(min(args.max_samples, len(dataset)))
            )

        rouge_scores = []
        count = 0

        for idx, sample in enumerate(
            tqdm(dataset, desc=f"Evaluating {subset}")
        ):
            if args.max_samples and count >= args.max_samples:
                break

            try:
                prompt_text = _build_prompt(sample, subset)

                prompt = apply_chat_template(
                    processor, model.config, prompt_text, num_images=0
                )

                start_time = time.perf_counter()
                gen_result = text_inference(
                    model,
                    processor,
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    **kv_kwargs,
                )
                elapsed = time.perf_counter() - start_time

                response = gen_result.text.strip()
                reference = sample.get("answers", [""])[0] if isinstance(
                    sample.get("answers"), list
                ) else sample.get("answers", "")

                scores = rouge_l(response, reference)

                entry = {
                    "subset": subset,
                    "index": idx,
                    "response": response,
                    "reference": reference[:200],
                    "rouge_l_f1": round(scores["f1"], 4),
                    "rouge_l_precision": round(scores["precision"], 4),
                    "rouge_l_recall": round(scores["recall"], 4),
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": gen_result.prompt_tokens,
                    "generation_tokens": gen_result.generation_tokens,
                    "peak_memory_gb": gen_result.peak_memory,
                }
                all_results.append(entry)
                rouge_scores.append(scores["f1"])
                count += 1

                if args.verbose:
                    logging.info(
                        f"[{subset}][{idx}] ROUGE-L F1={scores['f1']:.4f} "
                        f"time={elapsed:.1f}s"
                    )

            except Exception as e:
                logging.error(
                    f"Error processing sample {idx} in {subset}: {e}"
                )
                continue

        if rouge_scores:
            avg_f1 = sum(rouge_scores) / len(rouge_scores)
        else:
            avg_f1 = 0.0
        subset_scores[subset] = {
            "count": len(rouge_scores),
            "avg_rouge_l_f1": round(avg_f1, 4),
        }

    # ---- Save results ----
    model_name = args.model.split("/")[-1]
    csv_path = output_dir / f"{model_name}_longbench_summary.csv"
    json_path = output_dir / f"{model_name}_longbench_summary.json"

    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    overall_f1 = (
        sum(s["avg_rouge_l_f1"] * s["count"] for s in subset_scores.values())
        / max(1, sum(s["count"] for s in subset_scores.values()))
    )
    summary = {
        "model": args.model,
        "kv_bits": args.kv_bits,
        "kv_quant_scheme": args.kv_quant_scheme,
        "subsets_evaluated": list(subset_scores.keys()),
        "subset_scores": subset_scores,
        "overall_rouge_l_f1": round(overall_f1, 4),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print summary ----
    print(f"\n{'='*80}")
    print("LongBench-Summary Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    if args.kv_bits is not None:
        print(f"KV Bits: {args.kv_bits}  Scheme: {args.kv_quant_scheme}")
    print(f"\n{'Subset':<20} {'Count':>6} {'ROUGE-L F1':>12}")
    print("-" * 40)
    for subset, scores in subset_scores.items():
        print(
            f"{subset:<20} {scores['count']:>6} "
            f"{scores['avg_rouge_l_f1']*100:>11.2f}%"
        )
    print("-" * 40)
    print(f"{'Overall':<20} {'':>6} {overall_f1*100:>11.2f}%")
    print(f"{'='*80}")
    print(f"Results saved to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
