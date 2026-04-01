"""Needle-in-a-Haystack evaluation for long-context models.

A synthetic benchmark that inserts a specific fact (the *needle*) at varying
positions inside a long context (the *haystack*) and checks whether the model
can retrieve the needle when asked.

Usage example with TurboQuant KV-cache quantization::

    uv run -m mlx_vlm.evals.needle_in_haystack \
        --model "mlx-community/Qwen3.5-35B-A3B-4bit" \
        --kv-bits 3.5 \
        --kv-quant-scheme turboquant \
        --context-lengths 1024 2048 4096 8192 \
        --depth-percents 0 25 50 75 100
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

from tqdm import tqdm

from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

from .utils import text_inference

# ---------------------------------------------------------------------------
# Default needle / haystack content
# ---------------------------------------------------------------------------
DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich "
    "and sit in Dolores Park on a sunny day."
)

DEFAULT_RETRIEVAL_QUESTION = (
    "What is the best thing to do in San Francisco? "
    "Answer with the exact sentence from the context."
)

# A collection of filler sentences used to build the haystack.  They are
# intentionally generic so that the needle stands out.
FILLER_SENTENCES = [
    "The grass is green in the morning light.",
    "Clouds moved across the sky in shifting patterns.",
    "A river flowed gently through the valley below.",
    "Many types of birds can be found in forests around the world.",
    "The history of architecture reveals much about civilizations.",
    "Warm bread from the oven fills the kitchen with a pleasant aroma.",
    "Astronomy is one of the oldest natural sciences.",
    "Libraries have long been places of learning and community.",
    "Trains travel along tracks connecting cities and towns.",
    "Mountains provide habitats for a wide variety of species.",
    "Ocean currents influence weather patterns globally.",
    "Music has been part of human culture for thousands of years.",
    "Bridges are engineering feats that connect separated landmasses.",
    "Sunlight filters through the canopy of a dense forest.",
    "Ceramic pottery dates back to prehistoric times.",
    "Wind turbines convert kinetic energy into electrical energy.",
    "The study of languages reveals the evolution of human thought.",
    "Gardening is a practice that combines science and art.",
    "Photography captures moments in time for posterity.",
    "The alphabet we use today evolved over several millennia.",
]


# ---------------------------------------------------------------------------
# Haystack generation helpers
# ---------------------------------------------------------------------------


def _build_haystack(
    target_token_count: int,
    needle: str,
    depth_percent: float,
    processor,
) -> str:
    """Build a haystack string of approximately *target_token_count* tokens
    with the *needle* inserted at *depth_percent* (0–100) through the text.
    """
    tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
    )

    # Estimate tokens-per-filler sentence to decide how many to use.
    sample = " ".join(FILLER_SENTENCES[:5])
    sample_tokens = len(tokenizer.encode(sample, add_special_tokens=False))
    avg_tokens_per_sentence = sample_tokens / 5

    n_sentences = max(
        1, int(target_token_count / avg_tokens_per_sentence) + 10
    )

    sentences = [
        FILLER_SENTENCES[i % len(FILLER_SENTENCES)] for i in range(n_sentences)
    ]

    # Determine where to insert the needle.
    insert_idx = max(0, int(len(sentences) * depth_percent / 100))
    sentences.insert(insert_idx, needle)

    haystack = " ".join(sentences)

    # Trim to approximate target length.
    tokens = tokenizer.encode(haystack, add_special_tokens=False)
    if len(tokens) > target_token_count:
        tokens = tokens[:target_token_count]
        haystack = tokenizer.decode(tokens, skip_special_tokens=True)

    return haystack


def _check_retrieval(response: str, needle: str) -> bool:
    """Return *True* if the response contains the essential content of the
    needle (case-insensitive substring match on the core phrase).
    """
    response_lower = response.lower()
    # Extract key phrases from the needle for matching.
    key_phrases = [
        "sandwich",
        "dolores park",
        "san francisco",
    ]
    matches = sum(1 for phrase in key_phrases if phrase in response_lower)
    return matches >= 2


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Needle-in-a-Haystack evaluation for long-context models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HuggingFace repo id for the model",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional path for adapter weights",
    )

    # Haystack configuration
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="Context lengths (in tokens) to evaluate",
    )
    parser.add_argument(
        "--depth-percents",
        type=float,
        nargs="+",
        default=[0, 25, 50, 75, 100],
        help="Needle depth percentages (0=start, 100=end)",
    )
    parser.add_argument(
        "--needle",
        type=str,
        default=DEFAULT_NEEDLE,
        help="The needle sentence to hide in the haystack",
    )
    parser.add_argument(
        "--retrieval-question",
        type=str,
        default=DEFAULT_RETRIEVAL_QUESTION,
        help="Question to ask to retrieve the needle",
    )

    # KV cache quantization
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="KV cache quantization bits (e.g. 3.5 for TurboQuant)",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        default="uniform",
        help='KV quantization scheme: "uniform" or "turboquant"',
    )
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=5000)
    parser.add_argument("--prefill-step-size", type=int, default=2048)

    # Generation
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/needle_in_haystack",
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

    results = []
    correct = 0
    total = 0

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

    combos = [
        (ctx_len, depth)
        for ctx_len in args.context_lengths
        for depth in args.depth_percents
    ]

    for ctx_len, depth in tqdm(combos, desc="Evaluating"):
        haystack = _build_haystack(ctx_len, args.needle, depth, processor)

        prompt_text = (
            f"Below is a long passage of text. Read it carefully and then "
            f"answer the question.\n\n"
            f"--- START OF PASSAGE ---\n"
            f"{haystack}\n"
            f"--- END OF PASSAGE ---\n\n"
            f"Question: {args.retrieval_question}\n"
            f"Answer:"
        )

        prompt = apply_chat_template(
            processor, model.config, prompt_text, num_images=0
        )

        start_time = time.perf_counter()
        result = text_inference(
            model,
            processor,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
            **kv_kwargs,
        )
        elapsed = time.perf_counter() - start_time

        response = result.text.strip()
        is_correct = _check_retrieval(response, args.needle)
        if is_correct:
            correct += 1
        total += 1

        entry = {
            "context_length": ctx_len,
            "depth_percent": depth,
            "response": response,
            "correct": is_correct,
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "peak_memory_gb": result.peak_memory,
        }
        results.append(entry)

        if args.verbose:
            logging.info(
                f"ctx={ctx_len} depth={depth}% correct={is_correct} "
                f"time={elapsed:.1f}s"
            )

    # ---- Save results ----
    model_name = args.model.split("/")[-1]
    csv_path = output_dir / f"{model_name}_niah.csv"
    json_path = output_dir / f"{model_name}_niah.json"

    fieldnames = [
        "context_length",
        "depth_percent",
        "response",
        "correct",
        "elapsed_s",
        "prompt_tokens",
        "generation_tokens",
        "peak_memory_gb",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    accuracy = correct / total if total > 0 else 0
    summary = {
        "model": args.model,
        "kv_bits": args.kv_bits,
        "kv_quant_scheme": args.kv_quant_scheme,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "context_lengths": args.context_lengths,
        "depth_percents": args.depth_percents,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print summary ----
    print(f"\n{'='*80}")
    print("Needle-in-a-Haystack Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    if args.kv_bits is not None:
        print(f"KV Bits: {args.kv_bits}  Scheme: {args.kv_quant_scheme}")
    print(f"Total: {total}  Correct: {correct}  Accuracy: {accuracy*100:.2f}%")
    print(f"\n{'Context':>10} | ", end="")
    for d in args.depth_percents:
        print(f" {d:5.0f}%", end="")
    print()
    print("-" * (12 + 7 * len(args.depth_percents)))
    for ctx_len in args.context_lengths:
        print(f"{ctx_len:>10} | ", end="")
        for depth in args.depth_percents:
            match = next(
                (
                    r
                    for r in results
                    if r["context_length"] == ctx_len
                    and r["depth_percent"] == depth
                ),
                None,
            )
            symbol = "  ✓  " if match and match["correct"] else "  ✗  "
            print(f" {symbol}", end="")
        print()
    print(f"{'='*80}")
    print(f"Results saved to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
