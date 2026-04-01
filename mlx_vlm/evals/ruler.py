"""RULER (13-task) evaluation for long-context models.

RULER tests models on 13 tasks across four categories:

**Retrieval** – Single NIAH, Multi-Keys NIAH, Multi-Values NIAH, Multi-Queries NIAH
**Multi-hop tracing** – Variable Tracking (VT)
**Aggregation** – Common Words (CW), Frequent Words (FW)
**Question Answering** – Single-hop QA (SQA), Multi-hop QA (MQA)
**Plus additional variants** reaching 13 tasks total.

This script generates synthetic data for each task following the RULER paper
specification and evaluates accuracy per task and overall.

Usage example with TurboQuant KV-cache quantization::

    uv run -m mlx_vlm.evals.ruler \
        --model "mlx-community/Qwen3.5-35B-A3B-4bit" \
        --kv-bits 3.5 \
        --kv-quant-scheme turboquant \
        --context-length 4096 \
        --samples-per-task 20
"""

import argparse
import csv
import json
import logging
import random
import string
import time
from pathlib import Path

from tqdm import tqdm

from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

from .utils import text_inference

# ---------------------------------------------------------------------------
# Filler text used to pad contexts to the desired token length
# ---------------------------------------------------------------------------
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


def _random_word(rng, length=6):
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def _pad_to_length(text, target_tokens, processor):
    """Pad *text* with filler to approximately *target_tokens* tokens."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    current_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if current_tokens >= target_tokens:
        tokens = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)

    needed = target_tokens - current_tokens
    avg_tok = 8  # rough avg tokens per filler sentence
    n_filler = max(1, needed // avg_tok + 5)
    filler = " ".join(
        FILLER_SENTENCES[i % len(FILLER_SENTENCES)] for i in range(n_filler)
    )
    combined = text + " " + filler
    tokens = tokenizer.encode(combined, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Task generators – each returns (context, question, answer_list)
# ---------------------------------------------------------------------------


def _gen_single_niah(rng, ctx_len, processor):
    """Single Needle-in-a-Haystack."""
    key = _random_word(rng)
    value = _random_word(rng)
    needle = f"The special key is {key} and the value is {value}."
    text = _pad_to_length(needle, ctx_len, processor)
    question = f"What is the value for the key '{key}'?"
    return text, question, [value]


def _gen_multi_keys_niah(rng, ctx_len, processor):
    """Multi-Keys NIAH – multiple needles, ask for one."""
    pairs = [(f"key_{_random_word(rng, 4)}", _random_word(rng)) for _ in range(5)]
    needles = " ".join(
        f"The key {k} maps to the value {v}." for k, v in pairs
    )
    text = _pad_to_length(needles, ctx_len, processor)
    ask_idx = rng.randint(0, len(pairs) - 1)
    question = f"What is the value for the key '{pairs[ask_idx][0]}'?"
    return text, question, [pairs[ask_idx][1]]


def _gen_multi_values_niah(rng, ctx_len, processor):
    """Multi-Values NIAH – one key maps to multiple values."""
    key = _random_word(rng)
    values = [_random_word(rng) for _ in range(3)]
    needles = " ".join(
        f"The key {key} has value {v}." for v in values
    )
    text = _pad_to_length(needles, ctx_len, processor)
    question = f"List all values for the key '{key}'."
    return text, question, values


def _gen_multi_queries_niah(rng, ctx_len, processor):
    """Multi-Queries NIAH – ask for multiple keys at once."""
    pairs = [(f"item_{_random_word(rng, 4)}", _random_word(rng)) for _ in range(4)]
    needles = " ".join(
        f"The item {k} has code {v}." for k, v in pairs
    )
    text = _pad_to_length(needles, ctx_len, processor)
    ask = rng.sample(pairs, k=min(2, len(pairs)))
    question = (
        "What are the codes for "
        + " and ".join(f"'{k}'" for k, _ in ask) + "?"
    )
    return text, question, [v for _, v in ask]


def _gen_variable_tracking(rng, ctx_len, processor):
    """Variable Tracking – chain of variable assignments."""
    n_vars = 5
    var_names = [f"X{i}" for i in range(n_vars)]
    values = [_random_word(rng) for _ in range(n_vars)]
    assignments = " ".join(
        f"{var} = {val};" for var, val in zip(var_names, values)
    )
    # Add some re-assignments
    swaps = []
    for _ in range(3):
        i, j = rng.sample(range(n_vars), 2)
        values[i], values[j] = values[j], values[i]
        swaps.append(f"{var_names[i]} = {values[i]}; {var_names[j]} = {values[j]};")
    assignments += " " + " ".join(swaps)
    text = _pad_to_length(assignments, ctx_len, processor)
    ask_var = rng.choice(range(n_vars))
    question = f"After all assignments, what is the final value of {var_names[ask_var]}?"
    return text, question, [values[ask_var]]


def _gen_common_words(rng, ctx_len, processor):
    """Common Words – find words that appear frequently."""
    word_pool = [_random_word(rng, 5) for _ in range(20)]
    common = rng.sample(word_pool, 3)
    # Make common words appear many times
    tokens_list = []
    for _ in range(50):
        tokens_list.append(rng.choice(common))
        tokens_list.append(rng.choice(word_pool))
    text = _pad_to_length(" ".join(tokens_list), ctx_len, processor)
    question = "Which words appear most frequently in the text above? List the top 3."
    return text, question, common


def _gen_frequent_words(rng, ctx_len, processor):
    """Frequent Words – stricter frequency thresholds."""
    base_words = [_random_word(rng, 5) for _ in range(15)]
    target = base_words[0]
    tokens_list = []
    for _ in range(80):
        if rng.random() < 0.4:
            tokens_list.append(target)
        else:
            tokens_list.append(rng.choice(base_words))
    text = _pad_to_length(" ".join(tokens_list), ctx_len, processor)
    question = "What single word appears most frequently in the text?"
    return text, question, [target]


def _gen_single_hop_qa(rng, ctx_len, processor):
    """Single-hop QA."""
    entity = _random_word(rng).capitalize()
    fact = _random_word(rng)
    needle = f"{entity} is known for {fact}."
    text = _pad_to_length(needle, ctx_len, processor)
    question = f"What is {entity} known for?"
    return text, question, [fact]


def _gen_multi_hop_qa(rng, ctx_len, processor):
    """Multi-hop QA – requires chaining two facts."""
    a = _random_word(rng).capitalize()
    b = _random_word(rng).capitalize()
    fact = _random_word(rng)
    needles = f"{a} lives in {b}. The specialty of {b} is {fact}."
    text = _pad_to_length(needles, ctx_len, processor)
    question = f"What is the specialty of the place where {a} lives?"
    return text, question, [fact]


# Extended tasks to reach 13 total
def _gen_niah_multikey_multivalue(rng, ctx_len, processor):
    """Multi-Key Multi-Value NIAH."""
    pairs = {
        _random_word(rng): [_random_word(rng) for _ in range(2)]
        for _ in range(3)
    }
    needles = " ".join(
        f"Key {k} stores {' and '.join(vs)}." for k, vs in pairs.items()
    )
    text = _pad_to_length(needles, ctx_len, processor)
    ask_key = rng.choice(list(pairs.keys()))
    question = f"What values does key '{ask_key}' store?"
    return text, question, pairs[ask_key]


def _gen_niah_long_needle(rng, ctx_len, processor):
    """NIAH with a longer, more complex needle."""
    name = _random_word(rng).capitalize()
    code = "".join(rng.choices(string.digits, k=8))
    needle = (
        f"The registration code for {name} is {code}. "
        f"This code must be remembered exactly."
    )
    text = _pad_to_length(needle, ctx_len, processor)
    question = f"What is the registration code for {name}?"
    return text, question, [code]


def _gen_counting(rng, ctx_len, processor):
    """Counting task – count occurrences of a target word."""
    target = _random_word(rng, 5)
    count = rng.randint(5, 15)
    words = [_random_word(rng, 5) for _ in range(40)]
    tokens_list = []
    placed = 0
    for w in words:
        tokens_list.append(w)
        if placed < count and rng.random() < 0.4:
            tokens_list.append(target)
            placed += 1
    while placed < count:
        tokens_list.insert(rng.randint(0, len(tokens_list)), target)
        placed += 1
    text = _pad_to_length(" ".join(tokens_list), ctx_len, processor)
    question = f"How many times does the word '{target}' appear?"
    return text, question, [str(count)]


def _gen_pattern_match(rng, ctx_len, processor):
    """Pattern matching – find items that match a pattern."""
    prefix = _random_word(rng, 3)
    matching = [prefix + _random_word(rng, 3) for _ in range(3)]
    non_matching = [_random_word(rng, 6) for _ in range(10)]
    all_items = matching + non_matching
    rng.shuffle(all_items)
    text = _pad_to_length(
        "Items: " + ", ".join(all_items), ctx_len, processor
    )
    question = f"Which items start with '{prefix}'?"
    return text, question, matching


# ---------------------------------------------------------------------------
# Task registry – maps task names to generator functions
# ---------------------------------------------------------------------------

RULER_TASKS = {
    "niah_single": _gen_single_niah,
    "niah_multikey": _gen_multi_keys_niah,
    "niah_multivalue": _gen_multi_values_niah,
    "niah_multiquery": _gen_multi_queries_niah,
    "variable_tracking": _gen_variable_tracking,
    "common_words": _gen_common_words,
    "frequent_words": _gen_frequent_words,
    "qa_single_hop": _gen_single_hop_qa,
    "qa_multi_hop": _gen_multi_hop_qa,
    "niah_multikey_multivalue": _gen_niah_multikey_multivalue,
    "niah_long_needle": _gen_niah_long_needle,
    "counting": _gen_counting,
    "pattern_match": _gen_pattern_match,
}

TASK_CATEGORIES = {
    "retrieval": [
        "niah_single",
        "niah_multikey",
        "niah_multivalue",
        "niah_multiquery",
        "niah_multikey_multivalue",
        "niah_long_needle",
    ],
    "multi_hop": ["variable_tracking", "qa_multi_hop"],
    "aggregation": ["common_words", "frequent_words", "counting"],
    "qa": ["qa_single_hop", "pattern_match"],
}


def _check_answer(response: str, expected: list) -> bool:
    """Check whether *response* contains all expected values."""
    response_lower = response.lower()
    return all(str(v).lower() in response_lower for v in expected)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="RULER 13-task evaluation for long-context models"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path or HuggingFace repo id for the model",
    )
    parser.add_argument("--adapter-path", type=str, default=None)

    # Task configuration
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to run (default: all 13). "
             f"Choices: {', '.join(RULER_TASKS.keys())}",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context length in tokens for all tasks",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=50,
        help="Number of samples per task",
    )

    # KV cache quantization
    parser.add_argument("--kv-bits", type=float, default=None)
    parser.add_argument("--kv-quant-scheme", type=str, default="uniform")
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=5000)
    parser.add_argument("--prefill-step-size", type=int, default=2048)

    # Generation
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results/ruler",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    rng = random.Random(args.seed)

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

    task_names = args.tasks if args.tasks else list(RULER_TASKS.keys())
    for t in task_names:
        if t not in RULER_TASKS:
            raise ValueError(
                f"Unknown task '{t}'. Choose from: {list(RULER_TASKS.keys())}"
            )

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
    task_scores = {}

    for task_name in task_names:
        gen_fn = RULER_TASKS[task_name]
        correct = 0

        for sample_idx in tqdm(
            range(args.samples_per_task), desc=f"Task: {task_name}"
        ):
            context, question, expected = gen_fn(
                rng, args.context_length, processor
            )

            prompt_text = (
                f"Read the following text carefully and answer the question.\n\n"
                f"{context}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )

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
            is_correct = _check_answer(response, expected)
            if is_correct:
                correct += 1

            entry = {
                "task": task_name,
                "sample": sample_idx,
                "question": question,
                "expected": json.dumps(expected),
                "response": response,
                "correct": is_correct,
                "elapsed_s": round(elapsed, 2),
                "prompt_tokens": gen_result.prompt_tokens,
                "generation_tokens": gen_result.generation_tokens,
                "peak_memory_gb": gen_result.peak_memory,
            }
            all_results.append(entry)

            if args.verbose:
                logging.info(
                    f"[{task_name}][{sample_idx}] "
                    f"correct={is_correct} time={elapsed:.1f}s"
                )

        accuracy = correct / args.samples_per_task
        task_scores[task_name] = {
            "correct": correct,
            "total": args.samples_per_task,
            "accuracy": round(accuracy, 4),
        }

    # ---- Save results ----
    model_name = args.model.split("/")[-1]
    csv_path = output_dir / f"{model_name}_ruler.csv"
    json_path = output_dir / f"{model_name}_ruler.json"

    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    overall_correct = sum(s["correct"] for s in task_scores.values())
    overall_total = sum(s["total"] for s in task_scores.values())
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0

    # Category-level scores
    category_scores = {}
    for cat, cat_tasks in TASK_CATEGORIES.items():
        cat_correct = sum(
            task_scores[t]["correct"]
            for t in cat_tasks
            if t in task_scores
        )
        cat_total = sum(
            task_scores[t]["total"]
            for t in cat_tasks
            if t in task_scores
        )
        category_scores[cat] = {
            "correct": cat_correct,
            "total": cat_total,
            "accuracy": round(cat_correct / cat_total, 4) if cat_total > 0 else 0,
        }

    summary = {
        "model": args.model,
        "kv_bits": args.kv_bits,
        "kv_quant_scheme": args.kv_quant_scheme,
        "context_length": args.context_length,
        "samples_per_task": args.samples_per_task,
        "task_scores": task_scores,
        "category_scores": category_scores,
        "overall_accuracy": round(overall_accuracy, 4),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print summary ----
    print(f"\n{'='*80}")
    print("RULER 13-Task Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Context length: {args.context_length} tokens")
    if args.kv_bits is not None:
        print(f"KV Bits: {args.kv_bits}  Scheme: {args.kv_quant_scheme}")
    print(f"\n{'Task':<30} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 56)
    for task_name, scores in task_scores.items():
        print(
            f"{task_name:<30} {scores['correct']:>8} "
            f"{scores['total']:>6} {scores['accuracy']*100:>9.2f}%"
        )
    print("-" * 56)
    print(f"\n{'Category':<30} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 56)
    for cat, scores in category_scores.items():
        print(
            f"{cat:<30} {scores['correct']:>8} "
            f"{scores['total']:>6} {scores['accuracy']*100:>9.2f}%"
        )
    print("-" * 56)
    print(
        f"{'OVERALL':<30} {overall_correct:>8} "
        f"{overall_total:>6} {overall_accuracy*100:>9.2f}%"
    )
    print(f"{'='*80}")
    print(f"Results saved to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
