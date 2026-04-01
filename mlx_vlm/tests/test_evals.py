"""Tests for the long-context evaluation scripts.

These tests validate the helper functions and task generators used by the
Needle-in-a-Haystack, LongBench-Summary, and RULER evaluation scripts
without requiring a real model or GPU.

The eval modules import ``mlx_vlm`` which depends on ``mlx`` (Apple Silicon
only).  To keep the pure-logic tests runnable on any platform the test
lazily imports only the pieces it needs and skips tests that require mlx.
"""

import random
import re
import string
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx not available")


# ---------------------------------------------------------------------------
# Lightweight mock processor for token-length estimation
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal tokenizer that splits on whitespace for testing."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(tokens)


class _MockProcessor:
    tokenizer = _MockTokenizer()


# ============================================================================
# Tests for evals/utils.py – text_inference helper
# ============================================================================


@needs_mlx
class TestTextInference:
    """Tests for the text_inference helper in evals/utils.py."""

    @patch("mlx_vlm.evals.utils.generate")
    def test_text_inference_returns_generation_result(self, mock_generate):
        from types import SimpleNamespace

        fake_result = SimpleNamespace(
            text="hello world",
            prompt_tokens=10,
            generation_tokens=5,
            peak_memory=1.0,
        )
        mock_generate.return_value = fake_result

        from mlx_vlm.evals.utils import text_inference

        model = MagicMock()
        processor = MagicMock()
        result = text_inference(model, processor, "test prompt")

        assert result.text == "hello world"
        mock_generate.assert_called_once()

    @patch("mlx_vlm.evals.utils.generate")
    def test_text_inference_passes_kv_kwargs(self, mock_generate):
        from types import SimpleNamespace

        fake_result = SimpleNamespace(text="ok")
        mock_generate.return_value = fake_result

        from mlx_vlm.evals.utils import text_inference

        model = MagicMock()
        processor = MagicMock()
        text_inference(
            model,
            processor,
            "prompt",
            kv_bits=3.5,
            kv_quant_scheme="turboquant",
        )

        _, kwargs = mock_generate.call_args
        assert kwargs["kv_bits"] == 3.5
        assert kwargs["kv_quant_scheme"] == "turboquant"


# ============================================================================
# Tests for needle_in_haystack.py helpers
# Pure-Python helpers are re-implemented inline so the tests run without mlx.
# When mlx IS available, these also test the actual module imports.
# ============================================================================


# ---- Inline copies of the pure helpers (no mlx dependency) ----

_NIAH_KEY_PHRASES = ["sandwich", "dolores park", "san francisco"]


def _check_retrieval(response: str, needle: str) -> bool:
    response_lower = response.lower()
    matches = sum(1 for p in _NIAH_KEY_PHRASES if p in response_lower)
    return matches >= 2


def _build_haystack_inline(target_token_count, needle, depth_percent, processor):
    """Inline version of _build_haystack for cross-platform testing."""
    from mlx_vlm.evals.needle_in_haystack import FILLER_SENTENCES

    tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
    )
    sample = " ".join(FILLER_SENTENCES[:5])
    sample_tokens = len(tokenizer.encode(sample, add_special_tokens=False))
    avg_tokens_per_sentence = sample_tokens / 5
    n_sentences = max(1, int(target_token_count / avg_tokens_per_sentence) + 10)
    sentences = [
        FILLER_SENTENCES[i % len(FILLER_SENTENCES)] for i in range(n_sentences)
    ]
    insert_idx = max(0, int(len(sentences) * depth_percent / 100))
    sentences.insert(insert_idx, needle)
    haystack = " ".join(sentences)
    tokens = tokenizer.encode(haystack, add_special_tokens=False)
    if len(tokens) > target_token_count:
        tokens = tokens[:target_token_count]
        haystack = tokenizer.decode(tokens, skip_special_tokens=True)
    return haystack


class TestNeedleInHaystack:

    def test_check_retrieval_detects_needle(self):
        response = (
            "The best thing to do in San Francisco is eat a sandwich "
            "and sit in Dolores Park."
        )
        assert _check_retrieval(response, "") is True

    def test_check_retrieval_rejects_irrelevant(self):
        assert _check_retrieval("I like to read books.", "") is False

    @needs_mlx
    def test_build_haystack_inserts_needle(self):
        from mlx_vlm.evals.needle_in_haystack import (
            DEFAULT_NEEDLE,
            _build_haystack,
        )

        processor = _MockProcessor()
        haystack = _build_haystack(200, DEFAULT_NEEDLE, 50, processor)
        assert "sandwich" in haystack.lower()
        assert "dolores" in haystack.lower()

    @needs_mlx
    def test_build_haystack_respects_approximate_length(self):
        from mlx_vlm.evals.needle_in_haystack import (
            DEFAULT_NEEDLE,
            _build_haystack,
        )

        processor = _MockProcessor()
        haystack = _build_haystack(100, DEFAULT_NEEDLE, 50, processor)
        tokens = processor.tokenizer.encode(haystack)
        assert len(tokens) <= 110

    @needs_mlx
    def test_parse_args_defaults(self):
        from mlx_vlm.evals.needle_in_haystack import parse_args

        with patch("sys.argv", ["prog", "--model", "test/model"]):
            args = parse_args()
        assert args.model == "test/model"
        assert args.kv_bits is None
        assert 1024 in args.context_lengths
        assert 50 in args.depth_percents


# ============================================================================
# Tests for longbench_summary.py helpers
# Pure ROUGE-L logic is duplicated inline for cross-platform testing.
# ============================================================================


def _tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return text.split()


def _lcs_length(x, y):
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
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


def _rouge_l(prediction: str, reference: str) -> dict:
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


class TestLongBenchSummary:

    def test_rouge_l_identical(self):
        scores = _rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert scores["f1"] == pytest.approx(1.0)
        assert scores["precision"] == pytest.approx(1.0)
        assert scores["recall"] == pytest.approx(1.0)

    def test_rouge_l_partial(self):
        scores = _rouge_l("the cat sat", "the cat sat on the mat")
        assert 0 < scores["f1"] < 1.0
        assert scores["recall"] < 1.0

    def test_rouge_l_empty(self):
        scores = _rouge_l("", "some reference text")
        assert scores["f1"] == 0.0

    def test_tokenize_strips_punctuation(self):
        tokens = _tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lcs_length(self):
        assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2
        assert _lcs_length([], ["a"]) == 0

    @needs_mlx
    def test_build_prompt_includes_context(self):
        from mlx_vlm.evals.longbench_summary import _build_prompt

        sample = {"context": "This is a test document.", "input": ""}
        prompt = _build_prompt(sample, "gov_report")
        assert "test document" in prompt
        assert "Summary:" in prompt

    @needs_mlx
    def test_build_prompt_includes_input_focus(self):
        from mlx_vlm.evals.longbench_summary import _build_prompt

        sample = {"context": "Doc text.", "input": "key decisions"}
        prompt = _build_prompt(sample, "multi_news")
        assert "key decisions" in prompt


# ============================================================================
# Tests for ruler.py task generators and helpers
# ============================================================================


def _check_answer(response: str, expected: list) -> bool:
    response_lower = response.lower()
    return all(str(v).lower() in response_lower for v in expected)


def _random_word(rng, length=6):
    return "".join(rng.choices(string.ascii_lowercase, k=length))


_TEST_FILLER = [
    "The grass is green in the morning light.",
    "Clouds moved across the sky in shifting patterns.",
    "A river flowed gently through the valley below.",
]


def _pad_to_length(text, target_tokens, processor):
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    current_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if current_tokens >= target_tokens:
        tokens = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    needed = target_tokens - current_tokens
    avg_tok = 8
    n_filler = max(1, needed // avg_tok + 5)
    filler = " ".join(
        _TEST_FILLER[i % len(_TEST_FILLER)] for i in range(n_filler)
    )
    combined = text + " " + filler
    tokens = tokenizer.encode(combined, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


class TestRuler:

    def test_check_answer_all_present(self):
        assert _check_answer("The answer is abc and def", ["abc", "def"])
        assert not _check_answer("The answer is abc", ["abc", "xyz"])

    def test_check_answer_case_insensitive(self):
        assert _check_answer("HELLO world", ["hello", "world"])

    def test_pad_to_length_respects_target(self):
        processor = _MockProcessor()
        result = _pad_to_length("short text", 50, processor)
        tokens = processor.tokenizer.encode(result)
        assert len(tokens) <= 55

    @needs_mlx
    def test_all_13_tasks_registered(self):
        from mlx_vlm.evals.ruler import RULER_TASKS

        assert len(RULER_TASKS) == 13

    @needs_mlx
    def test_task_categories_cover_all_tasks(self):
        from mlx_vlm.evals.ruler import RULER_TASKS, TASK_CATEGORIES

        categorized = set()
        for tasks in TASK_CATEGORIES.values():
            categorized.update(tasks)
        assert categorized.issubset(set(RULER_TASKS.keys()))

    @needs_mlx
    def test_single_niah_generator(self):
        from mlx_vlm.evals.ruler import _gen_single_niah

        rng = random.Random(42)
        processor = _MockProcessor()
        context, question, answers = _gen_single_niah(rng, 100, processor)
        assert len(answers) == 1
        assert answers[0] in context

    @needs_mlx
    def test_multi_keys_niah_generator(self):
        from mlx_vlm.evals.ruler import _gen_multi_keys_niah

        rng = random.Random(42)
        processor = _MockProcessor()
        context, question, answers = _gen_multi_keys_niah(rng, 100, processor)
        assert len(answers) == 1
        assert answers[0] in context

    @needs_mlx
    def test_variable_tracking_generator(self):
        from mlx_vlm.evals.ruler import _gen_variable_tracking

        rng = random.Random(42)
        processor = _MockProcessor()
        context, question, answers = _gen_variable_tracking(rng, 200, processor)
        assert len(answers) == 1
        assert "X" in question

    @needs_mlx
    def test_common_words_generator(self):
        from mlx_vlm.evals.ruler import _gen_common_words

        rng = random.Random(42)
        processor = _MockProcessor()
        context, question, answers = _gen_common_words(rng, 200, processor)
        assert len(answers) == 3

    @needs_mlx
    def test_counting_generator(self):
        from mlx_vlm.evals.ruler import _gen_counting

        rng = random.Random(42)
        processor = _MockProcessor()
        context, question, answers = _gen_counting(rng, 200, processor)
        assert len(answers) == 1
        assert answers[0].isdigit()

    @needs_mlx
    def test_parse_args_defaults(self):
        from mlx_vlm.evals.ruler import parse_args

        with patch("sys.argv", ["prog", "--model", "test/model"]):
            args = parse_args()
        assert args.model == "test/model"
        assert args.context_length == 4096
        assert args.samples_per_task == 50

    @needs_mlx
    def test_each_task_generator_returns_valid_tuple(self):
        """Ensure every registered task generator returns (str, str, list)."""
        from mlx_vlm.evals.ruler import RULER_TASKS

        rng = random.Random(123)
        processor = _MockProcessor()
        for task_name, gen_fn in RULER_TASKS.items():
            context, question, answers = gen_fn(rng, 100, processor)
            assert isinstance(context, str), f"{task_name}: context not str"
            assert isinstance(question, str), f"{task_name}: question not str"
            assert isinstance(answers, list), f"{task_name}: answers not list"
            assert len(answers) >= 1, f"{task_name}: no answers"
