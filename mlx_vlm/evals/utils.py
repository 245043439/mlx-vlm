from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template


def inference(
    model,
    processor,
    question,
    image,
    max_tokens=3000,
    temperature=0.0,
    resize_shape=None,
    verbose=False,
):
    """Run inference on a single question."""
    if image is None:
        num_images = 0
    elif isinstance(image, list):
        num_images = len(image)
    else:
        num_images = 1

    prompt = apply_chat_template(
        processor, model.config, question, num_images=num_images
    )

    response = generate(
        model,
        processor,
        prompt,
        image=image,
        max_tokens=max_tokens,
        temperature=temperature,
        resize_shape=resize_shape,
        verbose=verbose,
    )
    return response.text


def text_inference(
    model,
    processor,
    prompt,
    max_tokens=3000,
    temperature=0.0,
    verbose=False,
    **kwargs,
):
    """Run text-only inference with optional KV quantization parameters.

    This is designed for long-context benchmarks that use text-only models
    with TurboQuant KV cache quantization.

    Args:
        model: The loaded language model.
        processor: The tokenizer/processor.
        prompt: The input prompt text (already formatted).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        verbose: Whether to print detailed output.
        **kwargs: Additional arguments passed to ``generate()``, including
            ``kv_bits``, ``kv_quant_scheme``, ``kv_group_size``,
            ``quantized_kv_start``, and ``prefill_step_size``.

    Returns:
        The generated text string.
    """
    response = generate(
        model,
        processor,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
        **kwargs,
    )
    return response
