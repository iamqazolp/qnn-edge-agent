import math
from typing import Any

import torch
from datasets import load_dataset


def evaluate_wikitext2_perplexity(
    model: Any,
    tokenizer: Any,
    device: str | torch.device | None = None,
    batch_size: int = 4,
    window_size: int = 1024,
    max_chunks: int | None = None,
) -> float:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    raw_texts = [text for text in dataset["text"] if text.strip()]
    full_text = "\n\n".join(raw_texts)

    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    total_tokens = encodings.size(0)
    n_chunks = total_tokens // window_size
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)

    if n_chunks == 0:
        return float("nan")

    total_nll = 0.0
    total_prediction_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, n_chunks, batch_size):
            chunk_batch_count = min(batch_size, n_chunks - i)
            batch_tensors = []
            for b in range(chunk_batch_count):
                idx = i + b
                start = idx * window_size
                end = start + window_size
                batch_tensors.append(encodings[start:end])

            input_batch = torch.stack(batch_tensors, dim=0).to(device)
            labels = input_batch.clone()

            outputs = model(input_batch, labels=labels)
            loss = outputs.loss
            tokens_in_batch = chunk_batch_count * (window_size - 1)

            total_nll += loss.item() * tokens_in_batch
            total_prediction_tokens += tokens_in_batch

    mean_nll = total_nll / total_prediction_tokens
    return float(math.exp(mean_nll))
