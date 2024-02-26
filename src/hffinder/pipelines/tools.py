from .. import device
from typing import Any

import torch


def _single_batch_forward(
    model: torch.nn.Module,
    encoded_input: dict | Any,
    input_filter_keys: list | str = None,
) -> torch.Tensor | Any:
    if isinstance(input_filter_keys, str):
        input_filter_keys = [input_filter_keys]

    encoded_input = {
        k: v.to(device) for k, v in encoded_input.items() if k not in input_filter_keys
    }

    with torch.no_grad():
        model_output = model(**encoded_input)

    return model_output


def forward(
    model: torch.nn.Module,
    encoded_input: dict | Any,
    input_filter_keys: list | str = None,
    batch_size: int = 400,
) -> Any:
    # Huge chunks of text gets split into higher dimensions and cause GPU OOM.
    # Use batch param to control the behaviour
    if encoded_input['input_ids'].shape[0] > batch_size:
        total_batch = encoded_input['input_ids'].shape[0]
        model_output = []

        for idx in range((total_batch + batch_size - 1) // batch_size):
            start, end = idx * batch_size, min((idx + 1) * batch_size, total_batch)
            batch = {
                k: v[start:end] if v.dim() == 1 else v[start:end, :]
                for k, v in encoded_input.items()
            }
            model_output.append(
                _single_batch_forward(model, batch, input_filter_keys=input_filter_keys)
            )
    else:
        model_output = _single_batch_forward(
            model, encoded_input, input_filter_keys=input_filter_keys
        )

    return model_output
