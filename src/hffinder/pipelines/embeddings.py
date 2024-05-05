from datasets import Dataset, concatenate_datasets
from .. import TransformerModel
from .tools import forward

import logging
import torch
import itertools


logger = logging.getLogger(__file__)


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    """ Mean Pooling - Take attention mask into account for correct averaging """
    token_embeddings, attention_mask = model_output.to('cpu'), attention_mask.to('cpu')
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(model_output: torch.Tensor) -> torch.Tensor:
    return model_output[:, 0]


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def flatten_list(text_list: list):
    return list(itertools.chain.from_iterable(text_list))


def get_embeddings(
    model_and_tokenizer: TransformerModel,
    text_list: list[str] | list,
    batch_size: int = 400,
    normalize: bool = False,
    reduce: bool = False,
) -> torch.Tensor:
    model = model_and_tokenizer.model
    tokenizer = model_and_tokenizer.tokenizer

    if isinstance(text_list, list):
        if isinstance(text_list[0], list):
            text_list = flatten_list(text_list)

    encoded_input = tokenizer(
        text_list,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    if len(encoded_input['input_ids']) > tokenizer.model_max_length:
        raise ValueError(f"Tokens has a size of={len(encoded_input['input_ids'])}"
                         f"which is greater than={tokenizer.model_max_length}")

    output = forward(
        model=model,
        encoded_input=encoded_input,
        input_filter_keys=None,
        batch_size=batch_size,
    )

    # Check if the output is batch of output or not
    if isinstance(output, list):
        output = torch.concat([v.last_hidden_state for v in output])
    else:
        output = output.last_hidden_state

    pooled_embeddings = cls_pooling(output)
    # pooled_embeddings = mean_pooling(output, encoded_input['attention_mask'])

    if reduce:
        pooled_embeddings = _batch_reduce(pooled_embeddings)

    if normalize:
        pooled_embeddings = normalize_embeddings(pooled_embeddings)

    return pooled_embeddings


def _batch_reduce(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.dim() == 2 and embeddings.shape[0] > 1:
        # Big chunk of text was processed in batches, so we average them again.
        return torch.mean(embeddings, dim=0, keepdim=True)


def compute_embeddings(
    ds: Dataset, model_and_tokenizer: TransformerModel, batch_size: int = 400
) -> Dataset:

    batched_ds = ds.filter(
        lambda x: True if len(x) > 1 else False, input_columns='input_texts'
    )
    single_ds = ds.filter(
        lambda x: True if len(x) == 1 else False, input_columns='input_texts'
    )

    if len(batched_ds) > 0:
        logger.info(f"Running batched input data with len={len(batched_ds)}")
        batched_ds = batched_ds.map(
            lambda texts: {
                "embeddings": get_embeddings(
                    model_and_tokenizer=model_and_tokenizer,
                    text_list=texts,
                    batch_size=batch_size,
                    reduce=True,
                ).detach().cpu().numpy()[0]
            },
            input_columns='input_texts',
            batched=False,
        )

    logger.info("Running single input data with batches")
    single_ds = single_ds.map(
        lambda texts: {
            "embeddings": get_embeddings(
                model_and_tokenizer=model_and_tokenizer,
                text_list=texts,
                batch_size=batch_size,
                reduce=False,
            ).detach().cpu().numpy()[0]
        },
        input_columns='input_texts',
        batched=True,
        batch_size=batch_size,
    )

    ds = concatenate_datasets([batched_ds, single_ds])

    return ds
