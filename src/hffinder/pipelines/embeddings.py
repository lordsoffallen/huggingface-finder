from datasets import Dataset
from .. import TransformerModel
from .tools import forward

import torch


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


def get_embeddings(
    model_and_tokenizer: TransformerModel,
    text_list: list[str],
    batch_size: int = 400,
    normalize: bool = False,
):
    model = model_and_tokenizer.model
    tokenizer = model_and_tokenizer.tokenizer

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

    if pooled_embeddings.dim() == 2 and pooled_embeddings.shape[0] > 1:
        # Big chunk of text was processed in batches, so we average them again.
        pooled_embeddings = torch.mean(pooled_embeddings, dim=0, keepdim=True)

    if normalize:
        pooled_embeddings = normalize_embeddings(pooled_embeddings)

    return pooled_embeddings


def compute_embeddings(
    ds: Dataset, model_and_tokenizer: TransformerModel, batch_size: int = 400
) -> Dataset:

    ds = ds.map(
        lambda texts: {
            "embeddings": get_embeddings(
                model_and_tokenizer=model_and_tokenizer,
                text_list=texts,
                batch_size=batch_size
            ).detach().cpu().numpy()[0]
        }, input_columns='input_texts'
    )

    return ds
