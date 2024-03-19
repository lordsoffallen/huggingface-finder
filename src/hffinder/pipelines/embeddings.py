from datasets import Dataset
from .. import TransformerModel
from .tools import forward

import torch.nn.functional as f
import torch


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    """ Mean Pooling - Take attention mask into account for correct averaging """
    token_embeddings, attention_mask = model_output.to('cpu'), attention_mask.to('cpu')
    last_hidden = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(
    model_and_tokenizer: TransformerModel,
    text_list: list[str],
    batch_size: int = 400,
    normalize: bool = True,
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

    pooled_embeddings = mean_pooling(output, encoded_input['attention_mask'])

    if pooled_embeddings.dim() == 2 and pooled_embeddings.shape[0] > 1:
        # Big chunk of text was processed in batches, so we average them again.
        pooled_embeddings = torch.mean(pooled_embeddings, dim=0, keepdim=True)

    if normalize:
        pooled_embeddings = f.normalize(pooled_embeddings, p=2, dim=1)

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
