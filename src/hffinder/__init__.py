from typing import NamedTuple
from transformers import PreTrainedTokenizer

import torch

__version__ = "0.1"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TransformerModel = NamedTuple(
    "TransformerModel", model=torch.nn.Module, tokenizer=PreTrainedTokenizer
)
