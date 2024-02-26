from collections import namedtuple

import torch

__version__ = "0.1"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TransformerModel = namedtuple("TransformerModel", ['model', 'tokenizer'])
