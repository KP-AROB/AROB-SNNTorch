import torch
from abc import ABC
from src.utils.parameters import load_fnc


class BaseFSNN(ABC, torch.nn.Module):
    def __init__(self,
                 n_input: int = 28 * 28,
                 n_hidden: int = 16,
                 n_output: int = 10,
                 beta: float = 0.8,
                 timesteps: int = 50,
                 encoding_type: str = None):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.timesteps = timesteps
        self.beta = beta
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.encoding = load_fnc(
            'snntorch.spikegen', encoding_type) if encoding_type else None
