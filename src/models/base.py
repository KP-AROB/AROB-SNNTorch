import torch
from abc import ABC
from src.utils.parameters import load_fnc


class BaseFSNN(ABC, torch.nn.Module):
    def __init__(self,
                 input_shape: tuple = (1, 28, 28),
                 n_output: int = 10,
                 n_steps: int = 50,
                 beta: float = 0.8,
                 encoding_type: str = None):
        super().__init__()
        self.input_shape = input_shape
        self.n_output = n_output
        self.n_steps = n_steps
        self.beta = beta
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.encoding = load_fnc(
            'snntorch.spikegen', encoding_type) if encoding_type else None
