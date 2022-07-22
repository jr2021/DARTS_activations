import torch
import torch.nn as nn
from torch import Tensor
from activation_sub_func.unary_func import Power
from activation_sub_func.binary_func import Mul, BetaMix, Stack


class DartsFunc_1(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.unary = Power(3)
        self.binary_1 = Mul()
        self.binary_2 = BetaMix(channels)

    def forward(self, input: Tensor) -> Tensor:
        x = self.unary(input)
        x_prim = torch.stack([x, x])
        x_prim = self.binary_1(x_prim)
        x = torch.stack([x_prim, x])
        return self.binary_2(x)
