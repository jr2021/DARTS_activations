import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Identity

from activation_sub_func.unary_func import Power, Log, Beta_add, Abs_op, Beta_mul, Sqrt, Sign, Maximum0, Sigmoid
from activation_sub_func.binary_func import Mul, BetaMix, Stack, Add, Minimum, SigMul


class DartsFunc_simple(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.unary = Log()
        self.binary_1 = BetaMix(channels)

    def forward(self, input: Tensor) -> Tensor:
        x = self.unary(input)
        x_prim = torch.stack([x, x])
        return self.binary_1(x_prim)


class DartsFunc_complex(nn.Module):
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




