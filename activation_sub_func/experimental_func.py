import torch
import torch.nn as nn
from torch import Tensor
from activation_sub_func.unary_func import Power, Log
from activation_sub_func.binary_func import Mul, BetaMix, Stack


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

class DartsFunc_simple_r(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.beta = torch.nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.eps = 1e-10

    def forward(self, input: Tensor) ->Tensor:
        #input = input.clamp(-10, 10)
        input = torch.log(torch.maximum(input, torch.tensor(self.eps).repeat(input.shape).cuda()))
        input = (-self.beta*input) + ((1-self.beta)*input)
        return input

class DartsFunc_complex_r(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.beta = torch.nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, input: Tensor) -> Tensor:
        input = input.clamp(-10, 10)
        input = torch.pow(input,3)
        input = (-self.beta*input*input) + ((1-self.beta)*input)
        return input


