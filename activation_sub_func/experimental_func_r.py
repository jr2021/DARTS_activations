import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Identity

from activation_sub_func.unary_func_nc import Power, Log, Beta_add, Abs_op, Beta_mul, Sqrt, Sign, Maximum0, Sigmoid
from activation_sub_func.binary_func_nc import Mul, BetaMix, Stack, Add, Minimum, SigMul


class DartsFunc_simple_r(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.unary_1 = Beta_add(channels)
        self.binary = Mul()

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        return self.binary(torch.stack([x_1, x_1]))


class DartsFunc_complex_r(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.unary_1 = Abs_op()
        self.unary_2 = Abs_op()
        self.unary_3 = Beta_mul(channels)
        self.unary_4 = Beta_add(channels)
        self.binary_1 = Mul()
        self.binary_2 = Add()

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        x_2 = self.unary_2(input)
        x_3 = self.unary_3(input)
        b_1 = self.binary_1(torch.stack([x_1, x_2]))
        x_4 = self.unary_4(b_1)
        return self.binary_2(torch.stack([x_3, x_4]))


class DrNasFunc_simple_r(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.unary_1 = Maximum0()
        self.binary = Add()

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        return self.binary(torch.stack([x_1, x_1]))


class DrNasFunc_complex_r(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.unary_1 = Sign()
        self.unary_2 = Sqrt()
        self.unary_3 = Identity()
        self.unary_4 = Beta_add(channels)
        self.binary_1 = Minimum()
        self.binary_2 = Add()

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        x_2 = self.unary_2(input)
        x_3 = self.unary_3(input)
        b_1 = self.binary_1(torch.stack([x_1, x_2]))
        x_4 = self.unary_4(b_1)
        return self.binary_2(torch.stack([x_3, x_4]))


class GdasFunc_simple_r(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, channels: int, inplace: bool = False) -> None:
        super().__init__()
        self.unary_1 = Sigmoid()
        self.binary = SigMul()

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        return self.binary(torch.stack([x_1, x_1]))
