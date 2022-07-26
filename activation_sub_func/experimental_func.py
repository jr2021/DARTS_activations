import torch
import torch.nn as nn
from torch import Tensor
from activation_sub_func.unary_func import Power, Log, Sinc, Exp2, Asinh, Beta_add, Maximum0
from activation_sub_func.binary_func import Mul, BetaMix, Stack, ExpBetaSubAbs


class DartsFunc_simple(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.unary_1 = Log()
        self.unary_2 = Log()
        self.binary_1 = BetaMix(channels)

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        x_2 = self.unary_2(input)
        x_prim = torch.stack([x_1, x_2], dim=0)
        x_prim.clamp(-10, 10)
        return self.binary_1(x_prim)


class DartsFunc_complex(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.unary_1 = Power(3)
        self.unary_2 = Power(3)
        self.unary_3 = Power(3)
        self.unary_4 = Power(3)
        self.binary_1 = Mul()
        self.binary_2 = BetaMix(channels)

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        x_2 = self.unary_2(input)
        x_prim = torch.stack([x_1, x_2], dim=0)
        x_prim.clamp(-10, 10)
        x_3 = self.binary_3(x_prim)
        res = torch.stack([x_prim, x_3], dim=0)
        res.clamp(-10, 10)
        return self.binary_2(res)


class GDAS_simple(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.unary_1 = Sinc()
        self.unary_2 = Exp2()
        self.binary_1 = ExpBetaSubAbs(channels)

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        x_2 = self.unary_2(input)
        x_prim = torch.stack([x_1, x_2], dim=0)
        x_prim.clamp(-10, 10)
        return self.binary_1(x_prim)


class GDAS_complex(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.unary_1 = Asinh()
        self.unary_2 = Beta_add(channels)
        self.unary_3 = Maximum0()
        self.unary_4 = Power(2)
        self.binary_1 = Mul()
        self.binary_2 = ExpBetaSubAbs(channels)

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        x_2 = self.unary_2(input)
        x_prim = torch.stack([x_1, x_2], dim=0)
        x_prim.clamp(-10, 10)
        x_3 = self.binary_3(x_prim)
        res = torch.stack([x_prim, x_3], dim=0)
        res.clamp(-10, 10)
        return self.binary_2(res)