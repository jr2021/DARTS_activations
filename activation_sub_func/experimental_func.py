import torch
import torch.nn as nn
from torch import Tensor
from activation_sub_func.unary_func import Power, Log, Sinc, Exp2, Asinh, Beta_add, Maximum0, Erf, Sigmoid, Tanh
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
        x_prim = torch.stack([x_1, x_2])
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
        x_3 = self.unary_3(input)

        x_prime = torch.stack([x_1, x_2])
        x_prime.clamp(-10, 10)
        x_prime = self.binary_1(x_prime)

        res = torch.stack([x_prime, x_3])
        res.clamp(-10, 10)
        return self.unary_4(self.binary_2(res))


class GDAS_simple(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.unary_1 = Sinc()
        self.unary_2 = Exp2()
        self.binary_1 = ExpBetaSubAbs(channels)

    def forward(self, input: Tensor) -> Tensor:
        x_1 = self.unary_1(input)
        x_2 = self.unary_2(input)
        x_prim = torch.stack([x_1, x_2])
        x_prim.clamp(-10, 10)
        return self.binary_1(x_prim)


class GDAS_complex(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.unary_1 = Erf()  # Erf error Tanh almost equal
        self.unary_2 = Beta_add(channels)
        self.unary_3 = Maximum0()
        self.unary_4 = Power(2)
        self.binary_1 = Mul()
        self.binary_2 = ExpBetaSubAbs(channels)

    def forward(self, input: Tensor) -> Tensor:
        assert torch.sum(torch.isinf(input)) == 0
        assert torch.sum(torch.isinf(input)) == 0
        x_1 = self.unary_1(input)
        x_2 = self.unary_2(input)
        x_3 = self.unary_3(input)

        x_prime = torch.stack([x_1, x_2])
        x_prime.clamp(-10, 10)
        x_prime = self.binary_1(x_prime)

        res = torch.stack([x_prime, x_3])
        res.clamp(-10, 10)
        return self.unary_4(self.binary_2(res))
