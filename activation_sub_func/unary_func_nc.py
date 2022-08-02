import torch
from naslib.search_spaces.core.primitives import AbstractPrimitive

"""Unary operations without clamp. 
used in final evaluation"""


class Power(AbstractPrimitive):
    def __init__(self, power):
        super().__init__(locals())
        self.power = power

    def forward(self, x, edge_data=None):
        result = torch.pow(x, self.power)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Sqrt(AbstractPrimitive):
    def __init__(self, eps=1e-10):
        super().__init__(locals())
        self.eps = eps

    def forward(self, x, edge_data=None):
        result = torch.pow(torch.maximum(x, torch.tensor(self.eps).repeat(x.shape).cuda()), .5)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Sin(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.sin(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Cos(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.cos(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Abs_op(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.abs(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Sign(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = x * -1
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Beta_mul(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, x, edge_data=None):
        result = x * self.beta
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Beta_add(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, x, edge_data=None):
        result = x + self.beta
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Log(AbstractPrimitive):
    def __init__(self, eps=1e-10):
        super().__init__(locals())
        self.eps = eps

    def forward(self, x, edge_data=None):
        result = torch.log(torch.maximum(x, torch.tensor(self.eps).repeat(x.shape).cuda()))
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Exp(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        x = torch.exp(x)
        result = x
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Sinh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.sinh(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Cosh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.cosh(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Tanh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.tanh(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Asinh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.asinh(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Atan(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.atan(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Sinc(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.sinc(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Maximum0(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.maximum(x, torch.zeros(x.shape).cuda())
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Minimum0(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.minimum(x, torch.zeros(x.shape).cuda())
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Sigmoid(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.sigmoid(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class LogExp(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        x = torch.log(1 + torch.exp(x))
        result = x
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Exp2(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.exp(-torch.pow(x, 2))
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Erf(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        result = torch.erf(x)
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Beta(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, x, edge_data=None):
        result = self.beta
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None


class Beta_GDAS(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones((1, channels, 1, 1)))
        # self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        result = self.beta.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        # assert torch.sum(torch.isnan(result)) == 0
        return result

    def get_embedded_ops(self):
        return None
