import torch
from naslib.search_spaces.core.primitives import AbstractPrimitive


class Add(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.add(x[0], x[1])

    def get_embedded_ops(self):
        return None


class Sub(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.sub(x[0], x[1])

    def get_embedded_ops(self):
        return None


class Mul(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.mul(x[0], x[1])

    def get_embedded_ops(self):
        return None


class Div(AbstractPrimitive):
    def __init__(self, eps=1e-10):
        super().__init__(locals())
        self.eps = eps

    def forward(self, x, edge_data=None):
        return torch.div(x[0], x[1] + self.eps)

    def get_embedded_ops(self):
        return None


class Maximum(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.maximum(x[0], x[1])

    def get_embedded_ops(self):
        return None


class Minimum(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.minimum(x[0], x[1])

    def get_embedded_ops(self):
        return None


class SigMul(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.mul(torch.sigmoid(x[0]), x[1])

    def get_embedded_ops(self):
        return None


class ExpBetaSub2(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return torch.exp(-self.beta * torch.pow(torch.sub(x[0], x[1]), 2))

    def get_embedded_ops(self):
        return None


class ExpBetaSubAbs(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return torch.exp(-self.beta * torch.abs(torch.sub(x[0], x[1])))

    def get_embedded_ops(self):
        return None


class BetaMix(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return torch.add(-self.beta * x[0], (1 - self.beta) * x[1])

    def get_embedded_ops(self):
        return None