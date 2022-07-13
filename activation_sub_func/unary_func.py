import torch
from naslib.search_spaces.core.primitives import AbstractPrimitive


class Power(AbstractPrimitive):
    def __init__(self,power):
        super().__init__(locals())
        self.power=power
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.pow(x,self.power)
    def get_embedded_ops(self):
        return None

class Sqrt(AbstractPrimitive):
    def __init__(self,eps=1e-10):
        super().__init__(locals())
        self.eps = eps
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.pow(torch.maximum(x,torch.tensor(self.eps).repeat(x.shape).cuda()),.5)
    def get_embedded_ops(self):
        return None

class Sin(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.sin(x)
    def get_embedded_ops(self):
        return None
    
class Cos(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.cos(x)
    def get_embedded_ops(self):
        return None

class Abs_op(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.abs(x)
    def get_embedded_ops(self):
        return None

class Sign(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return x*-1
    def get_embedded_ops(self):
        return None

class Beta_mul(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return x * self.beta
    def get_embedded_ops(self):
        return None

class Beta_add(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return x + self.beta
    def get_embedded_ops(self):
        return None

class Log(AbstractPrimitive):
    def __init__(self,eps=1e-10):
        super().__init__(locals())
        self.eps = eps
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.log(torch.maximum(x,torch.tensor(self.eps).repeat(x.shape).cuda()))
    def get_embedded_ops(self):
        return None

class Exp(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        x=torch.exp(x)
        return x
    def get_embedded_ops(self):
        return None

class Sinh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.sinh(x)
    def get_embedded_ops(self):
        return None

class Cosh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.cosh(x)
    def get_embedded_ops(self):
        return None

class Tanh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.tanh(x)
    def get_embedded_ops(self):
        return None

class Asinh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.asinh(x)
    def get_embedded_ops(self):
        return None
    
class Atan(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.atan(x)
    def get_embedded_ops(self):
        return None

class Sinc(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.sinc(x)
    def get_embedded_ops(self):
        return None

class Maximum0(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.maximum(x,torch.zeros(x.shape).cuda())
    def get_embedded_ops(self):
        return None
    
class Minimum0(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.minimum(x,torch.zeros(x.shape).cuda())
    def get_embedded_ops(self):
        return None
    
class Sigmoid(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.sigmoid(x)
    def get_embedded_ops(self):
        return None

class LogExp(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        x=torch.log(1+torch.exp(x))
        return x
    def get_embedded_ops(self):
        return None

class Exp2(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.exp(-torch.pow(x,2))
    def get_embedded_ops(self):
        return None

class Erf(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return torch.erf(x)
    def get_embedded_ops(self):
        return None

class Beta(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))
    def forward(self,x, edge_data=None):
        x=x.clamp(-10,10)
        return self.beta
    def get_embedded_ops(self):
        return None

