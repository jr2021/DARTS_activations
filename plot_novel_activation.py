import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from activation_sub_func.experimental_func import DartsFunc_complex, DartsFunc_simple
import torch
import torch.nn as nn

from activation_sub_func.binary_func import SigMul, BetaMix, Mul, Stack, Add
from activation_sub_func.unary_func import Cosh, Exp, Abs_op

lin = torch.linspace(-15, 15, 1000).reshape((1, 1, 1, 1000))
activation_func = DartsFunc_complex(1)
activation_func.binary_2.beta = nn.Parameter(activation_func.binary_2.beta - 0.7)

res = activation_func(lin).detach().numpy().flatten()
lin = lin.detach().numpy().flatten()

print(lin)
print(res)

plt.plot(lin, res)
plt.show()


lin = torch.linspace(-5, 5, 1000).reshape((1, 1, 1, 1000)).cuda()
activation_func = DartsFunc_simple(1).cuda()
activation_func.binary_1.beta = nn.Parameter(activation_func.binary_1.beta - 0.7)

res = activation_func(lin).cpu().detach().numpy().flatten()
lin = lin.cpu().detach().numpy().flatten()

print(lin)
print(res)

plt.plot(lin, res)
plt.show()
