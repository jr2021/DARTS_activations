import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from activation_sub_func.experimental_func_r import DrNasFunc_complex_r, DrNasFunc_simple_r
import torch
import torch.nn as nn

from activation_sub_func.binary_func import SigMul, BetaMix, Mul, Stack, Add
from activation_sub_func.unary_func import Cosh, Exp, Abs_op

lin = torch.linspace(-5, 5, 1000).reshape((1, 1, 1, 1000)).cuda()
activation_func = DrNasFunc_complex_r(1).cuda()

res = activation_func(lin).cpu().detach().numpy().flatten()
lin = lin.cpu().detach().numpy().flatten()

print(lin)
print(res)

plt.plot(lin, res)
plt.title("DrNas Complex\n"
          "min(-x, sqrt(x)) + x + beta")
plt.savefig(f"figures/big.png", dpi=300)
plt.show()


lin = torch.linspace(-5, 5, 1000).reshape((1, 1, 1, 1000)).cuda()
activation_func = DrNasFunc_simple_r(1).cuda()

res = activation_func(lin).cpu().detach().numpy().flatten()
lin = lin.cpu().detach().numpy().flatten()
plt.title("DrNas Simple\n"
          "max(0, x) + max(0, x)")

print(lin)
print(res)

plt.plot(lin, res)
plt.savefig("figures/samll.png", dpi=300)
plt.show()
