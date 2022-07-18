import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

from activation_sub_func.binary_func import SigMul, BetaMix, Mul, Stack, Add
from activation_sub_func.unary_func import Cosh, Exp, Abs_op

lin = np.linspace(-10, 10, num=1000)
# plt.plot(lin, BetaMix(1)([Mul()([Cosh()(torch.tensor(lin)), Exp()(torch.tensor(lin))]), Abs_op()(torch.tensor(lin))]).detach().numpy().flatten())
plt.plot(lin, SigMul()([Exp()(torch.tensor(lin)), Exp()(torch.tensor(lin))]).detach().numpy().flatten())
plt.show()
