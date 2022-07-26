import os
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def replace_id(x):
    if x == "eval_ResNet8_2":
        return "ReLU"
    elif x == "eval_ResNet8_3":
        return "Swish"
    elif x == "eval_ResNet8_0":
        return "DartsSimple"
    elif x == "eval_ResNet8_1":
        return "DartsComplex"
    elif x == "eval_ResNet20_2":
        return "ReLU"
    elif x == "eval_ResNet20_3":
        return "Swish"
    elif x == "eval_ResNet20_0":
        return "DartsSimple"
    elif x == "eval_ResNet20_1":
        return "DartsComplex"


labels = ["DartsSimple", "DartsComplex", "ReLU", "Swish"]

valid_acc_1 = []
seed = []
ex = []
epochs = []
for eval_i in os.listdir("eval"):
    if "ResNet20" in eval_i:
        continue
    with open(f"eval/{eval_i}/errors.json") as f:
        errors = json.load(f)
    valid_acc_1 += errors['valid_acc_1']
    ex += [replace_id(eval_i[:-2]) for _ in errors['valid_acc_1']]
    seed += [errors["seed"][0] for _ in errors['valid_acc_1']]
    epochs += [i for i in range(len(errors['valid_acc_1']))]

df = pd.DataFrame({"valid_acc_1": valid_acc_1, "experiment": ex, "seed": seed, "epochs": epochs})
sns.lineplot(data=df, x="epochs", y="valid_acc_1", hue="experiment", hue_order=labels)
plt.title("ResNet8 Evaluation on 5 seeds")
plt.show()

valid_acc_1 = []
seed = []
ex = []
epochs = []
for eval_i in os.listdir("eval"):
    if "ResNet8" in eval_i:
        continue
    with open(f"eval/{eval_i}/errors.json") as f:
        errors = json.load(f)
    valid_acc_1 += errors['valid_acc_1']
    ex += [replace_id(eval_i[:-2]) for _ in errors['valid_acc_1']]
    seed += [errors["seed"][0] for _ in errors['valid_acc_1']]
    epochs += [i for i in range(len(errors['valid_acc_1']))]

df = pd.DataFrame({"valid_acc_1": valid_acc_1, "experiment": ex, "seed": seed, "epochs": epochs})
plt.title("ResNet20 Evaluation on 5 seeds")
sns.lineplot(data=df, x="epochs", y="valid_acc_1", hue="experiment", hue_order=labels)
plt.show()
