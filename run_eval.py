import argparse

import codecs
import json
import os
import time

from ResNet20 import ResNet20
from ResNet8 import ResNet8

import torchvision
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from naslib.utils import utils
from activation_sub_func.experimental_func import DartsFunc_complex, DartsFunc_simple, GDAS_simple, GDAS_complex
from pathlib import Path

"""Evaluation of activations functions found on larger choice of operations"""

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default="ResNet20")
parser.add_argument('--ac_func', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--train_size', type=float, default=0.8)
parser.add_argument('--save_path', type=str, default="eval")
# 0: Darts_simple
# 1: Darts_complex
# 2: ReLU
# 3: SiLU
# 4: GDAS_simple
# 5: GDAS_complex

args = parser.parse_args()

if __name__ == '__main__':
    train_size = args.train_size
    batch_size = args.batch_size
    seed = args.seed
    epochs = 100
    save_path = f"{args.save_path}_{args.network}_{args.ac_func}_{seed}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_top1 = utils.AverageMeter()
    train_top5 = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    val_top1 = utils.AverageMeter()
    val_top5 = utils.AverageMeter()
    val_loss = utils.AverageMeter()
    test_top1 = utils.AverageMeter()
    test_top5 = utils.AverageMeter()
    test_loss = utils.AverageMeter

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    errors_dict = {'train_acc_1': [],
                   'train_acc_5': [],
                   'train_loss': [],
                   'valid_acc_1': [],
                   'valid_acc_5': [],
                   'valid_loss': [],
                   'test_acc_1': [],
                   'test_acc_5': [],
                   'test_loss': [],
                   'runtime': [],
                   'train_time': [],
                   'seed': [seed]}

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(train_size * num_train))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(seed))

    validloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(seed))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.network == "ResNet20":
        if args.ac_func == 0:
            net = ResNet20(ac_func=DartsFunc_simple, requires_channels=True).to("cuda:0")
        elif args.ac_func == 1:
            net = ResNet20(ac_func=DartsFunc_complex, requires_channels=True).to("cuda:0")
        elif args.ac_func == 2:
            net = ResNet20(ac_func=nn.ReLU, requires_channels=False).to("cuda:0")
        elif args.ac_func == 3:
            net = ResNet20(ac_func=nn.SiLU, requires_channels=False).to("cuda:0")
        elif args.ac_func == 4:
            net = ResNet20(ac_func=GDAS_simple, requires_channels=True).to("cuda:0")
        elif args.ac_func == 5:
            net = ResNet20(ac_func=GDAS_complex, requires_channels=True).to("cuda:0")
        else:
            raise KeyError(f"{args.ac_func} is no valid value for --ac_func")
    elif args.network == "ResNet8":
        if args.ac_func == 0:
            net = ResNet8(ac_func=DartsFunc_simple, requires_channels=True).to("cuda:0")
        elif args.ac_func == 1:
            net = ResNet8(ac_func=DartsFunc_complex, requires_channels=True).to("cuda:0")
        elif args.ac_func == 2:
            net = ResNet8(ac_func=nn.ReLU, requires_channels=False).to("cuda:0")
        elif args.ac_func == 3:
            net = ResNet8(ac_func=nn.SiLU, requires_channels=False).to("cuda:0")
        elif args.ac_func == 4:
            net = ResNet8(ac_func=GDAS_simple, requires_channels=True).to("cuda:0")
        elif args.ac_func == 5:
            net = ResNet8(ac_func=GDAS_complex, requires_channels=True).to("cuda:0")
        else:
            raise KeyError(f"{args.ac_func} is no valid value for --ac_func")
    else:
        raise KeyError(f"{args.network} is no valid value for --network")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.025, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"epoch: {epoch + 1}")
        running_loss = 0.0
        start_time = time.time()
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            train_top1.update(prec1)
            train_top5.update(prec5)
            train_loss.update(float(loss.detach().cpu()))
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        net.eval()
        running_loss = 0.0
        for i, data in enumerate(validloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            val_top1.update(prec1)
            val_top5.update(prec5)
            val_loss.update(float(loss.detach().cpu()))

            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] val_loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        end_time = time.time()
        errors_dict["train_acc_1"].append(float(train_top5.avg))
        errors_dict["train_acc_5"].append(float(train_top1.avg))
        errors_dict["train_loss"].append(float(train_loss.avg))
        errors_dict["valid_acc_1"].append(float(val_top1.avg))
        errors_dict["valid_acc_5"].append(float(val_top5.avg))
        errors_dict["valid_loss"].append(float(val_loss.avg))
        errors_dict["runtime"].append(end_time - start_time)

        print("Epoch {} done. Train accuracy (top1, top5): {:.5f}, {:.5f}, Validation accuracy: {:.5f}, {:.5f}".format(
            epoch, train_top1.avg, train_top5.avg, val_top1.avg, val_top5.avg))
        print("Train loss:{:.5f}, Validation Loss:{:.5f}".format(train_loss.avg, val_loss.avg))
        train_top1.reset()
        train_top5.reset()
        train_loss.reset()
        val_top1.reset()
        val_top5.reset()
        val_loss.reset()

        with codecs.open(os.path.join(save_path, 'errors.json'), 'w', encoding='utf-8') as file:
            json.dump(errors_dict, file, separators=(',', ':'), indent=4)

        torch.save(net.state_dict(), f"{save_path}/model.pth")

    print('Finished Training')

    print("Testing")
    net.eval()
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
        test_top1.update(prec1)
        test_top5.update(prec5)
        test_loss.update(float(loss.detach().cpu()))

    errors_dict["test_acc_1"].append(float(test_top5.avg))
    errors_dict["test_acc_5"].append(float(test_top1.avg))
    errors_dict["test_loss"].append(float(test_loss.avg))

    print(
        "Test loss:{:.5f}, Test Accuracy (top1, top5): {:.5f}, {:.5f} ".format(test_loss.avg, test_top1.avg, test_top5))

    with codecs.open(os.path.join(save_path, 'errors.json'), 'w', encoding='utf-8') as file:
        json.dump(errors_dict, file, separators=(',', ':'), indent=4)

    print("Finished")