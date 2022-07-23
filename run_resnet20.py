import codecs
import json
import os
from datetime import time

import numpy as np
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from ResNet20 import ResNet20
from activation_sub_func.experimental_func import DartsFunc_1
from naslib.utils import utils

train_size = 0.25
batch_size = 4
seed = 49
epochs = 100
save_path = ""

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
               'train_time': []}

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

# net = ResNet20()
net = ResNet20(ac_func=DartsFunc_1, requires_channels=True).to("cuda:0")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
        train_loss.update(float(loss.detach.cpu()))
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    net.eval()
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
        val_loss.update(float(loss.detach.cpu()))

    end_time = time.time()
    errors_dict["train_acc_1"].append(train_top5.avg)
    errors_dict["train_acc_5"].append(train_top1.avg)
    errors_dict["train_loss"].append(train_loss.avg)
    errors_dict["valid_acc_1"].append(val_top1.avg)
    errors_dict["valid_acc_5"].append(val_top5.avg)
    errors_dict["valid_loss"].append(val_loss.avg)
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
        json.dump(errors_dict, file, separators=(',', ':'))

    torch.save(net.state_dict(), os.join(save_path, "model.pth"))


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
    test_loss.update(float(loss.detach.cpu()))

errors_dict["test_acc_1"].append(test_top5.avg)
errors_dict["test_acc_5"].append(test_top1.avg)
errors_dict["test_loss"].append(test_loss.avg)

print("Test loss:{:.5f}, Test Accuracy (top1, top5): {:.5f}, {:.5f} ".format(test_loss.avg, test_top1.avg, test_top5))

with codecs.open(os.path.join(save_path, 'errors.json'), 'w', encoding='utf-8') as file:
    json.dump(errors_dict, file, separators=(',', ':'))

print("Finished")
