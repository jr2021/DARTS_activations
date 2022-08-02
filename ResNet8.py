import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# from activation_sub_func.experimental_func import DartsFunc_complex, DartsFunc_simple

"""
ResNet8 implementation in pytorch with our activation functions
"in_channels": 16,
"out_channels": 16,
"kernel_size": 3,
"padding": 1
"""


class BasicBlock(nn.Module):
    def __init__(self, channels, ac_func, requires_channels: bool = False):
        super().__init__()

        args = dict()
        if requires_channels:
            args["channels"] = channels

        self.model = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=channels, out_channels=channels, padding=1),
            nn.BatchNorm2d(channels),
            ac_func(**args),
            nn.Conv2d(kernel_size=3, in_channels=channels, out_channels=channels, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.final_ac = ac_func(**args)

    def forward(self, x):
        return self.final_ac(x + self.model(x))


class ReductionBasicBlock(nn.Module):
    def __init__(self, channels_in, channels_out, ac_func, requires_channels: bool = False):
        super().__init__()

        args = dict()
        if requires_channels:
            args["channels"] = channels_out

        self.model = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=channels_in, out_channels=channels_out, stride=2, padding=1),
            nn.BatchNorm2d(channels_out),
            ac_func(**args),
            nn.Conv2d(kernel_size=3, in_channels=channels_out, out_channels=channels_out, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
        )

        self.reduction_conv = nn.Conv2d(kernel_size=1, in_channels=channels_in, out_channels=channels_out, stride=2,
                                        padding=0)
        self.final_ac = ac_func(**args)

    def forward(self, x):
        res = self.final_ac(self.reduction_conv(x) + self.model(x))
        return res


class ResNet8(nn.Module):
    def __init__(self, ac_func=nn.ReLU, requires_channels: bool = False):
        super().__init__()

        args = dict()
        if requires_channels:
            args["channels"] = 16

        self.model = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16, padding=1),
            ac_func(**args),
            BasicBlock(channels=16, ac_func=ac_func, requires_channels=requires_channels),
            BasicBlock(channels=16, ac_func=ac_func, requires_channels=requires_channels),
            nn.Sequential(
                nn.AvgPool2d(8),
                nn.Flatten(),
                nn.Linear(256, 10),
                nn.Softmax()
            )
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # net = ResNet8()
    # net = ResNet8(ac_func=DartsFunc_complex, requires_channels=True).to("cuda:0")
    # net = ResNet8(ac_func=DartsFunc_simple, requires_channels=True).to("cuda:0")
    # net = ResNet8(ac_func=nn.ReLU, requires_channels=False).to("cuda:0")
    net = ResNet8(ac_func=nn.SiLU, requires_channels=False).to("cuda:0")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.025, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        print(f"epoch: {epoch + 1}")
        running_loss = 0.0
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

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')
