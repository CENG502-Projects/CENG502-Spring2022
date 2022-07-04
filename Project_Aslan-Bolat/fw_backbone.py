import torch
import torch.nn as nn
from torch import Tensor
from model import _resnet, BasicBlock

class ConvBlock(nn.Module):
    '''
    Conv Block Architecture used in 
    Matching Networks for One Shot Learning
    Prototypical Networks for Few-shot Learning
    '''
    def __init__(self,
                 inplanes: int,
                 planes: int = 64,
                 pooling: bool = True) -> None:
        super().__init__()
        ## in Matching Nets paper the authors says that
        ## when 4 of these blocks used with and 28x28 rgb image
        ## the output is 1x1 with 64 for channels.
        ## for the output size to match a padding of 1 is used
        ## which is not discussed in the paper.
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if pooling:
            self.maxpool = nn.MaxPool2d(2, 2)
        else:
            self.maxpool = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if not self.maxpool is None:
            x = self.maxpool(x)
        return x

class Conv4(nn.Module):
    def __init__(self, inplanes: int = 3) -> None:
        super().__init__()
        self.cb1 = ConvBlock(inplanes, 64)
        self.cb2 = ConvBlock(64, 64)
        self.cb3 = ConvBlock(64, 64)
        self.cb4 = ConvBlock(64, 64)
        self.channel = 64

    def forward(self, x: Tensor, a: Tensor = None) -> Tensor:
        x = self.cb1(x)
        x = self.cb2(x)
        self.size = x.size(2)
        if not a is None:
            x = a * x 
        x = self.cb3(x)
        x = self.cb4(x)
        return torch.flatten(x, 1)

class Conv6(nn.Module):
    '''
    Backbone network with 6 convolutional blocks from
    A CLOSER LOOK AT FEW-SHOT CLASSIFICATION
    '''
    def __init__(self, inplanes: int = 3) -> None:
        super().__init__()
        self.cb1 = ConvBlock(inplanes, 64)
        self.cb2 = ConvBlock(64, 64)
        self.cb3 = ConvBlock(64, 64)
        self.cb4 = ConvBlock(64, 64)
        self.cb5 = ConvBlock(64, 64, pooling=False)
        self.cb6 = ConvBlock(64, 64, pooling=False)
        self.channel = 64

    def forward(self, x: Tensor, a: Tensor = None) -> Tensor:
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        self.size = x.size(2)
        if not a is None:
            x = a * x 
        x = self.cb5(x)
        x = self.cb6(x)
        return torch.flatten(x, 1)

def resnet10(attention_layer: int = -1, few_shot=True, **kwargs):
    r"""ResNet-10 model from
    `"A CLOSER LOOK AT FEW-SHOT CLASSIFICATION" <https://arxiv.org/pdf/1904.04232.pdf>`_.
    """
    return _resnet("resnet10", BasicBlock, [1, 1, 1, 1], False, False, "", attention_layer, few_shot, **kwargs)

if __name__ == "__main__":
    conv4 = Conv4(3)
    image = torch.rand((1,3,84,84))
    feature = conv4(image)
    print(feature.size())
