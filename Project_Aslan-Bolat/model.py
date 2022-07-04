""" This code is adapted from pytorch. 
This is the simplified and adapted version of https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
"""

from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url

__all__ = [
    "resnet18"
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth"
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention_layer: int = -1,
        few_shot: bool = False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.attention_layer = attention_layer
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # Changed for CIFAR10
        # kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        # Taken from https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if not few_shot:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, attention_map: Tensor=None, out_layer: int=-1) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        for layer_ind in range(len(self.layers)):
            x = self.layers[layer_ind](x)
            if (not attention_map is None) and self.attention_layer >= 0 and self.attention_layer-1 == layer_ind:
                self.size = x.size(2)
                self.channel = x.size(1)
                x = attention_map * x
            if out_layer != -1 and out_layer-1 == layer_ind:  # out_layer-1 as indexing starts from 0, unlike layers: layer1 ...
                return x # Used at initialization to get attention map size

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if not self.fc is None:
            self.embedded_feature = x
            x = self.fc(x)

        return x

    def forward(self, x: Tensor, attention_map: Tensor=None, out_layer: int=-1) -> Tensor:
        return self._forward_impl(x, attention_map, out_layer)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    load_path: str,
    attention_layer: int,
    few_shot: bool = False,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, attention_layer=attention_layer, few_shot=few_shot, **kwargs)
    if pretrained and not load_path:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    elif pretrained and load_path:
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, load_path: str = "", attention_layer: int = -1, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, load_path, attention_layer, **kwargs)

class RLAgent(nn.Module):
    """This class implements RL related functions.
    """

    def __init__(self, 
                image_res: int = 256,
                channels: List[int] = [64, 64, 64],
                maxpool_size = 2,
                embed_feature_res: int = 512,
                attention_res: int = 512,
                attention_channels: int = 4,
                u_size: int = 512,  # Not used for current implementation. Used for softmax policy
                lin_block_depth: int = 1):
        super().__init__()

        layers = []
        in_channels = 3
        self.conv_out_size = image_res
        for i in range(len(channels)):
            # layer definition is taken from section 4.2
            layer = nn.Sequential(nn.Conv2d(in_channels, channels[i], kernel_size=3),
                                nn.BatchNorm2d(channels[i]),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size = maxpool_size))
            layers.append(layer)
            self.conv_out_size = int((self.conv_out_size - 2))  # pad = 0, kernel = 3, stride = 1, dilation = 1
            self.conv_out_size = int((self.conv_out_size - maxpool_size)/maxpool_size + 1)  # Max pooling output
            in_channels = channels[i]

        self.conv_block = nn.Sequential(*layers)

        # Create Linear Block (see Fig. 3)
        # Sigmoid is accounted at forward pass
        lin_input = channels[-1] + embed_feature_res
        layers = []
        for i in range(lin_block_depth):
            if i == lin_block_depth-1: 
                layers.append(nn.Sequential(nn.Linear(lin_input, attention_res*attention_res),
                                        nn.ReLU()))
            else:
                layers.append(nn.Sequential(nn.Linear(lin_input, lin_input),
                                        nn.ReLU()))
        self.lin_block = nn.Sequential(*layers)

        # Create policy to draw attention weights
        self.attn_res = attention_res
        self.attn_chn = attention_channels
        # self.policy_softmax = nn.Sequential(nn.Linear(u_size, attention_res*attention_res),
        #                             nn.Softmax(dim=-1))

    def forward(self, 
                image: Tensor,
                feature_vec: Tensor = None,
                test: bool = False):
        """ One step of RL agent.
        Args:
            image: To extract low level informations
            feature_vec (batch, backbone.fc.in_features): High level informations from pretrained backbone
        Returns:
            action: Attention maps
        """
        i_image = self.conv_block(image)  # Compute image feature vector
        i_image = torch.mean(i_image, dim=(2,3))
        pre_linblock = torch.cat([i_image, feature_vec], dim=1)
        g = torch.sigmoid(self.lin_block(pre_linblock))  # Compute policy function
        normal_cls = torch.distributions.normal.Normal(g, 1)
        if not test:
            attention = normal_cls.sample()
            log_prob = normal_cls.log_prob(attention)
        else:
            attention = g
            log_prob = normal_cls.log_prob(attention)
        # One way of using policy
        # attention = nn.functional.softmax(g, dim=1)  # Attention of shape B x hw
        attention = attention.view((attention.size(0), 1, self.attn_res, self.attn_res))  # B x h x w x 1
        attention = attention.repeat(1, self.attn_chn, 1, 1)  # Repeat attention for all channels
        return attention, log_prob

if __name__=="__main__":
    agent = RLAgent(image_res=128, attention_res=16, attention_channels=5)
    image = torch.rand((2, 3, 128, 128))
    feature_prev = torch.rand((2, 512))
    agent(image, feature_prev)