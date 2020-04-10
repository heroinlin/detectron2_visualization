# -*- coding: utf-8 -*-

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import CfgNode
from detectron2.modeling import META_ARCH_REGISTRY


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              1,
                              stride=stride,
                              padding=0,
                              bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              1,
                              stride=stride,
                              padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              3,
                              stride=stride,
                              padding=1,
                              bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """
    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               3,
                               stride=1,
                               padding=1,
                               bias=False,
                               groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""
    def __init__(self,
                 in_channels,
                 num_gates=None,
                 return_gates=False,
                 gate_activation='sigmoid',
                 reduction=16,
                 layer_norm=True):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Conv2d(in_channels,
        #                      in_channels // reduction,
        #                      kernel_size=1,
        #                      bias=True,
        #                      padding=0)
        self.fc1 = nn.Conv2d(in_channels,
                             in_channels // reduction,
                             kernel_size=1,
                             bias=False,
                             padding=0)
        self.norm1 = None
        if layer_norm:
            # self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
            self.norm1 = nn.BatchNorm2d((in_channels // reduction))
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Conv2d(in_channels // reduction,
        #                      num_gates,
        #                      kernel_size=1,
        #                      bias=True,
        #                      padding=0)
        self.fc2 = nn.Conv2d(in_channels // reduction,
                             num_gates,
                             kernel_size=1,
                             bias=False,
                             padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 IN=False,
                 bottleneck_reduction=4,
                 **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = ConvLayer(in_channels,
                              out_channels,
                              kernel_size=7,
                              stride=2,
                              padding=3)

    def forward(self, x):
        x = self.conv(x)
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv.conv.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


@META_ARCH_REGISTRY.register()
class OSNet(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(OSNet, self).__init__()

        layers = cfg.MODEL.OSNET.LAYERS
        channels = cfg.MODEL.OSNET.CHANNELS
        fc_layer_dim = cfg.MODEL.OSNET.FC_LAYER_DIM
        assert len(layers) + 1 == len(channels)
        class_num = cfg.MODEL.OSNET.CLASS_NUM
        self.device = cfg.MODEL.DEVICE

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages_and_names = []
        for i in range(len(layers)):
            if i == len(layers) - 1:
                reduce_spatial_size = False
            else:
                reduce_spatial_size = True
            stage = self._make_layer(layer=layers[i],
                                     in_channels=channels[i],
                                     out_channels=channels[i + 1],
                                     reduce_spatial_size=reduce_spatial_size)
            name = F"res{i+2}"
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_layer_dim,
                                           channels[-1],
                                           dropout_p=None)
        self.classifier = nn.Linear(fc_layer_dim, class_num)

        self._init_params()
        self.to(self.device)

    def _make_layer(self, layer: List[int], in_channels: int,
                    out_channels: int, reduce_spatial_size: bool):
        layers = []
        layers.append(OSBlock(in_channels, out_channels))
        for i in range(1, layer):
            layers.append(OSBlock(out_channels, out_channels))
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(Conv1x1(out_channels, out_channels),
                              nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(
                nn.Conv2d(input_dim, dim, kernel_size=1, stride=1, bias=False))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images):
        if isinstance(images, list):
            images = torch.stack(
                [x["image"].to(self.device) for x in images])
        x = self.conv1(images)
        x = self.max_pool(x)
        for stage, name in self.stages_and_names:
            x = stage(x)
        x = self.conv5(x)
        v = self.global_avgpool(x)
        if self.fc is not None:
            v = self.fc(v)
        if self.training:
            y = v.view(v.size(0), -1)
            y = self.classifier(y)
            return v, y
        else:
            v = v.view(v.size(0), -1)
            return v
