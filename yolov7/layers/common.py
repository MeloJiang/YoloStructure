#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import warnings
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from yolov7.utils.checkpoint import load_checkpoint


activation_dict = {'relu': nn.ReLU(),
                   'silu': nn.SiLU(),
                   'hardswish': nn.Hardswish()}


class SiLU(nn.Module):
    """SiLU激活函数"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    """卷积Conv + 批量归一化BN + 激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation_type, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_type is not None:
            self.act = activation_dict.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))


class ConvBNReLU(nn.Module):
    """卷积、批量归一化、ReLU激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                activation_type='relu',
                                padding=padding,
                                groups=groups,
                                bias=bias)

    def forward(self, x):
        return self.block(x)


class ConvBNSiLU(nn.Module):
    """卷积、批量归一化、SiLU激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                activation_type='silu',
                                padding=padding,
                                groups=groups,
                                bias=bias)

    def forward(self, x):
        return self.block(x)


class ConvBN(nn.Module):
    """卷积、批量归一化"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                activation_type=None,
                                padding=padding,
                                groups=groups,
                                bias=bias)

    def forward(self, x):
        return self.block(x)


class ConvBNHS(nn.Module):
    """卷积、批量归一化、Hardswish激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                activation_type='hardswish',
                                padding=padding,
                                groups=groups,
                                bias=bias)

    def forward(self, x):
        return self.block(x)


class SPPFModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        super().__init__()
        c_ = in_channels // 2  # 隐藏层通道数
        self.cv1 = block(in_channels=in_channels, out_channels=c_, kernel_size=1, stride=1)
        self.cv2 = block(in_channels=c_ * 4, out_channels=out_channels, kernel_size=1, stride=1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # -------------------------------------------------------------------
        # step1. 对于输入的x通过从cv1处理：
        #       shape: [batch, channels, h, w] --> [batch, channels//2, h, w]
        # -------------------------------------------------------------------
        x = self.cv1(x)

        # -------------------------------------------------------------------
        # step2. 对于输入的x通过m处理，最大池化操作：
        #       shape: [batch, channels//2, h, w] --> [batch, channels//2, h, w]
        # -------------------------------------------------------------------
        y1 = self.m(x)

        # -------------------------------------------------------------------
        # step3. 对于经过最大池化操作的输出y1，再进行一次最大池化操作：
        #       shape: [batch, channels//2, h, w] --> [batch, channels//2, h, w]
        # -------------------------------------------------------------------
        y2 = self.m(y1)

        # -------------------------------------------------------------------
        # step4. 对于经过最大池化操作的输出y2，再进行一次最大池化操作：
        #       shape: [batch, channels//2, h, w] --> [batch, channels//2, h, w]
        # -------------------------------------------------------------------
        y3 = self.m(y2)

        # -------------------------------------------------------------------
        # step5. 最后将经过处理的输出进行堆叠，在channels这一维度上堆叠：
        #       shape: [batch, channels//2, h, w] --> [batch, 4*channels//2, h, w]
        # step6. 最后将堆叠的结果再通过cv2卷积操作（不改变图像的宽和高）：
        #       shape: [batch, 4*channels//2, h, w] --> [batch, 4*channels//2, h, w]
        # -------------------------------------------------------------------
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class SimSPPF(nn.Module):
    """简化版的SPPF(激活函数为ReLU)"""
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        super().__init__()
        self.sppf = SPPFModule(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, block=block)

    def forward(self, x):
        return self.sppf(x)


class SPPF(nn.Module):
    """SPPF模块使用SiLU激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNSiLU):
        super().__init__()
        self.sppf = SPPFModule(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, block=block)

    def forward(self, x):
        return self.sppf(x)


class CSPSPPFModule(nn.Module):
    """CSP https://github.com/WongKinYiu/CrossStagePartialNetworks"""
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNReLU):
        super().__init__()
        c_ = int(out_channels * e)  # 隐藏层通道数
        self.cv1 = block(in_channels=in_channels, out_channels=c_,
                         kernel_size=1, stride=1)
        self.cv2 = block(in_channels=in_channels, out_channels=c_,
                         kernel_size=1, stride=1)
        self.cv3 = block(in_channels=c_, out_channels=c_,
                         kernel_size=3, stride=1)
        self.cv4 = block(in_channels=c_, out_channels=c_,
                         kernel_size=1, stride=1)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2)
        self.cv5 = block(in_channels=4 * c_, out_channels=c_,
                         kernel_size=1, stride=1)
        self.cv6 = block(in_channels=c_, out_channels=c_,
                         kernel_size=3, stride=1)
        self.cv7 = block(in_channels=2 * c_, out_channels=out_channels,
                         kernel_size=1, stride=1)

    def forward(self, x):
        # ------------------------------------------------------------
        # step1. 原始输入首先通过cv1、cv3、cv4的处理，得到x1：
        #       shapes: [batch, channels, h, w] --> [batch, c_, h, w]
        # ------------------------------------------------------------
        x1 = self.cv4(self.cv3(self.cv1(x)))

        # ------------------------------------------------------------
        # step2. 原始输入经过cv2的处理，得到y0：
        #       shapes: [batch, channels, h, w] --> [batch, c_, h, w]
        # ------------------------------------------------------------
        y0 = self.cv2(x)

        # ------------------------------------------------------------
        # step3. x1经过最大池化层的处理后，得到y1(最大池化层并未改变图片尺寸)：
        #       shapes: [batch, c_, h, w] --> [batch, c_, h, w]
        # ------------------------------------------------------------
        y1 = self.m(x1)

        # ------------------------------------------------------------
        # step4. step3中的y1再经过最大池化层处理，得到y2：
        #       shapes: [batch, c_, h, w] --> [batch, c_, h, w]
        # ------------------------------------------------------------
        y2 = self.m(y1)

        # ------------------------------------------------------------
        # step5. step4中的y2再经过最大池化层处理，得到y3：
        #       shapes: [batch, c_, h, w] --> [batch, c_, h, w]
        # ------------------------------------------------------------
        y3 = self.m(y2)

        # ------------------------------------------------------------
        # step6. 将step1、step3、step4、step5的结果按照channels维度拼接，再经过cv5、cv6处理：
        #       shapes: [batch, 4 * c_, h, w] --> [batch, c_, h, w]
        # ------------------------------------------------------------
        y4 = self.cv6(self.cv5(torch.cat([x1, y1, y2, y3], dim=1)))

        # ------------------------------------------------------------
        # step7. 将step2、step6的结果按照channels维度进行拼接，并通过cv7处理得到结果
        #       shapes: [batch, 2 * c_, h, w] --> [batch, out_channels, h, w]
        # ------------------------------------------------------------
        return self.cv7(torch.cat((y0, y4), dim=1))


class Transpose(nn.Module):
    """普通的转换类，默认为升采样"""
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.up_sample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.up_sample_transpose(x)


class RepVGGBlock(nn.Module):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False, use_se=False):
        super().__init__()
        """ 关于该类的参数的说明:
        Args:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数
            kernel_size (int or tuple): 卷积核的尺寸
            stride (int or tuple): 卷积核每次移动的距离
            padding (int or tuple): 0填充，输入的上下左右方向都会进行填充
            dilation (int or tuple): 卷积核元素之间的间隔，默认为1
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.non_linearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("SE block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_re_param = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvModule(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, activation_type=None,
                                        padding=padding, groups=groups)
            self.rbr_1x1 = ConvModule(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=1, stride=stride, activation_type=None,
                                      padding=padding_11, groups=groups)

    def forward(self, inputs):
        """Forward 过程"""
        if hasattr(self, 'rbr_re_param'):
            return self.non_linearity(self.se(self.rbr_re_param(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.non_linearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return (kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id,
                bias3x3 + bias1x1 + bias_id)

    def _avg_to_3x3_tensor(self, avg_p):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avg_p.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            return kernel, bias
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_val = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_val + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_re_param'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_re_param = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                      out_channels=self.rbr_dense.conv.out_channels,
                                      kernel_size=self.rbr_dense.conv.kernel_size[0],
                                      stride=self.rbr_dense.conv.stride[0],
                                      padding=self.rbr_dense.conv.padding,
                                      dilation=self.rbr_dense.conv.dilation[0],
                                      groups=self.rbr_dense.conv.groups,
                                      bias=True)
        self.rbr_re_param.weight.data = kernel
        self.rbr_re_param.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class BottleRep(nn.Module):
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = basic_block(in_channels=out_channels, out_channels=out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class RepBlock(nn.Module):
    """RepBlock是基本结构块"""
    def __init__(self, in_channels, out_channels, n=1,
                 block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()
        self.conv1 = block(in_channels=in_channels, out_channels=out_channels)
        self.block = nn.Sequential(*(block(in_channels=out_channels,
                                           out_channels=out_channels) for _ in range(n-1))) \
            if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels=in_channels, out_channels=out_channels,
                                   basic_block=basic_block, weight=True)
            n = n // 2

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class DetectBackend(nn.Module):
    def __init__(self, weights='yolov6s.pt', device=None, dnn=True):
        super().__init__()
        assert isinstance(weights, str) and Path(weights).suffix == '.pt',\
            f'{Path(weights).suffix} format is not supported.'
        model = load_checkpoint(weights, map_location=device)
        stride = int(model.stride.max())
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):
        y, _ = self.model(im)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return y


