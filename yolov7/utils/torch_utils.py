import time
from contextlib import contextmanager
from copy import deepcopy
import torch
import torch.distributed as dist
import torch.nn as nn


@contextmanager
def torch_distributed_zero_first(local_rank):
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    # yield语句用于将控制权移交给with语句块的内部代码
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def time_sync():
    '''Waits for all kernels in all streams on a CUDA device to complete if cuda is available.'''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def get_model_info(model, img_size=640):
    """获取模型的参数和计算速度"""
    from thop import profile
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)

    # --------------------------------------------------------------------
    # 直接使用profile测出模型的计算速度以及参数量：flops 和 params
    # --------------------------------------------------------------------
    flops, params = profile(deepcopy(model), inputs=(img, ), verbose=False)
    params /= 1e6  # 除以1000000，转换为单位M
    flops /= 1e9  # 除以100000000，转换为单位G
    img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
    flops *= img_size[0] * img_size[1] / stride / stride * 2
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv, bn):
    """Fuse convolution and batch_norm layers
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fused_conv = (
        nn.Conv2d(conv.in_channels,
                  conv.out_channels,
                  kernel_size=conv.kernel_size,
                  stride=conv.stride,
                  padding=conv.padding,
                  groups=conv.groups,
                  bias=True).requires_grad_(False).to(conv.weight.device)
    )

    # 准备过滤器
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.running_var + bn.eps)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))

    # 准备稀疏偏差
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fused_conv


def fuse_model(model):
    """Fuse convolution and batch normalization layers of the model"""
    from yolov7.layers.common import ConvModule

    for m in model.modules():
        if type(m) is ConvModule and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.forward_fuse
    return model
