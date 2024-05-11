from torch import nn
from yolov7.layers.common import RepVGGBlock, RepBlock, SPPF, ConvBNSiLU, SimSPPF


class EfficientRep(nn.Module):
    """
    EfficientRep backbone网络 是YOLOv6的基本骨架网络架构
    """
    def __init__(self, in_channels=3, channels_list=None, num_repeats=None,
                 block=RepVGGBlock, fuse_p2=False, cspsppf=False):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_p2 = fuse_p2

        # ---------------------------
        # 第一个输入处理模块：stem
        # ---------------------------
        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        # ---------------------------
        # 第二个输入处理模块：ERBlock-2
        # ---------------------------
        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block
            )
        )

        # ----------------------------
        # 第三个输入处理模块：ERBlock-3,
        # 从这个模块开始的输出进入Neck部分
        # ----------------------------
        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block
            )
        )

        # ----------------------------
        # 第四个输入处理模块：ERBlock-4,
        # 该模块的输出也要进入Neck部分
        # ----------------------------
        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_p2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)
    