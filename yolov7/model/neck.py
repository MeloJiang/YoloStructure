import torch
from torch import nn
from yolov7.layers.common import (RepVGGBlock, RepBlock, Transpose,
                                  SPPF, ConvBNSiLU, SimSPPF, ConvBNReLU)


class RepPANNeck(nn.Module):
    def __init__(self,
                 channels_list=None,
                 num_repeats=None,
                 block=RepVGGBlock):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[5],
            block=block
        )
        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[6],
            block=block
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=block
        )
        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=block
        )

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1
        )
        self.up_sampling0 = Transpose(
            in_channels=channels_list[5],
            out_channels=channels_list[5]
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        )
        self.up_sampling1 = Transpose(
            in_channels=channels_list[6],
            out_channels=channels_list[6]
        )
        self.down_sampling2 = ConvBNReLU(
            in_channels=channels_list[6],
            out_channels=channels_list[7],
            kernel_size=3,
            stride=2
        )
        self.down_sampling1 = ConvBNReLU(
            in_channels=channels_list[8],
            out_channels=channels_list[9],
            kernel_size=3,
            stride=2
        )

    def forward(self, inputs):
        (x2, x1, x0) = inputs

        # -------------------------------------------------------------
        # 接受的是ERBlock_5的输出，shape: [20, 20, 512] --> [20, 20, 128]
        # -------------------------------------------------------------
        fpn_out0 = self.reduce_layer0(x0)

        # -------------------------------------------------------------------
        # fpn_out0 再经过反卷积进行上采样，shape: [20, 20, 128] --> [40, 40, 128]
        # -------------------------------------------------------------------
        up_sample_feat0 = self.up_sampling0(fpn_out0)

        # -------------------------------------------------------------------
        # 经过上采样后的 up_sample_feat0 再与ERBlock_4的输出进行cat拼接
        #   shape: [40, 40, 128] + [40, 40, 256] --> [40, 40, 384]
        # 再经过一个 RepBlock 得到进一步的输出 f_out0
        # -------------------------------------------------------------------
        f_concat_layer0 = torch.cat([up_sample_feat0, x1], dim=1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        up_sample_feat1 = self.up_sampling1(fpn_out1)

        f_concat_layer1 = torch.cat([up_sample_feat1, x2], dim=1)
        neck_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.down_sampling2(neck_out2)
        n_concat_layer1 = torch.cat([fpn_out1, down_feat1], dim=1)

        neck_out1 = self.Rep_n3(n_concat_layer1)
        down_feat0 = self.down_sampling1(neck_out1)

        n_concat_layer2 = torch.cat([fpn_out0, down_feat0], dim=1)
        neck_out0 = self.Rep_n4(n_concat_layer2)

        outputs = [neck_out2, neck_out1, neck_out0]

        return outputs


