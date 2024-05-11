import functorch.dim
from torch import nn
from yolov7.layers.common import ConvBNSiLU
import math
import torch
import torch.nn.functional as F
from yolov7.assigners.anchor_generator import generate_anchors
from yolov7.utils.general import dist2bbox


class Head(nn.Module):
    def __init__(self, num_classes=80, num_layers=3, inplace=True,
                 use_dfl=False, reg_max=0, channels_list=None, num_anchors=1):
        super().__init__()
        assert channels_list is not None

        self.nc = num_classes
        self.no = num_classes + 5
        self.nl = num_layers

        self.reg_max = reg_max
        self.use_dfl = use_dfl

        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]
        self.stride = torch.tensor(stride)

        self.prior_prob = 1e-2

        self.projection = None
        self.projection_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

        self.stem0 = ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        )
        self.cls_conv0 = ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        )
        self.reg_conv0 = ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        )
        self.cls_pred0 = nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        )
        self.reg_pred0 = nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=5 * (reg_max + num_anchors),
            kernel_size=1
        )

        self.stem1 = ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        )
        self.cls_conv1 = ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        )
        self.reg_conv1 = ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        )
        self.cls_pred1 = nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        )
        self.reg_pred1 = nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=5 * (reg_max + num_anchors),
            kernel_size=1
        )

        self.stem2 = ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        )
        self.cls_conv2 = ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        )
        self.reg_conv2 = ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        )
        self.cls_pred2 = nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        )
        self.reg_pred2 = nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=5 * (reg_max + num_anchors),
            kernel_size=1
        )

        self.stems = nn.ModuleList([self.stem0, self.stem1, self.stem2])
        self.cls_convs = nn.ModuleList([self.cls_conv0, self.cls_conv1, self.cls_conv2])
        self.reg_convs = nn.ModuleList([self.reg_conv0, self.reg_conv1, self.reg_conv2])
        self.cls_preds = nn.ModuleList([self.cls_pred0, self.cls_pred1, self.cls_pred2])
        self.reg_preds = nn.ModuleList([self.reg_pred0, self.reg_pred1, self.reg_pred2])

    def initialize_biases(self):

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.projection = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1),
                                       requires_grad=True)
        self.projection_conv.weight = nn.Parameter(self.projection.view(
            [1, self.reg_max + 1, 1, 1]
        ).clone().detach(), requires_grad=True)

    def forward(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)

                # [batch, channels, h, w] --> [batch, h*w, channels]
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, dim=1)
            reg_distri_list = torch.cat(reg_distri_list, dim=1)

            return x, cls_score_list, reg_distri_list

        else:
            cls_score_list = []
            reg_dist_list = []
            batch = 0
            for i in range(self.nl):
                batch, _, height, width = x[i].shape
                l = height * width
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)[:, 1:, :]

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.projection_conv(F.softmax(reg_output, dim=1))

                cls_output = torch.sigmoid(cls_output)

                cls_score_list.append(cls_output.reshape([batch, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([batch, 4, l]))

            cls_score_list = torch.cat(cls_score_list, dim=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, dim=-1).permute(0, 2, 1)

            anchor_points, stride_tensor = generate_anchors(
                feats=x, fpn_strides=self.stride, grid_cell_size=self.grid_cell_size, mode='af',
                grid_cell_offset=self.grid_cell_offset, device=x[0].device, is_eval=True,
            )

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor

            return torch.cat([
                pred_bboxes,
                torch.ones((batch, pred_bboxes.shape[1], 1),
                           device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                cls_score_list
            ], dim=-1)




