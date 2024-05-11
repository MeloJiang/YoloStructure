#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov7.assigners.anchor_generator import generate_anchors
from yolov7.utils.general import xywh2xyxy, dist2bbox, bbox2dist
from yolov7.utils.figure_iou import IOULoss
from yolov7.assigners.atss_assigners import ATSSAssigner


class ComputeLoss:
    """Loss computation func."""
    def __init__(self,
                 fpn_strides=(8, 16, 32),
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 original_img_size=640,
                 iou_type='giou'
                 ):
        loss_weight = {
            'class': 1.0, 'iou': 2.5, 'dfl': 0.5
        }
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.original_img_size = original_img_size

        self.loss_weight = loss_weight

        self.assigner = ATSSAssigner(top_k=9, num_classes=self.num_classes)
        self.iou_type = iou_type
        self.vari_focal_loss = VariFocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.iou_type).cuda()

    def __call__(self, outputs, targets):
        # ----------------------------------------------------------------------------------------
        # 输入参数的解析：
        #       feats (list): 是一个列表，列表的内容为三个特征层级的输入，为：[[16, 128, 80, 80],
        #                                                             [16, 256, 40, 40],
        #                                                             [16, 512, 20, 20]]
        #       pred_scores (Tensor): 是一个张量，形状为：[16, 8400, 80]，表示每个生成框预测的物体的类别
        #       pred_distri (Tensor): 是一个张量，形状为：[16, 8400, 4]，表示每个生成框预测的框住物体的框大小
        #
        #       targets (Tensor): 是一个张量，形状为：[23, 6]，表示这一批次的图片的真实框总数为23个，每个真实
        #                       框的内容是 1+1+4，第一个表示所属图片的索引，第二个表示框的物体类别，第三部分表
        #                       示该框的四角坐标信息
        # ----------------------------------------------------------------------------------------
        feats, pred_scores, pred_distri = outputs

        # ----------------------------------------------------------------------------------------
        # 函数输出的内容解析：
        # 1. anchors (Tensor): shape=[8400, 4], 表示每个生成框的大小，大小使用对角坐标来表示，即对角坐标的xy值
        # 2. anchor_points (Tensor): shape=[8400, 2], 表示每个生成框的中心点坐标，比如第一个特征层级的生成框，
        #       因为第一个层级总共 80 * 80 个特征点，那么相当于原图 640 * 640 的尺寸，每个特征点代表的像素大小为
        #       640 / 80 = 8, 那么对于第一个特征点的生成框中心点的像素的坐标为： [4, 4], 因为一个特征格大小为
        #       8 * 8，那么中心点坐标就是 [4, 4]
        # 3. n_anchors_list (list): [6400, 1600, 400], 表示一共三个层级，每个层级分别有6400，1600，400个生
        #       成框
        # 4. stride_tensor (Tensor): shape=[8400, 1], 表示共计8400个生成框，每个生成框的一个特征格点代表的
        #       原始图片的像素大小，比如第一个特征层级的特征点大小为 80 * 80，那么对于原始图片 640 * 640 的大小
        #       来说，每个特征点所蕴含的像素信息就是 640 / 80 = 8，即每个特征点表示 8 * 8 的像素信息。
        # ----------------------------------------------------------------------------------------
        anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(feats, self.fpn_strides,
                                                                                 self.grid_cell_size,
                                                                                 self.grid_cell_offset,
                                                                                 device=feats[0].device)
        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.tensor([640, 640, 640, 640]).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # targets 的获取
        # ----------------------------------------------------------------------------------------
        # 输出内容的解析：
        # 1. targets (Tensor): shape=[16, 20, 5], 表示16张图片中，每张图片最多20个gt框，每个gt框有5个信息，
        #       分别是gt框所框住的目标实体的类别，后4个数字表示gt框的四角坐标信息
        # 2. gt_labels (Tensor): shape=[16, 20, 1], 表示16张图片中，每张图片最多20个gt框，每个gt框都框住
        #       实体，最后一维度的内容表示的是所框住实体的类别，取值范围为 0 ~ 79
        # 3. gt_bboxes (Tensor): shape=[16, 20, 4], 表示16张图片中，每张图片最多20个gt框，这些框住实体的
        #       gt框的位置信息，也就是对角坐标信息。
        # 4. mask_gt (Tensor): shape=[16, 20, 1], 将对角坐标的值相加，如果加和的结果大于0，则说明存在gt框，
        #       因此标记值为 1，说明此处存在gt框；如果取值为 0 ，则说明此处不存在gt框
        # ----------------------------------------------------------------------------------------
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes 的获取
        # -----------------------------------------------------------------------------------
        # 1. anchor_points_s (Tensor): shape=[8400, 2], 原本的anchor_points表示的是8400个生成框的
        #       特征点的中心坐标点，比如，第一个层级的特征点共有 80 * 80 = 6400 个，那么对于第一个特征点的
        #       中心坐标就是 [4, 4]，因为第一层级的特征点一个点就表示 8 * 8 个像素信息，那么中心点坐标相对
        #       于原图就是 [4, 4]，那么每个中心点坐标信息除以特征点层级相对应的像素信息，就可以得到他们在特征
        #       图层级上的相对坐标位置，比如第一层级的中心坐标点为 [4, 4],那么将其除以对应的像素 8 之后，得到
        #       的中心坐标点是 [4/8, 4/8] = [0.5, 0.5], 表示第0.5个特征格子处
        # 2. pred_bboxes (Tensor): shape=[16, 8400, 4],
        #       p
        # -----------------------------------------------------------------------------------
        anchor_points_s = anchor_points / stride_tensor
        pred_distri, pred_conf = pred_distri[:, :, 1:], pred_distri[:, :, 0]
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)

        try:
            target_labels, target_bboxes, target_scores, fg_mask, iou_mask = \
                self.assigner(
                    anchors,
                    n_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    mask_gt,
                    pred_bboxes.detach() * stride_tensor
                )
        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            _anchors = anchors.cpu().float()
            _n_anchors_list = n_anchors_list
            _gt_labels = gt_labels.cpu().float()
            _gt_bboxes = gt_bboxes.cpu().float()
            _mask_gt = mask_gt.cpu().float()
            _pred_bboxes = pred_bboxes.detach().cpu().float()
            _stride_tensor = stride_tensor.cpu().float()

            target_labels, target_bboxes, target_scores, fg_mask, iou_mask = \
                self.assigner(
                    _anchors,
                    _n_anchors_list,
                    _gt_labels,
                    _gt_bboxes,
                    _mask_gt,
                    _pred_bboxes * _stride_tensor
                )
        # 重新构建bbox的尺度
        target_bboxes /= stride_tensor

        # 分类损失计算--cls loss
        target_labels = torch.where(
            fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes)
        )
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        # print(torch.where(pred_scores <= 1.0)) # 找一找源头的图片输入有什么问题
        loss_cls = self.vari_focal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        if target_scores_sum > 1:
            loss_cls /= target_scores_sum

        # 锚框损失计算--bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                                            target_bboxes, target_scores, target_scores_sum,
                                            fg_mask)

        # 置信度损失计算--conf loss
        conf_loss = F.binary_cross_entropy_with_logits(pred_conf, iou_mask)

        loss = self.loss_weight['class'] * loss_cls + self.loss_weight['iou'] * loss_iou + conf_loss

        return loss, torch.cat([(self.loss_weight['iou']*loss_iou).unsqueeze(0),
                                (self.loss_weight['class']*loss_cls).unsqueeze(0)]).detach()

    @staticmethod
    def preprocess(targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])

        max_len = max(len(l) for l in targets_list)
        targets = torch.from_numpy(
            np.array(list(map(lambda l: l+[[-1, 0, 0, 0, 0]] * (max_len - len(l)),
                              targets_list)))[:, 1:, :]
        ).to(targets.device)

        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    @staticmethod
    def bbox_decode(anchor_points, pred_dist):
        return dist2bbox(pred_dist, anchor_points)


class VariFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(VariFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score, gt_score, label):
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - label) + gt_score * label

        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(),
                                           gt_score.float(), reduction='none') * weight).sum()
        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOULoss(box_format='xyxy', iou_type=iou_type, eps=1e-10)

        self.use_dfl = False

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        Args:
            pred_dist (Tensor): shape=[16, 8400, 4], 表示预测的8400个生成框，每个生成
                                框的框住目标实体的长度尺寸lt和rb表示
            pred_bboxes (Tensor): shape=[16, 8400, 4], 表示预测的8400个生成框，每个生
                                成框框住目标实体的锚框的对角坐标信息，格式为xyxy
            target_bboxes (Tensor): shape=[16, 8400, 4], 表示目标的真实gt框是否和8400
                                个生成框有关联，有关联的生成框的对角坐标信息是多少
            target_scores (Tensor): shape=[16, 8400, 80], 表示目标的真实gt框是否和8400
                                个生成框有关联，有关联的生成框所框住的目标实体类别是什么，用的
                                是one_hot向量表示，以此来表示分数scores
            fg_mask (Tensor): shape=[16, 8400], 表示总共8400个生成框中，哪些生成框和gt框、
                                有所关联，有关联的生成框的位置取值为1，否则为0
        """
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # -------------------------------------------------------------------------
            # 对于fg_mask的处理：
            #       假设fg_mask全部元素之和大于0，则说明存在gt框与生成框关联的情况，此时对最后一维度
            #       扩张并重复，制作锚框掩码 bbox_mask 用以表示哪些位置的锚框是存在的
            # -------------------------------------------------------------------------
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])

            # -------------------------------------------------------------------------
            # 根据 bbox_mask 选择出有效的预测的锚框信息，用以后续计算损失值
            # -------------------------------------------------------------------------
            pred_bboxes_pos = torch.masked_select(
                pred_bboxes, bbox_mask
            ).reshape([-1, 4])

            # -------------------------------------------------------------------------
            # 根据 bbox_mask 选择出有效的真实gt框的锚框信息，用于和预测框进行iou损失计算
            # -------------------------------------------------------------------------
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask
            ).reshape([-1, 4])

            # -------------------------------------------------------------------------
            # 将目标分数最后一位加和，得到的尺寸为 shape=[16, 8400], 以为表示8400个生成框中，存在
            # 目标实体的生成框所框住的目标实体的分数，然后根据 fg_mask 选择出有效的生成框位置，用以后
            # 续计算
            # -------------------------------------------------------------------------
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask
            ).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight

            if target_scores_sum > 1:
                loss_iou = loss_iou.sum() / target_scores_sum  # 相当于算个平均数
            else:
                loss_iou = loss_iou.sum()
            loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.long) - target
        weight_right = target - target_left

        loss_left = F.cross_entropy(
            pred_dist.view(-1, 1), target_left.view(-1), reduction='none'
        ).view(target_left.shape) * weight_left

        loss_right = F.cross_entropy(
            pred_dist.view(-1, 1), target_right.view(-1), reduction='none'
        ).view(target_left.shape) * weight_right

        return (loss_left + loss_right).mean(-1, keepdim=True)

