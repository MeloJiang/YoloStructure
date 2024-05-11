#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# The code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/general.py

import os
import time
import numpy as np
import cv2
import torch
import torchvision


def xywh2xyxy(x):
    """Convert boxes with shape [n, 4] from [x, y, w, h] to
     [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.5, classes=None,
                        agnostic_nms=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic_nms: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det: (int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """
    # -----------------------------------------------------------------------------------------------
    # NMS 的基本原理：
    #   先在图像中找到所有可能包含目标物体的矩形区域，并按照他们的置信度进行排列。然后从置信度最高的矩形开始，遍历所有的
    #   矩形，如果发现当前的矩形与前面任意一个矩形的重叠面积大于一个阈值，则将当前矩形舍去（因为重叠得太多了，所以视为对
    #   同一个目标实体的锚框，舍掉）。使得最终保留的预测框数量最少，但同时又能够保证检测的准确性和召回率。具体的实现方法
    #   包括以下几个步骤：
    #       1. 对于每个类别，按照预测框的置信度进行排序，将置信度最高的预测框作为基准。
    #
    #       2. 从剩余的预测框中选择一个与基准框的重叠面积最大的框，如果其重叠面积大于一定的阈值，则将其删除
    #
    #       3. 对于剩余的预测框，重复步骤2，直到所有的重叠面积都小于预测框，或者没有被删除的框剩余为止。
    #
    #   通过这样的方式，NMS可以过滤掉所有与基准框重叠面积大于阈值的冗余框，从而实现检测结果的优化。值得注意的是，NMS的阈
    #   值通常需要根据具体的数据集和应用场景进行调整，以兼顾准确性和召回率。
    # -----------------------------------------------------------------------------------------------
    num_classes = prediction.shape[2] - 5  # 类别的数量(80类)
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres,
                                        torch.max(prediction[..., 5:], dim=-1)[0] > conf_thres)

    # 检查参数：
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # 一些基本参数的设置，比如最大的NMS计算时间限制等
    max_wh = 4096
    max_nms = 30000
    time_limit = 10.0
    multi_label &= num_classes > 1  # 是否开启多目标NMS

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for img_idx, x in enumerate(prediction):
        # --------------------------------------------------------------------------------------------
        # 一开始的 x 代表的是一个批次中16张图片每一张图片的预测结果，大小为 [5733, 85] --- (本例中生成框数量为5733)
        # --------------------------------------------------------------------------------------------
        x = x[pred_candidates[img_idx]]

        if not x.shape[0]:
            continue

        # ----------------------------------------------------------------------------------------------
        # 置信度和是否包含物体的置信度相乘:
        #       置信度conf等于分类置信度和物体置信度的相乘
        #       x[:, 5:].shape = [5733, 80],   x[:, 4:5].shape = [5733, 1]
        #       x[:, 5:] * x[:, 4:5] = [5733, 80] * [5733, 1] = [5733, 80]
        #       也就是这一张图片的预测结果，5733个生成框中，每个生成框的类别预测置信度都乘上了一个是否包含物体的置信度
        # ----------------------------------------------------------------------------------------------
        x[:, 5:] *= x[:, 4:5]

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # 检测结果张量的形状是 (n, 6), 每一个行表示 (xyxy四角坐标， 置信度conf， 目标类别cls)
        if multi_label:
            # -----------------------------------------------------------------------------------------
            # 1. box 表示取出经过置信度筛选的 222 个预测结果的四角坐标信息，总共有 222 个box，box.shape=[222, 4]
            # 2. 对于经过置信度筛选后的 预测结果 x 进行类别预测的置信度位置转换，经过 x[:, 5:] > conf_thres 可以选出
            #       那些类别置信度大于阈值的位置，大于阈值的位置取 1，其他取0，因为一个锚框的预测结果可能有多于 1 个类别
            #       的置信度大于阈值，所以最后得到的结果长度可能大于 222 （本例为222个符合阈值要求的预测框），比如：
            #       第 110 号预测框的80个目标类别预测中，56类和70类的预测置信度都大于阈值，因此都要取上
            # -----------------------------------------------------------------------------------------
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T

            # -----------------------------------------------------------------------------------------
            # 经过最后选择，x的结果为满足置信度阈值的 358 个预测框 [358, 4 + 1 + 1]
            # 其中，因为同一个预测锚框所预测的类别可能有多于 1 个类别大于阈值，所以也要选上，这就导致了最后满足类别阈值的
            # 结果多于 222 个预测框，（即第 200 号预测框可能同时有第19类、第27类、第75类的预测分数大于阈值）
            # 最后得到的 x 的大小为：x.shape = [358, 4 + 1 + 1] 代表：4表示这个预测框的四角坐标位置；1 表示这个预测
            # 框的预测目标类别的分数；最后一个 1 表示这个类别的index值，即表示是什么类
            # -----------------------------------------------------------------------------------------
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)

        else:
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # 目标类别筛选器
        # -----------------------------------------------------------------------------
        # 如果用户提供了目标类别的数组 list, 那么NMS算法只会保留数组中存在的目标类别
        #       1. x.shape = [358, 6], 表示经过阈值conf筛选后的预测框，有358个
        #       2. x[:, 5:6].shape = [358, 1], 表示预测框第二维度最后一位的信息为目标框预测的类
        #           别的种类信息，
        #       3. torch.tensor(classes).shape = [20], 假设用户给定的classes只有20种，那么
        #           在预测结果中，所预测的类别不在这20种类中的，就会被筛掉
        #       4. (x[:, 5:6] == torch.tensor(classes)).shape = [358, 20], 相等比较运算
        #           会使得两个张量触发广播机制，得到大小为 [358, 20] 的结果，表示358个预测结果中，
        #           每个预测结果对应到筛选类别中的哪一个，对应的地方记为True，否则为False
        #       5. (x[:, 5:6] == torch.tensor(classes)).any(1) 表示对 [358, 20] 的第二维
        #           度取出值为 True 的，得到的结果为形状为 [358] 的张量，表示每个预测框所预测的类别，
        #           如果类别在用户给定的20个类别筛选中，记为True，否则为False
        # -----------------------------------------------------------------------------
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 获取经过conf置信度阈值筛选后的预测锚框数量
        # -----------------------------------------------------------------------------
        # 满足阈值的预测框数量： num_box = x.shape[0], 即num_box = 358
        # 如果预测框数量大于给定的NMS筛选框数量，就截断：
        #       按照预测分数scores=x[:, 4]从大到小排列后，只取前 max_nms 个，即：
        #       x[:, 4].argsort(descending=True)[:max_nms]，得到对应的index值
        # -----------------------------------------------------------------------------
        num_box = x.shape[0]
        if not num_box:
            continue
        elif num_box > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        # ---------------------------------------------------------------------------------------------------
        # 由于是多目标的NMS，因此可能会出现的一种情况是：一个区域上有两个几乎重叠的框，但是这两个框所检测
        #   的目标类别不一样，因此如果直接对其使用NMS算法，可能就会有一个框被抹除，这显然不合适！
        #
        # 因此，对于多目标的NMS，其基本的思路是：在每一个目标类别的内部自己做NMS，也就是说每一个类别分别
        #   做自己的NMS，互不相干
        #
        # 具体的实现方法是：对于每一个类别cls，乘上一个很大的offset偏移值，比如这个很大的值为10000，那
        #   么对于每个类别 [1, 2, 3, ..., 80]，得到的结果就是 offset=[10000, 20000, ..., 800000]
        #   如此一来，再将这个偏移值加到预测锚框的对角坐标位置上，就可以完全地将不同类别的锚框的位置区分开来
        #
        # 举一个简单的数字例子：
        #   1. pred=[4, 6], 假设预测的锚框为 4 个，那么类别为 pred[:, 5:6]=[[2], [1], [4], [2]], 即
        #       假设 4 个预测框所预测的类别分别为 2、1、4、2类。然后对于每个类别乘上一个相对较大的数，比如1000，
        #       就可以得到 pred[:, 5:6] * 1000 = [[2000], [1000], [4000], [2000]] 的类别偏移值，记为
        #       offset = [[2000], [1000], [4000], [2000]]。
        #   2. 锚框的位置信息为 bbox = pred[:, :4], 假设其值为：
        #       [[x1_1, y1_1, x1_2, y1_2],     [[2000],     [[x1_1+2000, y1_1+2000, x1_2+2000, y1_2+2000],
        #        [x2_1, y2_1, x2_2, y2_2],  +   [1000],  =   [x2_1+1000, y2_1+1000, x2_2+1000, y2_2+1000],
        #        [x3_1, y3_1, x3_2, y3_2],      [4000],      [x3_1+4000, y3_1+4000, x3_2+4000, y3_2+4000],
        #        [x4_1, y4_1, x4_2, y4_2]]      [2000]]      [x4_1+2000, y4_1+2000, x4_2+2000, y4_2+2000]]
        #       通过以上操作后，再进行 torchvision.ops.nms 操作，就可以保证相同类别的预测框在相对同样的区域里面“自己和自己”
        #       做 NMS 运算，从而实现多目标的 NMS
        # ---------------------------------------------------------------------------------------------------
        class_offset = x[:, 5:6] * (0 if agnostic_nms else max_wh)
        boxes, scores = x[:, :4] + class_offset, x[:, 4]
        keep_box_ids = torchvision.ops.nms(boxes, scores, iou_thres)

        # 如果经过NMS运算后，预测框数量比最大预测数 max_det 要大，此时需要对结果进行截断：
        if keep_box_ids.shape[0] > max_det:
            keep_box_ids = keep_box_ids[:max_det]

        # 将对应图片的预测结果根据 NMS 筛选后的锚框index进行选择
        output[img_idx] = x[keep_box_ids]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded
    return output  # 最后得到的 output.shape=[after_NMS, 6], after_NMS 为经过NMS筛选后的数量(不超过最大检测数max_det)
