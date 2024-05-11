import torch
import torch.nn.functional as F


def dist_calculator(gt_bboxes, anchor_bboxes):
    """compute center distance between all bbox and gt

    Args:
        gt_bboxes (Tensor): shape(bs*n_max_boxes, 4)
        anchor_bboxes (Tensor): shape(num_total_anchors, 4)
    Return:
        distances (Tensor): shape(bs*n_max_boxes, num_total_anchors)
        ac_points (Tensor): shape(num_total_anchors, 2)
    """
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = torch.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = torch.stack([ac_cx, ac_cy], dim=1)

    # ------------------------------------------------------------------------
    # 利用广播机制计算真实框和锚框之间的中心点的距离
    # gt_points = shape: [320, 2] --> [320, 1, 2]
    # ac_points = shape: [8400, 2] --> [1, 8400, 2]
    # 利用广播机制可以实现每个gt锚框中心点和8400个生成框的中心点两两计算L2距离
    # 结果 distances = shape: [320, 8400, 1]
    # ------------------------------------------------------------------------
    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, ac_points


