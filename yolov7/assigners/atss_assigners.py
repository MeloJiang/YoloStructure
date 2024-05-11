import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov7.assigners.iou2d_calculator import iou2d_calculator
from yolov7.assigners.assigner_utils import dist_calculator


class ATSSAssigner(nn.Module):
    """Adaptive Training Sample Selection Assigner"""
    def __init__(self, top_k=9, num_classes=80):
        super(ATSSAssigner, self).__init__()
        self.top_k = top_k
        self.num_classes = num_classes
        self.bg_idx = num_classes

        self.n_anchors = None
        self.batch_size = None
        self.n_max_boxes = None

    @torch.no_grad()
    def forward(self, anchor_bboxes, n_level_bboxes,
                gt_labels, gt_bboxes, mask_gt, pd_bboxes):
        """
        Adaptive Training Sample Selection算法的输入与输出：
        Inputs:
            G --> 图片的真实框的集合
            L --> 特征层的数量
            Ai--> 来自第i层特征层的锚框集合
            A --> 所有锚框的集合
            k --> 用于选取topk个锚框的参数

        Outputs:
            P --> 正样本集合
            N --> 负样本集合

        Args:
            anchor_bboxes: 表示每一个特征图上特征点的锚框的四角坐标
            mask_gt: [batch_size, max_len, 1], 表示每一张图片中，最多max_len个位置，哪些位置真实存在ground-truth锚框
        """
        self.n_anchors = anchor_bboxes.size(0)  # 8400
        self.batch_size = gt_bboxes.size(0)     # 16
        self.n_max_boxes = gt_bboxes.size(1)    # 20

        assert self.n_anchors is not None

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full(size=[self.batch_size, self.n_anchors],
                               fill_value=self.bg_idx).to(device),
                    torch.zeros([self.batch_size, self.n_anchors, 4]).to(device),
                    torch.zeros([self.batch_size, self.n_anchors, self.num_classes]).to(device),
                    torch.zeros([self.batch_size, self.n_anchors]).to(device))

        # -----------------------------------------------------------------
        # 计算真实框gt_bboxes和特征图锚框anchor_bboxes的相交面积
        #       overlaps的尺寸为： [batch_size * max_len, 8400]
        #       所表示的含义是，一个批次中16张图片，每张图片最多20个真实锚框，即总共
        #       320个锚框（其中包含一些填充的空锚框）和8400个特征图生成锚框进行两两计算
        #       所以得到的尺寸为总计：[batch_size * max_len, 8400], 每个值表示的是
        #       一个真实框和生成框的overlap的面积
        # 第二步是将overlaps的尺寸进行重置：
        #       overlaps: [16 * 20, 8400] --> [16, 20, 8400]
        # -----------------------------------------------------------------
        overlaps = iou2d_calculator(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        overlaps = overlaps.reshape([self.batch_size, -1, self.n_anchors])

        # -----------------------------------------------------------------
        # 计算真实锚框和生成锚框之间的锚框中心点的 L2 距离
        #       distances.shape = [320, 8400]
        #       ac_point.shape = [8400, 2]
        # reshape后的distances为：
        #       distances.reshape = [16, 20, 8400]
        # -----------------------------------------------------------------
        distances, ac_points = dist_calculator(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        distances = distances.reshape([self.batch_size, -1, self.n_anchors])

        # ----------------------------------------------------------------
        # 输入的参数：
        #       n_level_bboxes: [6400, 1600, 400], 将总共8400个生成框按照特征层级划分
        #       为6400个(第一层级)，1600个(第二层级)，400个(第三层级)
        #       distances: shape = [16, 20, 8400], 表示总共8400个生成框，每个生成框和
        #       每一个ground-truth锚框的中心点之间的 L2 距离
        # 根据生成框和真实框的中心点 L2 距离来进行选取
        #       得到的两个返回值是 is_in_candidate, candidate_indexes
        # ----------------------------------------------------------------
        is_in_candidate, candidate_indexes = self.select_top_k_candidates(
            distances, n_level_bboxes, mask_gt
        )

        overlaps_threshold_per_gt, iou_candidates = self.threshold_calculator(
            is_in_candidate, candidate_indexes, overlaps
        )

        # -----------------------------------------------------------------------------------
        # 计算出了每个 ground_truth 的 threshold 之后，进行筛选
        #       iou_candidates: [16, 20, 8400], 表示每个ground-truth锚框和8400个生成框的面积相交值
        #       overlaps_threshold_per_gt: [16, 20, 1], 表示每个gt锚框的阈值
        # is_positive 的生成过程：
        #       对于每一个gt，和8400个生成框进行两两计算，锚框相交面积值大于overlaps_threshold_per_gt
        #       中对应gt锚框的阈值所对应的生成框，就纳入正样本之中
        #       is_positive: [16, 20, 8400]
        # -----------------------------------------------------------------------------------
        is_positive = torch.where(
            iou_candidates > overlaps_threshold_per_gt.repeat([1, 1, self.n_anchors]),
            is_in_candidate, torch.zeros_like(is_in_candidate)
        )

        # -----------------------------------------------------------------------------------
        # 对于8400个生成框的中心坐标点，计算每一个中心坐标点是否在ground-truth锚框内部
        # 如果在gt真实锚框的内部，则对应的8400个生成框的位置记为True，否则记为False
        # is_in_gts: [16, 20, 8400]
        # is_positive: [16, 20, 8400]
        # mask_gt: [16, 20, 1]
        # mask_positive = is_positive * is_in_gts * mask_gt
        # -----------------------------------------------------------------------------------
        is_in_gts = self.select_candidates_in_gts(ac_points, gt_bboxes)
        mask_positive = is_positive * is_in_gts * mask_gt

        target_gt_idx, fg_mask, mask_positive = self.select_highest_overlaps(
            mask_positive, overlaps, self.n_max_boxes
        )

        # ---------------------------------------------------------------
        # 分配目标部分 >> 输入参数解析：
        #       gt_labels: shape=[16, 20, 1]
        #       gt_bboxes: shape=[16, 20, 4]
        #       target_gt_idx: shape=[16, 8400]
        #       fg_mask: shape=[16, 8400]
        #
        # ---------------------------------------------------------------
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # ---------------------------------------------------------------
        #
        #
        # ---------------------------------------------------------------
        if pd_bboxes is not None:
            iou_tag = overlaps.max(dim=1)[0]
            iou_mask = iou_tag * fg_mask
        else:
            iou_mask = fg_mask

        """
        输出内容的解析：
        1. target_labels.long() (Tensor): shape=[16, 8400], 表示16张图片中，每张图片总共
            8400个生成框，每个生成框上如果有和gt框关联，那么这个生成框所框住的实体类别是什么，取值
            范围是 0 ~ 79 (因为总共80个实体类别)
        2. target_bboxes (Tensor): shape=[16, 8400, 4], 表示16张图片中，每张图片总共8400个
            生成框，每个生成框上是否与gt框相关联，如果和gt框关联，那么这个生成框所关联的gt框的大小尺
            寸是多少，使用4个值的对角坐标来表示
        3. target_scores (Tensor): shape=[16, 8400, 80], 表示16张图片中，每张图片总共8400个
            生成框，每个生成框是否和gt框相关联，如果和gt框相关联，那么这个生成框所框住的实体类别是什么
            用one_hot编码来表示，并且以此来作为该类别的分数scores
        4. fg_mask (Tensor): shape=[16, 8400], 表示16张图片中，每张图片总共8400个生成框，每个
            生成框是否和gt框有关联，如果这个生成框和gt框有关联，就设置为 1 ；没有关联则设置为 0
        """
        return target_labels.long(), target_bboxes, target_scores, fg_mask.bool(), iou_mask

    def select_top_k_candidates(self, distances, n_level_bboxes, mask_gt):
        # ---------------------------------------------------------------
        # 首先将 mask_gt 进行改造：
        #       [16, 20, 1] --> [16, 20, 9]
        # ---------------------------------------------------------------
        mask_gt = mask_gt.repeat(1, 1, self.top_k).bool()

        # ---------------------------------------------------------------
        # 1. n_level_bboxes = [6400, 1600, 400]
        # 2. distances.shape = [16, 20, 8400]
        # 3. 对输入 distances 进行切分处理，切分依据是 1 中的 n_level_bboxes，
        #       经过切分后的结果为
        #       level_distances = [[16, 20, 6400],
        #                          [16, 20, 1600],
        #                          [16, 20, 400]]
        #       分别表示每一个特征层级的生成框和gt框的中心点 L2 距离
        # ---------------------------------------------------------------
        level_distances = torch.split(distances, n_level_bboxes, dim=-1)
        is_in_candidate_list = []
        candidate_indexes = []
        start_index = 0

        for per_level_distances, per_level_boxes in zip(level_distances, n_level_bboxes):
            end_index = start_index + per_level_boxes
            selected_k = min(self.top_k, per_level_boxes)

            # ----------------------------------------------------------------------------
            # per_level_top_k_indexes 的尺寸为: [batch_size, max_len, top_k]
            # 这个变量所存储的是每个特征层级上的符合条件的 top_k 个锚框的索引 index 值
            #       将每个层级的 per_level_top_k_indexes 加入到candidate_indexes列表中，并使用
            #       torch.cat操作进行合并可以得到：
            #       [[16, 20, 9], [16, 20, 9], [16, 20, 9]] --> [16, 20, 9+9+9]
            # ----------------------------------------------------------------------------
            _, per_level_top_k_indexes = per_level_distances.topk(selected_k, dim=-1, largest=False)
            candidate_indexes.append(per_level_top_k_indexes + start_index)
            per_level_top_k_indexes = torch.where(mask_gt, per_level_top_k_indexes,
                                                  torch.zeros_like(per_level_top_k_indexes))

            # -------------------------------------------------------------------------
            # 获得每个特征层的单元格上是否存在符合top_k的候选锚框的步骤：
            # 1.首先初步经过one_hot编码处理
            #       F.one_hot(per_level_top_k_indexes, per_level_boxes)
            #       得到的尺寸是: [16, 20, 9, 6400], [16, 20, 9, 1600], [16, 20, 9, 400]
            # 2.再把倒数第二维度的进行加和处理
            #       得到的尺寸是： [16, 20, 6400], [16, 20, 1600], [16, 20, 400]
            # 3.然后再使用where函数进行处理，对于每个生成框的位置上，如果该生成框与0个或1个gt框关联，
            #       保留原值；如果和2个或以上的gt框关联，则值置为0
            # 4.最后将第二步中的结果添加进is_in_candidate_list中
            #       得到的尺寸是： [[16, 20, 6400], [16, 20, 1600], [16, 20, 400]]
            # -------------------------------------------------------------------------
            is_in_candidate = F.one_hot(per_level_top_k_indexes, per_level_boxes).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1,
                                          torch.zeros_like(is_in_candidate), is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
            start_index = end_index

        # -----------------------------------------------------------------------------
        # 将上述第三步中的 is_in_candidate_list 进行 torch.cat 操作
        #    [[16, 20, 6400], [16, 20, 1600], [16, 20, 400]] --> [16, 20, 640+1600+400]
        #     --> [16, 20, 8400]
        #
        # 将上述第二步中的 candidate_indexes 进行 torch.cat 操作
        #    [[16, 20, 9], [16, 20, 9], [16, 20, 9]] --> [16. 20. 9+9+9], 表示16张图片中，
        #    每张图片最多20个gt框中，符合top_k条件的生成框的索引位置值 (0 ~ 8399) 因为总共有8400个
        #    生成框
        # -----------------------------------------------------------------------------
        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_indexes = torch.cat(candidate_indexes, dim=-1)

        """
        输出内容的解析：
        1. is_in_candidate_list (Tensor): shape=[16, 20, 8400], 表示16张图片，每张图片最多20个gt框，
            每个gt框与8400个生成框中的哪些相关联（通过top_k筛选），相关联的生成框的位置取值为 1，否则为 0
            注意点：
                从top_k个生成框到 is_in_candidate_list 经过了两个where函数的处理，第一个where函数的处理
                是，根据mask_gt，如果这个gt框存在，那么top_k候选人保持原样；如果这个gt框不存在，则top_k候
                选人取值 0 ；第二个where操作则是发生在one_hot编码之后，在top_k这一维度上面加和之后，检查每
                个生成框的取值是否大于 1 ，如果是，则说明这个地方不存在gt框，top_k取值都是0，因此one_hot编
                码相同，加和之后自然大于1，需要置0，其他地方保留原值即可
        2. candidate_indexes (Tensor): shape=[16, 20, 27], 表示16张图片，每张图片最多20个gt框，每个层
            级top_k个满足条件的候选人的位置索引值（总共三个层级，一共就 3 * top_k 个候选人），索引值表示8400个
            生成框的位置索引值
        """
        return is_in_candidate_list, candidate_indexes

    def threshold_calculator(self, is_in_candidate, candidate_indexes, overlaps):
        """
        Adaptive Training Sample Selection
        接受输入的参数为：
            1. is_in_candidate: [16, 20, 8400]
            2. candidate_indexes: [16, 20, 27]
            3. overlaps: [16, 20, 8400]
        计算Threshold是算法的第二大步，需要计算的部件有：
            1. gt_bboxes和top_k_box的IoU值
            2. IoU值的平均值：Mean(IoU)
            3. IoU值的标准差：Std(IoU)
        """
        n_batch_size_max_boxes = self.batch_size * self.n_max_boxes  # 16 * 20 = 320

        # --------------------------------------------------------------------------
        # 首先根据每个特征图层级的像素点中是否存在候选框进行overlaps值的标记
        # 当总共8400个特征格点中，存在top_k要求的候选框时，这个位置填上候选框和真实框的overlaps值
        #       _candidate_overlaps 的尺寸为： [16, 20, 8400]
        # --------------------------------------------------------------------------
        _candidate_overlaps = torch.where(
            is_in_candidate > 0, overlaps, torch.zeros_like(overlaps)
        )

        # --------------------------------------------------------------------------
        # shape: [16, 20, 27] --> [320, 27]
        # assist_indexes 的尺寸为：[320], 内容是 [0, 8400, 16800, ..., 2679600]
        # --------------------------------------------------------------------------
        candidate_indexes = candidate_indexes.reshape([n_batch_size_max_boxes, -1])
        assist_indexes = self.n_anchors * torch.arange(n_batch_size_max_boxes,
                                                       device=candidate_indexes.device)
        assist_indexes = assist_indexes[:, None]  # shape: [320, 1]
        flatten_indexes = candidate_indexes + assist_indexes  # shape: [320, 27]

        candidate_overlaps = _candidate_overlaps.reshape(-1)[flatten_indexes]
        candidate_overlaps = candidate_overlaps.reshape([self.batch_size, self.n_max_boxes, -1])  # [16, 20, 27]

        overlaps_mean_per_gt = candidate_overlaps.mean(dim=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(dim=-1, keepdim=True)
        overlaps_threshold_per_gt = overlaps_std_per_gt + overlaps_mean_per_gt  # shape:[16, 20, 1]

        return overlaps_threshold_per_gt, _candidate_overlaps

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        """选择那些出现在ground-truth的锚框中的特征图的中心点位

        Args:
            xy_centers (Tensor): shape-->[batch_size * n_max_boxes, num_total_anchors, 4]
            gt_bboxes (Tensor): shape-->[batch_size, n_max_boxes, 4]
            eps (float): 此参数的作用是防止出现除以0的情况
        """
        n_anchors = xy_centers.size(0)
        _gt_bboxes = gt_bboxes.reshape([-1, 4])  # shape: [320, 4]
        xy_centers = xy_centers.unsqueeze(0).repeat(self.batch_size * self.n_max_boxes, 1, 1)

        # --------------------------------------------------------------------
        # 1. gt_bboxes_lt 表示ground_truth锚框的左上角坐标位置，构建过程为
        #       _gt_bboxes:[320, 2] --> [320, 1, 2] --> [320, 8400, 2]
        # 2. gt_bboxes_rb 表示ground_truth锚框的右下角坐标位置，构建过程为
        #       _gt_bboxes:[320, 2] --> [320, 1, 2] --> [320, 8400, 2]
        # --------------------------------------------------------------------
        gt_bboxes_lt = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
        gt_bboxes_rb = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)

        # -------------------------------------------------------------------------------------
        # 3. b_lt 表示ground-truth锚框的左上角坐标点对于每一个特征图中心点坐标的距离
        #       [320, 8400, [center_x, center_y]] - [320, 8400, [gt_top_left_x, gt_top_left_y]]
        #       得到的b_lt为--> [320, 8400, [w_x, h_x]]
        # 4. b_rb 表示ground-truth锚框的右下角坐标点对于每一个特征图中心点坐标的距离
        #       [320, 8400, [gt_bottom_right_x, gt_bottom_right_y]] - [320, 8400, [c_x, c_y]]
        #       得到的b_rb为--> [320, 8400, [w_x, h_x]]
        # -------------------------------------------------------------------------------------
        b_lt = xy_centers - gt_bboxes_lt
        b_rb = gt_bboxes_rb - xy_centers

        # -------------------------------------------------------------------------------------
        # 5. 将上述的对角坐标到中点的距离进行torch.cat 拼接操作
        #       [320, 8400, [lt2c_x, lt2c_y]] + [320, 8400, [rb2c_x, rb2c_y]] -->
        #       bbox_deltas = [320, 8400, [lt2c_x, lt2c_y, rb2c_x, rb2c_y]]
        # 6. 再将bbox_deltas进行reshape操作
        #       [320, 8400, [lt2c_x, lt2c_y, rb2c_x, rb2c_y]] -->
        #       [16, 20, 8400, [lt2c_x, lt2c_y, rb2c_x, rb2c_y]]
        # -------------------------------------------------------------------------------------
        bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
        bbox_deltas = bbox_deltas.reshape([self.batch_size, self.n_max_boxes, n_anchors, -1])

        # -------------------------------------------------------------------------------------
        # 7. 最后将ground-truth对角对于每个特征图的中心点坐标的距离小于预设值eps的设置为False
        #    而将大于eps的ground-truth的锚框设置为True，这相当于一个mask，用于标示8400个特征图中心点，哪些
        #    中心点落在ground-truth锚框的内部，哪些没有落在ground-truth内部
        #    结果的尺寸为 : [16, 20, 8400]
        # -------------------------------------------------------------------------------------
        return (bbox_deltas.min(dim=-1)[0] > eps).to(gt_bboxes.dtype)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Args:
            mask_pos (Tensor): shape=[16, 20, 8400]
            overlaps (Tensor): shape=[16, 20, 8400]
            n_max_boxes (Int): 表示一张图片中最多几个gt框
        """
        fg_mask = mask_pos.sum(dim=-2)  # shape=[16, 8400]
        # fg_mask.max() > 1 表示8400个生成框中存在1个生成框对应2个或2个以上的gt真实框
        if fg_mask.max() > 1:
            # ---------------------------------------------------------------------
            # (fg_mask.unsqueeze(1) > 1) >> [16, 1, 8400], 得到的结果表示总共
            # 8400个生成框中，一个生成框对应多个gt真实框的情况，符合这样情况的生成框标记为True
            # 然后将 fg_mask 的第二维度重复 n_max_boxes 次，获得mask_multi_gts
            # mask_multi_gts = [16, 20, 8400]
            # ---------------------------------------------------------------------
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])

            # ---------------------------------------------------------------------
            # overlaps.shape = [16, 20, 8400], 8400个生成框，对于每个生成框，其与每个gt框
            # 的相交面积最大的那个gt框的索引位置记录下来，即 overlaps.argmax(dim=1)
            # 得到的结果 max_overlaps_idx = [16, 8400], 8400个生成框，每个生成框最大iou的
            # gt框的索引位置 (0~19, 因为一个图片中总共最多20个gt框)
            # ---------------------------------------------------------------------
            max_overlaps_idx = overlaps.argmax(dim=1)

            # ---------------------------------------------------------------------
            # 对于gt框的索引位置再进行独热编码一下可得：
            #       is_max_overlaps: shape = [16, 8400, 20]
            # 然后再对 is_max_overlaps 进行permute操作：
            #       is_max_overlaps: shape = [16, 20, 8400]
            # 最后使用torch.where来重新构建mask掩码位置：
            #       总共8400个生成框中，那些一个生成框对应多个gt框的位置，选择生成框与gt框相交
            #       面积最大的gt框的位置，标上True；而一个生成框仅对应一个gt框的情况，填入原来
            #       的mask_pos即可
            #
            # 通过上述操作，将所有的8400个生成框都处理成一个生成框只对应一个gt框的情况（那些一个
            # 生成框对应多个gt框的，选择相交面积最大的gt框作为对应框）
            # ---------------------------------------------------------------------
            is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)

            # ---------------------------------------------------------------------
            # 将 mask_pos=[16, 20, 8400] 的第二维度求和，就可以得到 shape=[16, 8400]
            # 在这8400个生成框中，取值只有 0 或 1 ，代表该生成框是否有对应的gt框，1 表示有；
            # 0 表示没有
            # ---------------------------------------------------------------------
            fg_mask = mask_pos.sum(dim=-2)

        # ---------------------------------------------------------------------
        # 最后，对于 mask_pos=[16, 20, 8400] 的第二维度求argmax，即可得到8400个生成框
        # 对应的gt框的索引号（0~19，因为max_len=20）
        # ---------------------------------------------------------------------
        target_gt_idx = mask_pos.argmax(dim=-2)  # target_gt_idx.shape=[16, 8400]

        """
        最后的输出的解析：
        1. target_gt_idx (Tensor): shape=[16, 8400], 表示一张图片总计8400个生成框，每个生成框所对应的gt框的索引取值，
            因为一张图片中最多也就20个gt框（max_len），因此8400个生成框的取值范围是：0 ~ 19
        2. fg_mask (Tensor): shape=[16, 8400], 表示一张图片总计8400个生成框，每个生成框是否存在对用的gt框，如果存在对
            应的gt框，取值为 1；不存在对应gt框则取值为 0
        3. mask_pos (Tensor): shape=[16, 20, 8400], 表示一张图片总计8400个生成框，每个生成框在一张图片中，与图片中的
            哪个gt框相对应，比如这个例子中一张图片最多20个gt框，如果生成框与这20个gt框中的某个对应，就在这个位置取值 1；其他
            地方取值 0 
        """
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        # ---------------------------------------------------------------------------
        # 输入参数的解析：
        # gt_labels (Tensor): shape=[16, 20, 1]
        # gt_bboxes (Tensor): shape=[16, 20, 4]
        # target_gt_idx (Tensor): shape=[16, 8400]
        # ---------------------------------------------------------------------------
        # assigned target labels
        batch_index = torch.arange(self.batch_size, dtype=gt_labels.dtype,
                                   device=gt_labels.device)
        batch_index = batch_index[..., None]

        # --------------------------------------------------------------------------
        # target_gt_idx 的尺寸为：[16, 8400]
        # 而 batch_index * self.n_max_boxes == [[0], [20], ..., [300]]
        # 并且 batch_index * self.n_max_boxes 的尺寸为：[16, 1]
        # 因此 target_gt_idx + batch_index * self.n_max_boxes >> [16, 8400] + [16, 1]
        # 这样会使用广播机制进行相加
        # target_gt_idx.flatten() >> shape=[16 * 8400]
        # 当 gt_labels.flatten() 展平之后，尺寸变为 [16 * 20]
        # gt_labels.flatten()[target_gt_idx.flatten()] 即表示根据8400个生成框，每个生成框
        # 对应的gt框的索引值，来从gt_labels中选出对应gt框的目标类别标签
        # 最终得到的 target_labels 表示8400个生成框中，存在对应gt框的生成框，它的对应gt框的目标
        # 类别是什么
        # --------------------------------------------------------------------------
        target_gt_idx = (target_gt_idx + batch_index * self.n_max_boxes).long()
        target_labels = gt_labels.flatten()[target_gt_idx.flatten()]  # [16 * 8400]
        target_labels = target_labels.reshape([self.batch_size, self.n_anchors])
        target_labels = torch.where(
            fg_mask > 0, target_labels, torch.full_like(target_labels, self.bg_idx)
        )  # [16, 8400]

        # assigned target boxes
        # --------------------------------------------------------------------------
        # 1. gt_bboxes=[16, 20, 4], 表示图片最多20个gt框，每个gt框的四角坐标信息
        #       对其进行reshape操作得到 [16 * 20, 4]
        # 2. target_gt_idx=[16, 8400], 表示每张图片总计8400个生成框，每个生成框对应的gt框的
        #       索引位置 (0 ~ 19)，因为本例中max_len=20，即每张图片gt框最多也就20个，对其进行
        #       展平操作得到：target_gt_idx.flatten() = [16 * 8400]
        # 3. 使用 target_gt_idx 来选择出8400个生成框每个框对应的gt框的四角坐标信息，总共为
        #       [16 * 8400, 4], 表示16张图片，每张图片8400个生成框，每个生成框对应的gt框的四角
        #       坐标信息
        # 4. 最后将选择好的 target_bboxes 进行reshape操作，可以得到：
        #       [16 * 8400, 4] --> [16, 8400, 4]
        # --------------------------------------------------------------------------
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx.flatten()]  # [16 * 8400, 4]
        target_bboxes = target_bboxes.reshape([self.batch_size, self.n_anchors, 4])  # [16, 8400, 4]

        # assigned target scores
        # [16, 8400, 81] --> [16, 8400, 80]
        # --------------------------------------------------------------------------
        # 1. target_labels=[16, 8400], 表示16张图片中，每张图片的8400个生成框，每个生成框
        #       是否和gt框关联，如有关联，取值则是对应gt框中的目标实体的类别 (0 ~ 79, 因为总共80种)
        # 2. 对 target_labels 作one_hot编码，可以得到 target_scores=[16, 8400, 81], 即
        #       每个类别有自己的独热编码，可以看作这个类别的分数scores，最后取前80位，最终得到
        #       target_scores=[16, 8400, 80]
        # --------------------------------------------------------------------------
        target_scores = F.one_hot(target_labels.long(), self.num_classes + 1).float()
        target_scores = target_scores[:, :, :self.num_classes]

        """
        输出内容的解析：
        1. target_labels (Tensor): shape=[16, 8400], 表示16张图片每张图片总共8400个生成框，
            每个生成框是否有和gt框相关联，如果有的话，取值就是关联gt框的目标实体类别，取值范围 0 ~ 79(总共80类目标)
        2. target_bboxes (Tensor): shape=[16, 8400, 4], 表示16张图片每张图片总共8400个生成框，
            每个生成框是否和gt框有关联，如果有的话，取值就是关联的哪个gt框的四角坐标位置信息
        3. target_scores (Tensor): shape=[16, 8400, 80], 表示16张图片每张图片总共8400个生成框，
            每个生成框是否和gt框有关联，如果有的话，取值就是关联的那个gt框的目标类别的one_hot编码，作为类别分数scores
        """
        return target_labels, target_bboxes, target_scores


if __name__ == '__main__':
    pass
