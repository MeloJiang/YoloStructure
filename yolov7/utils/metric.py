# Model validation metrics
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import warnings
import seaborn as sns
from yolov7.utils import general, nms


class PRMetric:

    @staticmethod
    def ap_per_class(tp, conf, pred_cls, target_cls, plot=True, save_dir='.', names=()):
        # 根据物体置信度进行排序
        i = np.argsort(-conf)

        # ------------------------------------------------------------------------------
        # 对输入变量的详解：
        #   1. tp.shape=[550, 10], 550表示这一次的验证数据的推理结果为总共550个预测框，10则表示总
        #       共有10个iou的阈值 (0.5 ~ 0.95)，总体的表示是在不同的iou阈值之下，该预测框是否为真阳
        #       性(TP)，是的话为True；否则置为False
        #   2. conf.shape=(550, ), 550表示所有的预测框的预测置信度，总共550个预测置信度(已按照从大
        #       到小排列)
        #   3. pred_cls.shape=(550, ), 550表示所有的预测框，每个预测框预测到的物体类别(已按照预测
        #       置信度从大到小排序)
        # ------------------------------------------------------------------------------
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]  # 根据预测的置信度conf进行从大到小的排序

        # 找到独一无二的类（其实就是统计一下类别数量）
        unique_classes = np.unique(target_cls)
        nc = unique_classes.shape[0]

        # ------------------------------------------------------------------------------
        # 创建的变量详解：
        #   1. px.shape=[1000, ]: 这是一个np.ndarray()类型数组，所存储的内容是
        #       [0.000, 0.001, ..., 1.000], 长度为1000的数组，步长为0.001
        #   2. py = [], py暂时为空列表
        # ------------------------------------------------------------------------------
        px, py = np.linspace(0, 1, 1000), []

        # ------------------------------------------------------------------------------
        # 所创建的变量的详解：
        #   1. ap.shape=[80, 10], 80表示的是总共有80个物体类别(coco数据集之下), 10表示的是共有10
        #       个iou的阈值，即 (0.5 ~ 0.95) 表示总共10个iou的阈值，因此，ap的含义是总共80个物体种
        #       类中，每个种类在不同的iou阈值之下的平均预测准确率(average precision, AP)
        #   2. p.shape=[80, 1000], 80表示的是总共80个物体类别(coco数据集之下), 1000表示的是p的数
        #       组长度，表示从高到低排序的预测准确率precision，用于绘制PR曲线图
        #   3. r.shape=[80, 1000], 80表示的是总共80个物体类别(coco数据集之下), 1000表示的是r的数
        #       组长度，表示从低到高排序的召回率recall，用于绘制PR曲线图
        # ------------------------------------------------------------------------------
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            num_labels = (target_cls == c).sum()  # 计算指定类别的个数(真实标签的总数)
            num_predictions = i.sum()

            if num_predictions == 0 or num_labels == 0:
                continue
            else:
                # 累计的 FP(假阳性) 和 TP(真阳性)
                fp_accumulate = (1 - tp[i]).cumsum(0)
                tp_accumulate = tp[i].cumsum(0)

                # -----------------------------------------------------------------------------------
                # 召回值：用正确预测的数量TP(真阳性) / 该类别的标签总数 = 准确地查找出的比例，即召回率
                #   1. tp_accumulate.shape=[113, 10]（对于‘person’这个类别为例，此次验证推理共有113个预测结
                #       果），10 表示iou阈值从 (0.5 ~ 0.95), 步长为0.05，每个阈值之下的该预测框是否是TP(真阳性)
                #   2. num_labels 表示这一批验证推理数据的该类别（本例为person类别）的锚框总数
                #   3. recall 表示的是该类别（本例为person类别），使用 tp_accumulate 除以 总的该类别的锚框数量，
                #       可以得到召回率，也就是从这个类的锚框数中，真正成功检验出来的比率，也就是真阳性比率
                #
                # 插值操作：
                #   1. np.interp插值函数的参数以及作用：
                #       (1) x: 表示要进行插值的目标点或一组目标点
                #       (2) xp: 已知数据点的 x 值，必须按照升序排列
                #       (3) fp: 已知数据点的对应的 y 值
                #       应用举例：
                #           # 已知的数据点
                #           xp = [1, 2, 3, 4, 5]
                #           fp = [10, 20, 30, 40, 50]
                #           # 目标点数组
                #           x = np.array([1.5, 2.5, 3.5, 4.5])
                #           # 进行线性插值
                #           result = np.interp(x, xp, fp)  >>  [15., 25., 35., 45.]
                #   2. 对于本例求recall的对应值
                #       px = [0.000, 0.001, 0.002, ..., 1.000]长度为1000，步长为0.001的数组
                #       conf[i] = [降序排列的该类别person类别的预测置信度]
                #       recall[:, 0] = [升序排列的召回率]
                #       """
                #       经过线性插值函数np.interp的处理，从高到低的预测框的conf置信度对应了每一个预测框的到目前为
                #       止的accumulate的召回率，因为参数xp要求输入的数组必须要升序排列，因此在降序排列的置信度conf
                #       上加一个负号即可；最终需要产生插值的x值px，同样也需要添加负号，因为要与"低置信度--高召回率"方
                #       向保持一致；最终得到的y值结果就是从高到低排列的召回率recall，数组长度为1000
                #       """
                # -----------------------------------------------------------------------------------
                recall = tp_accumulate / (num_labels + 1e-16)
                r[ci] = np.interp(x=-px, xp=-conf[i], fp=recall[:, 0], left=0)

                # -----------------------------------------------------------------------------------
                # 精确率：用正确的预测数量TP(真阳性) / {正确的预测数量 + 预测为正类别但是错误的数量}
                #   1. fp_accumulate.shape=[113, 10]（对于‘person’这个类别为例，此次验证推理共有113个预测结
                #       果），10 表示iou阈值从 (0.5 ~ 0.95), 步长为0.05，每个阈值之下的该预测框是否为预测假阳
                #       性(FP)，其中如果这个预测框不是TP真阳性，那就是FP假阳性，因为该类别的预测框没有和真实框匹配、
                #       或者和其他类别的框匹配，都算作是假阳性FP，因此计算方法上直接用 1 - tp_accumulate 即可得到
                #   2. precision预测准确率的计算方法是 tp / (tp + fp)，表示在作出一个类别的预测的情况下，真正准确
                #       地预测出该类别的比率
                #
                # 求插值操作：
                #   px = [0.000, 0.001, 0.002, ..., 1.000]长度为1000，步长为0.001的数组
                #   conf[i] = [降序排列的该类别person类别的预测置信度]
                #   precision[:, 0] = [降序排列的预测准确率]
                #   """
                #   从高到低的预测框的conf预测置信度对应了每一个预测框的到目前为accumulate的预测准确率precision，
                #   因为参数xp要求输入必须要升序排列，因此在降序排列的置信度conf上加一个负号即可；最终需要产生插值的x
                #   值px，同样也需要添加负号，为了和加了负号的conf值保持方向一致；最终得到的y值结果就是从低到高排序的
                #   预测准确率precision，数组长度为1000
                #   """
                # -----------------------------------------------------------------------------------
                precision = tp_accumulate / (tp_accumulate + fp_accumulate)
                p[ci] = np.interp(x=-px, xp=-conf[i], fp=precision[:, 0], left=1)

                # 使用 recall 和 precision 来构建出 PR曲线，然后再根据 PR曲线计算出 AP
                for j in range(tp.shape[1]):
                    # ------------------------------------------------------------------
                    # 1. ap[ci, j] 表示第ci个种类的第j个iou阈值之下的平均预测准确率ap
                    # 2. mean_precision 表示的是第ci个种类的第j个iou阈值之下的预测准确率(PR曲线中
                    #       的P的取值，从大到小排序)
                    # 3. mean_recall 表示的是第ci个种类的第j个iou阈值之下的召回率(PR曲线中的R的取
                    #       值，从小到大配许)
                    # ------------------------------------------------------------------
                    ap[ci, j], mean_precision, mean_recall = PRMetric.compute_ap(recall[:, j], precision[:, j])
                    if plot and j == 0:
                        # py用于计算 mAP@0.5 的预测准确率，并且只计算 iou=0.5 的情况，使用线性插值函数求取
                        # 长度为 1000 的预测准确率precision值，以便后续绘制PR曲线时使用(每个物体类别都有一
                        # 个长度为 1000 的precision数组，则列表py的长度为80，物体种类的数量)
                        py.append(np.interp(x=px, xp=mean_recall, fp=mean_precision))

        # 计算F1值，f1.shape=[80, 1000]，表示总共80个物体种类，每个种类的F1值(长度为1000，用于后续绘制F1曲线)
        f1 = 2 * p * r / (p + r + 1e-16)

        if plot:
            PRMetric.plot_pr_curve(px, py, ap, str(Path(save_dir) / 'PR_curve.png'), names)
            PRMetric.plot_mc_curve(px, f1, str(Path(save_dir) / 'F1_curve.png'), names, ylabel='F1')
            PRMetric.plot_mc_curve(px, p, str(Path(save_dir) / 'P_curve.png'), names, ylabel='Precision')
            PRMetric.plot_mc_curve(px, r, str(Path(save_dir) / 'R_curve.png'), names, ylabel='Recall')
        return p, r, ap, f1, unique_classes.astype(int)

    @staticmethod
    def process_one_image_prediction(detections, labels):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        # --------------------------------------------------------------------------------
        #
        # --------------------------------------------------------------------------------
        iou_vector = torch.linspace(0.5, 0.95, 10)
        correct = np.zeros((detections.shape[0], iou_vector.shape[0])).astype(bool)
        iou = general.box_iou(labels[:, 1:], detections[:, :4])  # shape=[N, M]
        correct_class = labels[:, 0:1] == detections[:, 5]  # shape=[N, M]

        for i in range(len(iou_vector)):
            # 选择出iou取值既大于阈值，预测框的预测物体种类又等于真实框种类的位置
            iou_correct = iou >= iou_vector[i]
            x = torch.where(iou_correct & correct_class)  # shape = 2 * [7]

            if x[0].shape[0]:
                # 得到的 matches 是一张图片的预测结果中，既满足与真实框的iou大于阈值，又要满足和真实框的物体类别一致
                #   例如：
                #       [[ 0.          1.          0.61136556]
                #        [ 0.          9.          0.56145734]
                #        [ 1.          2.          0.8350369 ]
                #        [ 1.         15.          0.60688615]
                #        [ 1.         55.          0.6360255 ]
                #        [ 2.          0.          0.95228755]
                #        [ 2.         59.          0.5457003 ]]
                #
                # -------------------------------------------------------------------------------------
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    # ---------------------------------------------------------------------------------
                    # 对于 matches 的处理：
                    #   1. 首先将matches的预测框和真实框的iou从大到小排列
                    #   2. 然后对于一个预测框对应多个真实框的情况下，需要取出那些iou较小的预测框，只保留一个iou值最大
                    #       的预测框，与一个真实框相关联
                    #   3. 然后对于一个真实框被多个预测框对应的情况下，需要去除那些iou较小的真实框，只保留一个iou值最
                    #       大的真实框，与一个预测框对应
                    #   4. 经过上述三个步骤后，可以得到一个真实框和一个预测框之间对应的关系，因为前面经过了类别一致的筛
                    #       选(即labels[:, 0:1] == detections[:, 5])，因此经过筛选后的 真实框-预测框 的对应关
                    #       系都是类别一致的，所以是 TP (真阳性)
                    # ---------------------------------------------------------------------------------
                    matches = matches[matches[:, 2].argsort()[::-1]]  # 根据iou从大到小排序
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                # 在对应的预测结果中填入这个预测框是否预测对了真实框，也就是说这个预测框是否预测的是TP(真阳性)
                correct[matches[:, 1].astype(int), i] = True  # correct 表示预测类 == 真实类，即true_positive

        return torch.tensor(correct, dtype=torch.bool, device=iou_vector.device)

    @staticmethod
    def compute_ap(recall, precision):
        # -----------------------------------------------------------------
        # 接受的输入的具体解析：
        #   1. mean_recall.shape=[113,]: 表示的是指定的物体种类的，某一个iou阈值
        #       (0.5 ~ 0.95) 之下的召回率
        #   2. mean_precision.shape=[113,]: 表示的是指定的物体种类的，某一个iou
        #       阈值 (0.5 ~ 0.95) 之下的预测准确率
        # -----------------------------------------------------------------
        mean_recall = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
        mean_precision = np.concatenate(([1.], precision, [0.0]))

        # 计算precision的各个断点
        # ------------------------------------------------------------------------
        # 函数np.flip, np.maximum.accumulate的含义解析：
        #   1. np.flip(): 该函数的作用是将输入的numpy数组进行数组翻转
        #   2. np.maximum.accumulate(): 该函数的作用是求出给定数列的累积最大值，考虑一下的
        #       例子：
        #           [3, 8, 2, 10, 1, 5] --> [3, 8, 8, 10, 10, 10]
        #       由以上例子可知，使用accumulate函数的目的是保持数列的单调递增的特性
        # ------------------------------------------------------------------------
        mean_precision = np.flip(np.maximum.accumulate(np.flip(mean_precision)))

        # method='interp' 表示使用的是插值模式，通过更细粒度的插值来构建PR曲线
        method = 'interp'
        if method == 'interp':
            # x 在这里代表的是间隔粒度更小的recall值(0.0 ~ 1.00, 间隔粒度为0.01)
            x = np.linspace(0, 1, 101)
            # np.trapz() 函数用于计算定积分，在这里计算PR曲线下方的面积，表示AP的值
            ap = np.trapz(np.interp(x, mean_recall, mean_precision), x)

        else:
            i = np.where(mean_recall[1:] != mean_recall[:-1])[0]
            ap = np.sum((mean_recall[i + 1] - mean_recall[i]) * mean_precision[i + 1])

        return ap, mean_precision, mean_recall

    @staticmethod
    def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
        # 绘制P-R曲线
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        py = np.stack(py, axis=1)

        if 0 < len(names) < 21:
            for i, y in enumerate(py.T):
                ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')
        else:
            ax.plot(px, py, linewidth=1, color='blue')

        ax.plot(px, py.mean(1), linewidth=1, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.savefig(Path(save_dir), dpi=250)

    @staticmethod
    def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
        # 绘制Metric-confidence曲线
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        if 0 < len(names) < 21:
            for i, y in enumerate(py):
                ax.plot(px, y, linewidth=1, label=f'{names[i]}')
        else:
            ax.plot(px, py.T, linewidth=1, color='grey')

        y = py.mean(0)
        ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.savefig(Path(save_dir), dpi=250)


class ConfusionMatrix:
    def __init__(self, num_classes, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.nc = num_classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        # ----------------------------------------------------------------------------------------
        #
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)  # shape = ([M], [N])

        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), dim=1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = torch.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = np.array(matches.T).astype(int)  # m0=gt类别索引， m1=dt类别索引
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                # ---------------------------------------------------------------
                # 如果真实类别gt_classes找到了与之匹配的预测结果，如果真实框类别与预测框的
                # 类别一致，说明正确地预测了，属于TP(真阳性)；但是如果真实框类别和预测框的类
                # 别不一致，就说明类别预测错了，预测到了正样本，但实际上的真实框是个负样本，因
                # 此属于FP(假阳性)
                # ---------------------------------------------------------------
                self.matrix[detection_classes[m1[j]], gc] += 1
            else:
                # ---------------------------------------------------------------
                # 如果真实类别gt_classes没有预测结果与之匹配，就说明这是一个漏检的真实框，
                # 属于FN(假阴性)，也就是说预测框的预测内容是背景值，相对于这个真实框的类别，
                # 预测值是个负类，并且预测错误(因为这个预测值本应预测得到的结果是这个真实框
                # 的类别)
                # ---------------------------------------------------------------
                self.matrix[self.nc, gc] += 1

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    # --------------------------------------------------------------
                    # 这个代码块的含义是，对于预测类别 detection_classes 中的类，不在那些与真
                    # 实框匹配的预测框上，也就是说这个类的预测框没有和真实框匹配，预测框框住的是背
                    # 景，因此被判定为背景误检，属于FP(假阳性)
                    # --------------------------------------------------------------
                    self.matrix[dc, self.nc] += 1

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(axis=1) - tp
        fn = self.matrix.sum(axis=0) - tp  # 漏检
        return tp[:-1], fp[:-1], fn[:-1]

    def plot(self, normalize=True, save_dir='', names=()):
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)
        array[array < 0.005] = np.nan

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)
        sns.set(font_scale=1.0 if nc < 50 else 0.8)
        labels = (0 < nn < 99) and (nn == nc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.heatmap(
                array, annot=nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f',
                xticklabels=list(names)+['background FP'] if labels else 'auto',
                yticklabels=list(names)+['background FN'] if labels else 'auto',
                square=True, vmin=0.0
            )
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close()

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


if __name__ == '__main__':
    with open('dataN.json', 'r') as file:
        data = json.load(file)

    confusion_matrix = ConfusionMatrix(80)
    stats, ap = [], []

    outputs = data['eval_outputs']
    targets = data['targets']
    img_size = data['img_size']

    for batch in range(625):
        eval_outputs = outputs[batch]
        eval_targets = torch.tensor(targets[batch])
        img_size_this_batch = img_size[batch]

        for si, pred in enumerate(eval_outputs):
            pred_tensor = torch.tensor(pred).float()
            labels = eval_targets[eval_targets[:, 0] == si, 1:]
            nl = len(labels)
            target_cls = labels[:, 0].tolist() if nl else []

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, 10, dtype=torch.bool),
                                  torch.Tensor(), torch.Tensor(), target_cls))
                continue

            predn = pred_tensor.clone()
            correct = torch.zeros(pred_tensor.shape[0], 10, dtype=torch.bool)

            if nl:
                tbox = nms.xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= img_size_this_batch[si][0]
                tbox[:, [1, 3]] *= img_size_this_batch[si][1]

                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = PRMetric.process_one_image_prediction(predn, labelsn)

            stats.append((correct.cpu(), pred_tensor[:, 4].cpu(), pred_tensor[:, 5].cpu(),
                          target_cls))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    p, r, ap, f1, ap_class = PRMetric.ap_per_class(*stats)
    AP50_F1_max_idx = f1.mean(0).argmax()
    print(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx/1000.0}.")

    ap50, ap = ap[:, 0], ap.mean(1)
    mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
    num_targets = np.bincount(stats[3].astype(np.int64), minlength=80)
    s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format

    print(ap_class)



