#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import copy
import os

from tqdm import tqdm
import numpy as np
import json
import torch
import yaml
from copy import deepcopy
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolov7.data.data_load import create_dataloader
from yolov7.utils.checkpoint import load_checkpoint
from yolov7.layers.common import RepVGGBlock
from yolov7.utils.torch_utils import *
from yolov7.utils.nms import *
from yolov7.utils.metric import PRMetric


class Evaler:
    def __init__(self, data, batch_size, img_size, conf_threshold, iou_threshold,
                 device, half, save_dir, shrink_size, infer_on_rect, verbose,
                 do_coco_metric, do_pr_metric, plot_curve, plot_confusion_matrix,
                 specific_shape, height, width):
        assert do_pr_metric or do_coco_metric, 'ERROR: at least set one val metric'
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.half = half
        self.save_dir = save_dir
        self.shrink_size = shrink_size
        self.infer_on_rect = infer_on_rect
        self.verbose = verbose
        self.do_coco_metric = do_coco_metric
        self.do_pr_metric = do_pr_metric
        self.plot_curve = plot_curve
        self.plot_confusion_matrix = plot_confusion_matrix
        self.specific_shape = specific_shape
        self.height = height
        self.width = width

        self.half = False
        self.is_coco = None
        self.ids = None
        self.stride = None

        self.speed_result = None

        self.pr_results = None

    def init_model(self, model, weights, task='val'):
        if task != 'train':
            if not os.path.exists(weights):
                pass
            model = load_checkpoint(weights=weights, map_location=self.device)
            self.stride = int(model.stride.max())
            # 切换为验证推理模式
            for layer in model.modules():
                layer.switch_to_deploy()

            print("Switch model to deploy modality.")
            print("Model Summary: {}".format(get_model_info(model, self.img_size)))
        model.half() if self.half else model.float()
        return model

    def init_dataloader(self, dataloader=None, task='val'):
        self.is_coco = self.data.get('is_coco', False)  # self.is_coco = True
        self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        if task != 'train':
            eval_hyp = {
                'shrink_size': self.shrink_size,
            }
            rect = self.infer_on_rect
            pad = 0.5 if rect else 0.0
            dataloader = create_dataloader(
                path=self.data[task if task in ('train', 'val', 'test') else 'val'],
                img_size=self.img_size,
                batch_size=self.batch_size,
                hyp=eval_hyp,
                check_labels=False,
                check_images=False,
                stride=self.stride,
                data_dict=self.data,
                shuffle=False,
                task=task,
                rect=rect,
                pad=pad
            )[0] if dataloader is None else dataloader
        return dataloader

    def predict_model(self, model, dataloader, task):
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        pbar = tqdm(dataloader, desc=f"Inferencing model in {task} datasets.", ncols=100)

        stats, seen = [], [0] if self.do_pr_metric else (None, None)

        for i, (images, targets, paths, shapes) in enumerate(pbar):
            # 预处理过程
            t1 = time_sync()
            images = images.to(self.device, non_blocking=True)
            images = images.half() if self.half else images.float()
            images /= 255.0
            self.speed_result[1] += time_sync() - t1  # 预处理时间

            # 推理过程
            t2 = time_sync()
            outputs = model(images)
            self.speed_result[2] += time_sync() - t2  # 推理时间

            # 后续处理
            t3 = time_sync()
            outputs = non_max_suppression(outputs, self.conf_threshold, self.iou_threshold,
                                          multi_label=True)
            self.speed_result[3] += time_sync() - t3
            self.speed_result[0] += len(outputs)

            if self.do_pr_metric:
                eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])
                self.pr_metric_preprocess(eval_outputs, images, targets, stats, seen)

            # save result
            pred_results.extend(self.convert_to_coco_format(outputs=outputs,
                                                            images=images,
                                                            paths=paths,
                                                            shapes=shapes,
                                                            ids=self.ids,))

        if self.do_pr_metric:
            self.pr_metric_calculate(stats, seen)

        return pred_results

    def eval_model(self, pred_results, model, dataloader, task):
        """Evaluate models
        For task speed, this function only evaluates the speed of model and outputs inference time.
        For task val, this function evaluates the speed and mAP by pycocotools, and returns
        inference time and mAP value.
        """
        print(f'\nEvaluating speed...')
        self.eval_speed(task)

        if not self.do_coco_metric and self.do_pr_metric:
            return self.pr_results

        print(f'\nEvaluating mAP by pycocotools...')
        if task != 'speed' and len(pred_results):
            if 'anno_path' in self.data:
                # 如果原本就有anno文件，就直接读取原anno文件即可
                anno_json = self.data['anno_path']
            else:
                # 生成coco格式的标签
                task = 'val' if task == 'train' else task
                if not isinstance(self.data[task], list):
                    self.data[task] = [self.data[task]]
                dataset_root = os.path.dirname(os.path.dirname(self.data[task][0]))
                base_name = os.path.basename(self.data[task][0])
                anno_json = os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json')

            # 对于预测结果，将其保存为与anno标注文件格式相同的预测结果文件
            pred_json = os.path.join(self.save_dir, "predictions.json")
            print(f'Saving {pred_json}...')
            with open(pred_json, 'w') as f:
                json.dump(pred_results, f)

            # 使用pycocotools来对anno文件进行处理
            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            coco_eval = COCOeval(anno, pred, 'bbox')

            # ----------------------------------------------------------------------------
            # 如果是coco格式类型的数据，需要进行的一步处理是，将图片的文件名去除后缀名后，只取其名字来
            # 构成图片的id，即imgIds
            # ----------------------------------------------------------------------------
            if self.is_coco:
                imgIds = [int(os.path.basename(x).split(".")[0])
                          for x in dataloader.dataset.img_paths]
                coco_eval.params.imgIds = imgIds

            """
            COCO-tools 的计算预测结果的三步：
                (1) coco_eval.evaluate(): 这一步的含义是根据输入COCOEval类的ground-truth标签以及
                        预测结果的标签annotations.json文件来进行计算各项指标：mAP、F1、precision等
                (2) coco_eval.accumulate(): 这一步的含义是将上一步中的计算结果进行汇总累积
                (3) coco_eval.summarize(): 这一步的含义是将前两步的计算结果进行打印输出
            """
            coco_eval.evaluate()
            coco_eval.accumulate()

            # -----------------------------------------------------------------------------
            # 如果是需要输出详细的过程信息：
            #       1. val_dataset_img_count: 取出所有的ground-truth图片信息，并获取这些信息的
            #           长度，因为验证集中共有4952张图片，所以得到的ground-truth信息的长度也为4952
            #       2. label_count_dicts: 生成所有种类(共计80个种类)的统计字典，列表长度为种类数，
            #           并且列表的元素为字典，每个字典记录了某个目标类别的所有图片的图片id，以及这个目标
            #           类别存在多少条annotations
            # -----------------------------------------------------------------------------
            if self.verbose:
                val_dataset_img_count = coco_eval.cocoGt.imgToAnns.__len__()
                val_dataset_anns_count = 0
                label_count_dict = {"images": set(), "anns": 0}
                label_count_dicts = [deepcopy(label_count_dict) for _ in range(model.nc)]
                for _, ann_i in coco_eval.cocoGt.anns.items():
                    val_dataset_anns_count += 1
                    # 用category_id去选出对应的类别标号
                    nc_i = self.coco80_to_coco91_class().index(ann_i['category_id']) \
                        if self.is_coco else ann_i['category_id']

                    # ---------------------------------------------------------------------
                    # 统计一下属于这个类别的图片有哪些(集合图片的id)
                    # 有一张图片中包含这个类别就anns加一
                    #   1. 在遍历annotations的过程中，对于当前的annotation所指的类别，找到类别对应的
                    #       信息字典，在该信息字典中的 'images' 键值对中添加当前annotation所属的图片
                    #       的image_id
                    #   2. 在完成添加图片 image_id 后，记录annotation数量的键值对数量 +1，最终的
                    #       anns 表示的是这个类别的目标实体总共有多少条标注
                    #
                    #   最终得到的结果是：
                    #       len(label_count_dicts)=80, 总长度为80类，每个dict装载着对应类别的包
                    #       含这个类别的图片id，以及这个类别总共有几条标注anns
                    # ---------------------------------------------------------------------
                    label_count_dicts[nc_i]['images'].add(ann_i['image_id'])
                    label_count_dicts[nc_i]['anns'] += 1

                s = (('%-16s' + '%12s' * 7) %
                     ('Class', 'Labeled_images', 'Labels', 'P@.5iou', 'R@.5iou',
                      'F1@.5iou', 'mAP@.5', 'mAP@.5:.95'))
                print(s)

                # -------------------------------------------------------------------------
                # cocoEval.eval['precision'] 是一个 5 维的数组
                # precision 的基础组成 [T, R, K, A, M]， 以下分别解析数组的内容和含义
                #   1. 第一维度 T ：表示IoU的 10 个阈值，从0.5 到 0.95, 间隔步长为 0.05，即分别表
                #       示[mAP@0.5, mAP@0.55, ..., mAP@0.90, mAP@0.95]
                #   2. 第二维度 R ：表示 101 个recall召回率的阈值，从 0 到 101
                #   3. 第三维度 K ：表示类别，如果是想展示第一类就取值为0，第二类就取值为1，以此类推...
                #   4. 第四维度 A ：表示 area 目标的大小范围 (all, small, medium, large)
                #   5. 第五维度 M ：表示 maxDet，即单张图片中最多检测框的数量三种 [1, 10, 100]
                #
                # 因此，coco_precision[:, :, :, 0, 2] 的含义就表示的是：所有类别的物体在最大检测框
                # 数量为 100，每个类别物体从mAP@0.5~mAP@0.95的10个步长中，每个步长101个recall所对应
                # 的precision值
                #
                # 最后，对所有的图片的AP求和取平均即可得到mAP
                # -------------------------------------------------------------------------
                coco_precision = coco_eval.eval['precision']
                coco_precision_all = coco_precision[:, :, :, 0, 2]
                map_ = np.mean(coco_precision_all[coco_precision_all > -1])

                # -------------------------------------------------------------------------
                # coco_precision_iou50 = coco_precision[0, :, :, 0, 2]，表示IoU阈值为0.5这一种
                # 情况下的所有的精确度值 AP
                #
                #   1. 求出 IoU阈值=0.5 情况下的精确度 AP，记为coco_precision_iou50，并且shape=
                #       [101, 80]
                #   2. map50, 表示在iou=0.5阈值下的mAP平均AP值大小，对于每个类别的AP，求取方法是求101
                #       个recall阈值下的precision的平均值；对于mAP，求取方法是求出80个目标类别的AP的
                #       平均值
                #   3. mean_precision, 表示不同recall尺度下的精度precision的平均值，求的是目标类别
                #       上的平均(后续继续用于mean_f1的计算)
                #   4. mean_recall, 表示不同的recall尺度，用于与上面的平均precision对应(方便后续
                #       计算mean_f1)
                #   5. 选出最大的mean_f1的索引位置，然后打印输出对应的mean_precision和mean_recall
                # -------------------------------------------------------------------------
                coco_precision_iou50 = coco_precision[0, :, :, 0, 2]
                map50 = np.mean(coco_precision_iou50[coco_precision_iou50 > -1])
                mean_precision = np.array([
                    np.mean(coco_precision_iou50[k][coco_precision_iou50[k] > -1])  # 对所有类别的这个维度求平均
                    for k in range(coco_precision_iou50.shape[0])
                ])
                mean_recall = np.linspace(0.0, 1.00,
                                          int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
                mean_f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-16)
                i = mean_f1.argmax()

                pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5
                print(pf % ('all', val_dataset_img_count, val_dataset_anns_count,
                            mean_precision[i], mean_recall[i], mean_f1[i], map50, map_))

                for nc_i in range(model.nc):
                    # ---------------------------------------------------------------------------------
                    # coco_precision_c = coco_precision[:, :, nc_i, 0, 2]
                    #   表示的含义是：所有类别的物体在最大检测框数量 100，类别为 nc_i 的物体从mAP@0.5~mAP@0.95的
                    #   10个步长中，每个步为101个recall阈值的所对应的precision值，共计 10 * 101 个precision值
                    #
                    # ap = np.mean(coco_precision_c[coco_precision_c > -1])，ap代表对这 10 * 101 个
                    #   precision 值求一个平均，即可得到这一个 nc_i 类别的平均精确度AP
                    # ---------------------------------------------------------------------------------
                    coco_precision_cls = coco_precision[:, :, nc_i, 0, 2]
                    ap = np.mean(coco_precision_cls[coco_precision_cls > -1])

                    # ---------------------------------------------------------------------------------
                    # coco_precision_cls_iou50 表示在iou=0.5的阈值之下，类别为 nc_i 的目标实体在最多 100 个预测框
                    #   的状况下，总共101个recall阈值对应的 precision 值
                    #
                    #   1. ap50 = np.mean(coco_precision_cls_iou50[coco_precision_cls_iou50 > -1])，ap50
                    #       表示对类别 nc_i 在 iou=0.5 阈值之下求平均，得到平均精度AP50
                    #   2. precision = [0, :, nc_i, 0, 2] 表示在 iou=0.5 的阈值之下，类别nc_i在不同的召回
                    #       率recall下对应的精确度precision
                    #   3. recall.shape=[101, ] 表示不同的recall召回率，用于和上述精确度precision对应，以方便
                    #       后续计算f1值
                    #   4. f1值使用precision和recall进行计算，然后选择出f1值最大的索引位置，然后再根据索引值选择出
                    #       对应的precision和recall进行打印输出
                    # ---------------------------------------------------------------------------------
                    coco_precision_cls_iou50 = coco_precision[0, :, nc_i, 0, 2]
                    ap50 = np.mean(coco_precision_cls_iou50[coco_precision_cls_iou50 > -1])
                    precision = coco_precision_cls_iou50
                    recall = np.linspace(0.0, 1.00,
                                         int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
                    f1 = 2 * precision * recall / (precision + recall + 1e-16)
                    i = f1.argmax()
                    print(pf % (model.names[nc_i], len(label_count_dicts[nc_i]['images']),
                                label_count_dicts[nc_i]['anns'], precision[i], recall[i], f1[i],
                                ap50, ap))
            coco_eval.summarize()
            map_, map50 = coco_eval.stats[:2]
            model.float()
            if task != 'train':
                print(f"Results saved to {self.save_dir}")
            return map50, map_
        return 0.0, 0.0

    def convert_to_coco_format(self, outputs, images, paths, shapes, ids):
        """
        Args:
             outputs (list): len=batch_size, 装的是预测的输出结果，每一个元素（一个元素表示一张图片）的shape=[300, 6], 本例
                            中是300个经过NMS算法筛选的预测框，6=4+1+1，分别代表锚框的xyxy四角坐标信息、预
                            测物体的置信度conf、以及最后的锚框所检测出来的
             images (Tensor): shape=[batch_size, 3, height, width], 表示的是输入的图片数据，这些图片
                            全部都被除以 255.0 进行了标准化
             paths (list): len=batch_size, 表示这一批次进行预测的图片的文件路径，每一个元素都是图片的文件路
                            径，在本例中，每个批次有 8 张图片进行验证，因此paths长度为8
            shapes (tuple): (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)
        """

        pred_results = []

        for i, pred in enumerate(outputs):
            if len(pred) == 0:
                continue
            # --------------------------------------------------------------------------
            # 这里的操作是取出图片的原始大小和图片的路径
            #       1. path: type='pathlib.WindowsPath', 表示的是图片的文件路径
            #       2. shape: type=tuple, 表示的是图片的原始大小，height * width
            # --------------------------------------------------------------------------
            path, shape = Path(paths[i]), shapes[i][0]

            # --------------------------------------------------------------------------
            # scale_coords 函数的解析：
            #   接受的参数：
            #       1. 放缩操作后的图片尺寸：images[i].shape[1:]
            #       2. 预测结果的锚框位置信息(xyxy坐标)：pred[:, :4]
            #       3. 未经放缩的原始图片的大小信息(height, width)：h0、w0
            #       4. 从原始图片到接受预测的图片经过的放缩倍数、填充数值的信息：ratio, pad
            #   得到的结果：
            #       把在放缩操作后的图片上的锚框预测结果反向还原为在原始未经放缩图片上的锚框大小信息
            #           pred[:, 4:] ===> pred[:, :4] 虽然未声明新变量，但原变量内容已改变
            # --------------------------------------------------------------------------
            self.scale_coords(images[i].shape[1:], pred[:, :4], shape, shapes[i][1])

            # --------------------------------------------------------------------------
            # 其余信息的处理：
            #   1. 获取图片的id值：用Path类处理直接得到文件名（不带文件后缀名）
            #   2. 将预测锚框的对角坐标形式xyxy转换为中心点宽高形式xywh
            #   3. 将中心点宽高模式的中心点减去半宽高可以得到左上角的坐标
            #   4. 获取锚框所预测的物体种类：class=pred[:, 5]
            #   5. 获取每个锚框的预测分数：scores=pred[:, 4]
            # --------------------------------------------------------------------------
            image_id = int(path.stem) if self.is_coco else path.stem
            bboxes = self.box_convert(pred[:, 0:4])
            bboxes[:, :2] -= bboxes[:, 2:] / 2
            cls = pred[:, 5]
            scores = pred[:, 4]

            # ---------------------------------------------------------------------------
            # 将预测值pred.shape = [300, 6] 逐一取出：
            # pred预测值所包含的内容是：
            #       1. bbox.shape = [300, 4] ==> 逐一取出就是长度为4的list，
            #           装有:[左上角横坐标, 左上角纵坐标, 中心点横坐标, 中心点纵坐标]
            #       2. score.shape = [300, 1] ==>  逐一取出，并从Tensor转换为python的int型，因
            #           为python的int型才能写入json文件之中
            #       3. cls.shape = [300, 1] ==> 逐一取出，并从Tensor转换为python的int型
            # ---------------------------------------------------------------------------
            for ind in range(pred.shape[0]):
                category_id = ids[int(cls[ind])]
                bbox = [round(x, 3) for x in bboxes[ind].tolist()]
                score = round(scores[ind].item(), 5)
                pred_data = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score
                }
                pred_results.append(pred_data)
        return pred_results

    def eval_speed(self, task):
        """Evaluate model inference speed."""
        if task != 'train':
            n_samples = self.speed_result[0].item()
            pre_time, inf_time, nms_time = 1000 * self.speed_result[1:].cpu().numpy() / n_samples
            for n, v in zip(["pre-process", "inference", "NMS"], [pre_time, inf_time, nms_time]):
                print("Average {} time: {:.2f} ms".format(n, v))

    def pr_metric_calculate(self, stats, seen):
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 转换为numpy形式
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = PRMetric.ap_per_class(*stats, plot=self.plot_curve,
                                                           save_dir=self.save_dir, )
            AP50_F1_max_idx = f1.mean(0).argmax()
            print(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx/1000.0}.")
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
            num_target = np.bincount(stats[3].astype(np.int64), minlength=80)

            s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou',
                                          'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
            print(s)
            pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
            print(pf % ('all', seen[0], num_target.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))
            self.pr_results = (map50, map)

            if self.verbose:
                for i, c in enumerate(ap_class):
                    print(pf % (model.names[c], seen, num_target[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                f1[i, AP50_F1_max_idx], ap50[i], ap[i]))
        else:
            print("Calculate metric failed, might check dataset.")
            self.pr_results = (0.0, 0.0)

    @staticmethod
    def pr_metric_preprocess(eval_outputs, images, targets, stats, seen):
        assert eval_outputs is not None, "Eval_outputs is None!"
        for si, pred in enumerate(eval_outputs):
            labels = targets[targets[:, 0] == si, 1:]
            num_labels = len(labels)
            target_cls = labels[:, 0].tolist() if num_labels else []
            seen[0] += 1

            if len(pred) == 0:
                if num_labels:
                    stats.append((torch.zeros(0, 10, dtype=torch.bool),
                                  torch.Tensor(), torch.Tensor(), target_cls))
                continue

            predn = pred.clone()
            correct = torch.zeros(pred.shape[0], 10, dtype=torch.bool)

            if num_labels:
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= images[si].shape[1:][1]
                tbox[:, [1, 3]] *= images[si].shape[1:][0]

                labels_n = torch.cat((labels[:, 0:1], tbox), 1)
                correct = PRMetric.process_one_image_prediction(predn, labels_n)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_cls))

    @staticmethod
    def scale_coords(image1_shape, coords, image0_shape, ratio_pad=None):
        """
        Args:
             coords (Tensor): shape=[300, 4], 本例中的形状为 300 * 4，其中300表示共有300个生成框预测，
                        4则表示为预测的锚框的xyxy的四角坐标的结果
             image0_shape (tuple): shape=(height, width), 表示为输入的这个预测图片的原始图片的大小
             ratio_pad (tuple): 内容为=((h * ratio / h0, w * ratio / w0), pad)，表示原始图片的进入
                        模型进行预测的图片大小的比值，pad则表示从原始图片放缩到接受预测的图片所填充的大小
        """
        gain = ratio_pad[0]  # gain表示原始图片放缩到接受检测图片所乘上的倍数
        pad = ratio_pad[1]  # pad表示原始图片经过放缩后，变换到接受检测的图片所需要的填充大小

        # ------------------------------------------------------------------------------------
        # coords[:, [0, 2]] 表示所有预测锚框的坐标的x值（分别为坐标左上角和右下角的x值）
        # coords[:, [1, 3]] 表示所有预测锚框的坐标的y值（分别为坐标左上角和右下角的y值）
        # 预测结果的锚框复原操作：
        #       1. 首先减去填充操作所填充的pad值
        #       2. 然后再除以原始图片放大的倍数，最终回到原始图片的尺寸
        # -------------------------------------------------------------------------------------
        coords[:, [0, 2]] -= pad[0]
        coords[:, [0, 2]] /= gain[1]
        coords[:, [1, 3]] -= pad[1]
        coords[:, [1, 3]] /= gain[0]

        if isinstance(coords, torch.Tensor):
            # 把复原后的值固定在原始图片尺寸之间，防止图片范围越界
            coords[:, 0].clamp_(0, image0_shape[1])
            coords[:, 1].clamp_(0, image0_shape[0])
            coords[:, 2].clamp_(0, image0_shape[1])
            coords[:, 3].clamp_(0, image0_shape[0])
        else:
            coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, image0_shape[1])
            coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, image0_shape[0])
        return coords

    @staticmethod
    def box_convert(x):
        if isinstance(x, torch.Tensor):
            y = x.clone()
        else:
            y = np.copy(x)
        # -------------------------------------------------------------------------
        # 本函数的作用在于将预测锚框的对角坐标xyxy形式转换为whxy的bbox形式
        #       1. 首先计算出中心点的横坐标：(左上角的横坐标x+右下角的横坐标x)/2=中心点x坐标
        #       2. 然后再计算出中心点的纵坐标：(左上角的纵坐标y+右下角的纵坐标y)/2=中心点y坐标
        #       3. 第三步计算锚框的半宽：右下角的横坐标x - 左上角横坐标x = 锚框的宽
        #       4. 第四步计算锚框的半高：右下角的纵坐标y - 左上角纵坐标y = 锚框的高
        # -------------------------------------------------------------------------
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return y

    @staticmethod
    def coco80_to_coco91_class():
        return [int(i) for i in np.arange(1, 91)]

    @staticmethod
    def check_task(task):
        if task not in ['train', 'val', 'test', 'speed']:
            raise Exception("task argument error: only support "
                            "'train' / 'val' / 'test' / 'speed' task.")

    @staticmethod
    def check_threshold(conf_thres, iou_thres, task):
        """Check whether confidence and iou threshold are best for task val/speed"""
        if task != 'train':
            if task == 'val' or task == 'test':
                if conf_thres > 0.03:
                    print(f'Warning! The best conf_thresh when evaluate the '
                          f'model is less than 0.03, while you set it to: {conf_thres}')
                if iou_thres != 0.65:
                    print(f'Warning! The best iou_thresh when evaluate the '
                          f'model is 0.65, while you set it to: {iou_thres}')
            if task == 'speed' and conf_thres < 0.4:
                print(f'Warning! The best conf_thresh when test the speed of the '
                      f'model is larger than 0.4, while you set it to: {conf_thres}')

    @staticmethod
    def reload_dataset(data, task='val'):
        with open(data, errors='ignore') as yaml_file:
            data = yaml.safe_load(yaml_file)
        task = 'test' if task == 'test' else 'val'
        path = data.get(task, 'val')
        if not isinstance(path, list):
            path = [path]
        for p in path:
            if not os.path.exists(p):
                raise Exception(f'Dataset path {p} not found.')
        return data


if __name__ == '__main__':
    from yolov7.model.model_yolo import ModelYOLO
    from yolov7.utils.events import load_yaml
    from yolov7.data.data_load import create_dataloader
    model = ModelYOLO().eval().to(device='cuda')
    data_dict = load_yaml('../../data/coco.yaml')
    evaler = Evaler(data=data_dict, batch_size=4, img_size=640, conf_threshold=0.03,
                    iou_threshold=0.65, device='cuda', half=False, save_dir='',
                    shrink_size=640, infer_on_rect=False, verbose=False, do_coco_metric=True,
                    do_pr_metric=False, plot_curve=True, plot_confusion_matrix=False,
                    specific_shape=False, height=640, width=640)
    model = evaler.init_model(model=model, weights=None, task='train')
    dataloader = create_dataloader(path='../../coco/images/val2017',
                                   img_size=640, batch_size=8, stride=32, hyp={},
                                   data_dict=load_yaml('../../data/coco.yaml'))[0]
    dataloader = evaler.init_dataloader(dataloader, 'train')
    evaler.predict_model(model=model, dataloader=dataloader, task='train')


