# Datasets
import glob
import json
import logging
import math
import os
import os.path as osp
import hashlib
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from yolov7.utils.events import load_yaml
from yolov7.data.data_augment import augment_hsv, mosaic_augmentation, mix_up, letterbox


# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        orientation = k
        break


class LoadImagesAndLabels(Dataset):
    """用于创建训练、测试用的数据集"""
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, task='val',
                 rect=False, image_weights=False, cache_images=False, single_cls=False, data_dict=None,
                 stride=32, pad=0.0, prefix='', rank=-1, check_images=False, check_labels=False):
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        t1 = time.time()
        # 图片的初始化信息
        self.path = path
        self.img_size = img_size
        self.image_weights = image_weights

        # 图片增强等处理的参数
        self.stride = stride
        self.pad = pad
        self.batch_size = batch_size
        self.augment = augment
        self.hyp = hyp

        # 有关于本数据集的cache路径信息
        self.valid_img_record = None

        # 是否检查图片和标签的标记值(boolean)
        self.check_images = check_images
        self.check_labels = check_labels

        # 当前进程号的标记
        self.rank = rank
        self.rect = False if image_weights else rect
        self.task = task

        # 一种图片数据增广的方式，将四张不同的图片拼接成一整张
        self.mosaic = self.augment and not self.rect
        self.main_process = self.rank in (-1, 0)
        self.data_dict = data_dict
        self.class_name = data_dict["names"]
        self.img_paths, self.labels, self.img_info = self.get_images_and_labels(img_dir=path)

        # 对于图片rect的处理：
        if self.rect:
            assert self.img_info, "WRONG! img_info is NULL!"
            shapes = [self.img_info[p]["shape"] for p in self.img_paths]
            self.shapes = np.array(shapes, dtype=np.float64)
            self.batch_indices = np.floor(
                np.arange(len(shapes)) / self.batch_size
            ).astype(np.int_)
            self.batch_shapes = self.sort_files_shapes()

        t2 = time.time()
        if self.main_process:
            print("✅Dataset initialization completed in {:.2f}s.".format(t2 - t1))

    def __len__(self):
        """获取数据集的长度"""
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        根据给定index来获取对应的数据样本，
        该函数在训练集情况下使用马赛克图片增强技术以及混合增强技术，而
        在验证数据的情况下使用letterbox图片增强技术
        """
        # 马赛克图片数据增强
        if self.augment and random.random() < self.hyp["mosaic"]:
            img, labels = self.get_mosaic(index)
            shapes = None
            # Mix up图片数据增强
            if random.random() < self.hyp["mixup"]:
                # 再创造一个马赛克图片增强样本出来
                img_other, labels_other = self.get_mosaic(
                    random.randint(0, len(self.img_paths) - 1)
                )
                # 和前面的马赛克图片增强样本合并，得到MixUp图片增强样本
                img, labels = mix_up(img, labels, img_other, labels_other)

        else:
            # -------------------------------------------------------------------------
            # 加载图片..., self.load_image只是单纯地把图片加载成np数组，不包含任何图片数据增强操作
            # self.load_image 会把原生图片等比例resize成最长边等于预设值（640）的大小
            # h0w0 是图片的原始尺寸，而hw就是图片经过resize之后的尺寸(一般为预设值:最长边为640)
            # -------------------------------------------------------------------------
            if self.hyp and "test_load_size" in self.hyp:
                img, (h0, w0), (h, w) = self.load_image(index, self.hyp["test_load_size"])
            else:
                img, (h0, w0), (h, w) = self.load_image(index)

            # letterbox变换(其实就是将图片填充成正方形，边长为给定的img_size)
            shape = (
                self.batch_shapes[self.batch_indices[index]]
                if self.rect else self.img_size
            )  # 如果不采用rect模式，就使用预设的img_size来作为验证集图片填充成的正方形的边长

            # ---------------------------------------------------------
            # 检查参数字典，看是否要求返回整型的填充数值：
            # pad是指填充成给定边长的正方形的上下左右的填充值
            # 注意，这里输入的img的尺寸已经经过了resize
            # ---------------------------------------------------------
            if self.hyp and "letterbox_return_int" in self.hyp:
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment,
                                            return_int=self.hyp["letterbox_return_int"])
            else:
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

            # ---------------------------------------------------------
            # shapes的输出内容的解析：
            #       1. (h0, w0): 表示的是原始图片的高和宽
            #       2.
            shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:
                # 将resize后的尺寸hw * 预设边长和原始边长的比值ratio 后，赋值给hw
                w *= ratio  # ratio是指letterbox指定尺寸和load_image加载进来的图片尺寸的比值
                h *= ratio  # load_image加载出来的图片高宽乘上ratio可以调整为letterbox下的尺寸
                # 新的锚框
                boxes = np.copy(labels[:, 1:])
                # 坐标的中心点的x坐标减去 width / 2，再乘上整个图片的宽度，就可以得到左上角的横坐标（别忘了添加填充值）
                # 注意，coco数据集的坐标分别是 中心点x，中心点y，框宽，框高(全部都是相对于整张图片，因此取值0~1)

                """
                之所以要加上pad的值：
                    假设有如下的情况，图片的大小为 640 * 480, 经过letterbox填充成 640 * 640
                    那么就会在图片的width两边分别加上大小为 pad=80 的填充
                    假设原图的锚框左上角的坐标点为 x=0.2, y=0.3 (表示坐标点处于整张图片高和宽的比例位置)
                    因为图片的两边经过填充，因此计算实际像素位置时为：80 + 0.3 * 480
                """
                # 坐标的中心点的x坐标减去 width / 2，再乘上整个图片的宽度，就可以得到左上角的横坐标
                boxes[:, 0] = (
                        w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
                )  # 左上角的横坐标 x 取值

                # 坐标的中心点的y坐标减去 height / 2，再乘上整个图片的高度，就可以得到左上角的纵坐标
                boxes[:, 1] = (
                        h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
                )  # 左上角的纵坐标 y 取值

                # 坐标的中心点x坐标加上 width / 2，再乘上整个图片的宽度，就可以得到右下角的横坐标
                boxes[:, 2] = (
                        w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
                )  # 右下角的横坐标 x 取值

                # 坐标的中心点的y坐标加上 height / 2，再乘上整个图片的高度，就可以得到右下角的纵坐标
                boxes[:, 3] = (
                        h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
                )  # 右下角的纵坐标 y 取值
                labels[:, 1:] = boxes

        """
        这一步操作的目的是将labels的对角坐标形式重新转换成中心宽高模式
        即：[x1, y1, x2, y2] --> [x_center, y_center, width, height]
        并且中心宽高模式都是相对值，即中心坐标点相对于左边界上边界位于整张图片的什么位置（取值0~1）
        """
        if len(labels):
            h, w = img.shape[:2]
            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:] = boxes

        if self.augment:
            img, labels = self.general_augment(img, labels)

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return (torch.from_numpy(img).to(dtype=torch.float32),
                labels_out, self.img_paths[index], shapes)

    def get_images_and_labels(self, img_dir):
        """首先先获取图片的路径、标签的路径、img_info、cache_info"""
        img_paths, label_paths, img_info, cache_info = self._get_img_and_label_list(img_dir)

        num_threads = min(12, os.cpu_count())  # 多线程处理图片所设定的线程数

        # 查看是否需要检验图片路径文件
        if self.check_images and self.main_process:
            img_info = {}
            nc, msgs = 0, []
            with Pool(num_threads) as pool:
                pbar = tqdm(pool.imap(LoadImagesAndLabels.check_image, img_paths),
                            total=len(img_paths))
                for img_path, shape_per_img, nc_per_img, msg in pbar:
                    # 如果返回结果显示为：图片无损坏，则向img-info中添加图片的高宽信息
                    if nc_per_img == 0:
                        img_info[img_path] = {"shape": shape_per_img}
                    nc += nc_per_img
                    if msg:
                        msgs.append(msg)
            # nc 记录有无图片损坏 num_corruption
            pbar.desc = f"{nc} image(s) corrupted"
            # 关闭tqdm进度条
            pbar.close()
            cache_info = {"information": img_info, "image_hash": self.get_hash(img_paths)}
            # 保存cache信息，方便下次快速检查
            with open(self.valid_img_record, "w") as f:
                json.dump(cache_info, f)

        # 查看是否需要检验标签路径文件
        if self.check_labels:
            cache_info["label_hash"] = self.get_hash(label_paths)  # 将计算得到的哈希值存入标签key
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
            print(f"Checking formats of labels with {num_threads} processes:")
            with Pool(num_threads) as pool:
                pbar = tqdm(pool.imap(LoadImagesAndLabels.check_label_files,
                                      zip(img_paths, label_paths)),
                            total=len(label_paths))
                for (img_path, labels_per_file, nc_per_file, nm_per_file,
                     nf_per_file, ne_per_file, msg) in pbar:
                    if nc_per_file == 0:
                        # 如果确认该文件并未损坏，将label文件的标签值添加到info字典中
                        img_info[img_path]["labels"] = labels_per_file
                    else:
                        # 如果label标签文件损坏，则将对应的图片img文件从字典中删去
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    pbar.desc = (f"{nf} label(s) found, {nm} label(s) missing, "
                                 f"{ne} label(s) empty, {nc} invalid label files")
            # 最后关闭tqdm时间模块并保存img与labels的检查cache信息
            pbar.close()
            with open(self.valid_img_record, "w") as f:
                json.dump(cache_info, f)

        # 如果任务是构建验证数据集：
        if self.task.lower() == 'val':
            self._build_annotations(img_dir=img_dir, img_info=img_info)

        img_paths, labels = list(
            zip(
                *[
                    (img_path,
                     np.array(info["labels"], dtype=np.float32)
                     if info["labels"] else np.zeros((0, 5), dtype=np.float32))
                    for img_path, info in img_info.items()  # 取出img_info所有的键值对
                ]
            )
        )
        print(f"{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. ")
        return img_paths, labels, img_info

    def _get_img_and_label_list(self, img_dir):
        assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"

        # 递归地遍历出所有图片文件的路径，并将其存储在数组中
        print("🔍️Searching all the images...")
        start_time = time.time()
        img_paths = glob.glob(osp.join(img_dir, "**/*"), recursive=True)
        img_paths = [p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p)]
        end_time = time.time()
        print("✅Searching completed in {:.2f}s".format(end_time - start_time))

        assert img_paths, f"No images found in {img_dir}!"

        # 以下开始搜索所有的标签文件路径，并将其存储在数组中
        label_dir = osp.join(
            osp.dirname(osp.dirname(img_dir)), "labels", osp.basename(img_dir)
        )  # coco\labels\train2017
        assert osp.exists(label_dir), f"{label_dir} is an invalid directory path!"

        label_paths = [osp.join(str(label_dir), osp.splitext(osp.basename(p))[0]+'.txt')
                       for p in img_paths]
        assert label_paths, f"No labels found in {label_dir}."

        # 构建缓存文件的路径
        """
        这里的 img_info 存储的信息是每一张图片的相对路径，以及对应的尺寸和标签值
        其结构为：键-值对模式，key==img_path，value=={shape, labels}
        其中，shape和labels亦为键值对格式，shape == [height, width]，
        labels == [[class, 左上横坐标，左上纵坐标，右下横坐标，右下纵坐标], ...] 二维数组
        """
        print('Building cache path......')
        cache_info = {}
        img_info = {}
        valid_img_record = osp.join(
            osp.dirname(img_dir), "." + osp.basename(img_dir) + ".json"
        )
        self.valid_img_record = valid_img_record
        img_hash = self.get_hash(img_paths)
        label_hash = self.get_hash(label_paths)
        if osp.exists(valid_img_record):
            print('Cache file exists, now loading...')
            with open(valid_img_record, 'r') as f:
                cache_info = json.load(f)
                if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:
                    img_info = cache_info["information"]
                    if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:
                        self.check_labels = True
                else:
                    self.check_images, self.check_labels = True, True
        else:
            print(f'Cache file {valid_img_record} does not exist, '
                  f'need to check images and labels...')
            self.check_images, self.check_labels = True, True

        return img_paths, label_paths, img_info, cache_info

    def _build_annotations(self, img_dir, img_info):
        """检查data_dict中是否有is_coco, 没有的话就创建并存储annotations"""
        if self.data_dict.get("is_coco", False):
            assert osp.exists(self.data_dict["anno_path"]), \
                ("Eval on coco dataset must provide valid path "
                 "of the annotation file in config file: data/coco.yaml")
            print(f"anno path {self.data_dict['anno_path']} already exists!")
        else:
            assert self.class_name, \
                "Class names is required when converting labels to coco format for evaluating."
            save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
            if not osp.exists(save_dir):
                os.mkdir(save_dir)
            save_path = osp.join(
                # 得到的annotations保存路径为: /coco/annotations/instances_val2017.json
                save_dir, "instance_" + osp.basename(img_dir) + ".json")
            print(f"anno path dose not exists! now creating annotations in {save_path}")
            t1 = time.time()
            LoadImagesAndLabels.generate_coco_format_labels(
                img_info=img_info, class_names=self.class_name, save_path=save_path
            )
            t2 = time.time()
            print("{:.2f}s for generating COCO format labels.".format(t2 - t1))

    def sort_files_shapes(self):
        """
        该函数的作用和好处：
        1-更均匀的批次形状分布：
            通过根据图片的高宽比来确定每个批次训练图片的形状，可以尽量保证每个批次中的图片
            形状相似。这样可以避免在训练过程中出现形状差异过大的情况，有助于减少批次之间的
            差异性，提高训练的稳定性。
        2-最大程度地利用计算资源
            将形状相似的图片放在同一个批次中进行训练，可以最大程度地利用计算资源，减少在
            处理不同图片时的计算资源的浪费。这样可以提高训练效率，加速模型收敛速度。
        3-减少内存占用和数据加载时间
            经过形状调整后的图片可以更好地适应模型的输入要求，从而减少内存占用和数据加载
            时间。这对于处理大规模数据集时尤为重要，可以减少数据预处理时间，提高训练速度。
        """
        # 计算这批数据集共有多少个批次(batch)
        batch_num = self.batch_indices[-1] + 1
        s = self.shapes  # 原始图片的 height 和 width

        aspect_ratio = s[:, 1] / s[:, 0]  # 宽高比=width/height

        # 将高宽比按照从小到大的顺序排列，并得到排列后的原始索引值index
        index_rect = aspect_ratio.argsort()

        # 按照高宽比从小到大的索引重新加载图片数据的顺序
        self.img_paths = [self.img_paths[i] for i in index_rect]

        # 按照高宽比从小到大的索引重新加载图片标签数据的顺序
        self.labels = [self.labels[i] for i in index_rect]

        # 形状信息也重新排列
        self.shapes = s[index_rect]
        # 获得从小到大排列的高宽比aspect_ratio
        aspect_ratio = aspect_ratio[index_rect]

        # 设置训练图片的形状
        shapes = [[1, 1]] * batch_num
        for i in range(batch_num):
            # -----------------------------------------------------------------------------
            # 然后再按照批次号去取出一整个批次的所有图片的高宽比：
            #       aspect_ratio_index.shape = [batch_size]，表示的是每个批次中，每一张图片的高宽
            #       比，比如一个批次的图片数量 batch_size=8，因此一次选出的 aspect_ratio_index 的长
            #       度就是 8，每一个值都表示一张图片的高宽比
            # -----------------------------------------------------------------------------
            aspect_ratio_index = aspect_ratio[self.batch_indices == i]

            # -----------------------------------------------------------------------------
            # 选出这个批次中的最大高宽比、最小高宽比：
            #       1. aspect_ratio_index.min() 表示取出一个批次中所有图片的最小min的高宽比值
            #       2. aspect_ratio_index.max() 表示取出一个批次中所有图片的最大max的高宽比值
            # -----------------------------------------------------------------------------
            minimum, maximum = aspect_ratio_index.min(), aspect_ratio_index.max()

            if maximum < 1:
                """
                如果最大宽/高比小于1，就设置这个批次的所有图片的相对比例为 [1, maximum]
                注意：
                    shape[0]=height, shape[1]=width, 图片shape的第一维度表示height高， 第二维度表示width宽
                    因此，当这一批次的图片中，最大的宽/高比也小于 1，则说明这一批次的图片全部都是高大于宽，因此将这一批
                    次的统一形状比例设置为 [1, maximum]
                """
                shapes[i] = [1, maximum]  # maximum = height / width
            elif minimum > 1:
                """
                如果最小高宽比大于1，就设置这个批次的所有图片的相对比例为 [1/minimum, 1]
                注意：
                    shape[0]=height, shape[1]=width, 图片shape的第一维度表示height高，第二维度表示width宽，
                    因此，当这一批次的图片中，最小的宽/高比要大于 1，则说明这一批次的所有图片都是宽大于高的，因此这一
                    批次的统一形状比例设置为 [1 / minimum, 1]
                """
                shapes[i] = [1 / minimum, 1]  # minimum = width / height

        batch_shapes = (
            np.ceil(
                # 把制作好的批次图片形状比例
                np.array(shapes) * self.img_size / self.stride + self.pad
            ).astype(np.int_) * self.stride
        )
        return batch_shapes

    def load_image(self, index, force_load_size=None):
        """
        加载图片
        该函数通过cv2来进行图片加载，对原始图片进行大小调整（保持原有的高宽比）
        Returns:
            Image图片，图片的原始尺寸，更改后的图片尺寸
        """
        path = self.img_paths[index]
        try:
            im = cv2.imread(path)
            assert im is not None, f"opencv cannot read image correctly or {path} not exists"
        except Warning as warning:
            im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

        """
        cv2读取图片的格式是 Height-Width-Channels
        最后一维表示图片通道数，因此只需要取前两位高宽即可
        """
        h0, w0 = im.shape[:2]  # 原始图片尺寸
        # ------------------------------------------------------------------------------------
        # r表示指定的img_size与实际图片之间的比例关系
        # r < 1 表示原始图片较长边大于指定大小img_size，此时需要缩小操作(图片等比例缩小到较长边等于img_size)
        # r > 1 表示原始图片较长边小于指定大小img_size，此时需要增大操作(图片等比例扩大到较长边等于ing_size)
        # ------------------------------------------------------------------------------------
        if force_load_size:
            ratio = force_load_size / max(h0, w0)
        else:
            ratio = self.img_size / max(h0, w0)
        if ratio != 1:
            im = cv2.resize(
                im,
                dsize=(int(w0 * ratio), int(h0 * ratio)),
                interpolation=cv2.INTER_AREA  # 图片缩小用区间插值
                if ratio < 1 and not self.augment else cv2.INTER_LINEAR)  # 图片增大用线性插值
        return im, (h0, w0), im.shape[:2]

    def general_augment(self, img, labels):
        """
        获取经过图片增强操作后的images和labels
        该函数的主要增强模式是：hsv颜色改变，随机垂直翻转，随机水平翻转
        """
        num_labels = len(labels)

        # HSV color-space
        augment_hsv(img,
                    h_gain=self.hyp["hsv_h"],
                    s_gain=self.hyp["hsv_s"],
                    v_gain=self.hyp["hsv_v"])

        # Flip up-down 垂直翻转
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)
            if num_labels:
                # 图像翻转了，labels标记的目标的锚框坐标位置也别忘记翻转！
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right 水平翻转
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)
            if num_labels:
                labels[:, 1] = 1 - labels[:, 1]

        return img, labels

    def get_mosaic(self, index):
        """获取经过马赛克图片增强的图片以及标签值"""
        indices = [index] + random.choices(
            range(0, len(self.img_paths)), k=3
        )  # 因为马赛克增强需要四张图片拼在一起，因此这里除了已经提供的index外，还需额外再随机选三张
        random.shuffle(indices)
        images, hs, ws, labels = [], [], [], []
        for index in indices:
            # 在图片加载的过程就已经完成了将原始图片resize到较长边小于等于给定尺寸img_size
            im, _, (h, w) = self.load_image(index)
            labels_per_img = self.labels[index]
            images.append(im)
            hs.append(h)
            ws.append(w)
            labels.append(labels_per_img)
        images, labels = mosaic_augmentation(self.img_size, images, hs, ws, labels, self.hyp)
        return images, labels

    def _rect_shape_process(self):
        shapes = [self.img_info[p]["shape"] for p in self.img_paths]

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        # --------------------------------------------------------------
        # collate_fn 的作用是在创建DataLoader的过程中，提供一种对于每个mini-batch
        # 进行更加细致的预处理的方法，在本例中提供的处理方法是，对于每个batch的每张图片
        # 对应labels，都在第一维度插入对应图片的index索引值
        # ---------------------------------------------------------------
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # 为创建标签添加目标图片的索引值
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def get_hash(paths):
        """获取路径的哈希值"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()

    @staticmethod
    def check_image(img_file):
        """验证图片"""
        nc, msg = 0, ""
        try:
            img = Image.open(img_file)
            img.verify()
            img = Image.open(img_file)
            shape = (img.height, img.width)  # !!!PIL的读取是(width, height)
            try:
                img_exif = img.getexif()
                if img_exif and orientation in img_exif:
                    rotation = img_exif[orientation]
                    if rotation in (6, 8):
                        shape = (shape[1], shape[0])
            except ValueError:
                img_exif = None

            assert (shape[0] > 9) and (shape[1] > 9), f"image size {shape} < 10 pixels"
            assert img.format.lower() in IMG_FORMATS, f"invalid image format {img.format}!"
            if img.format.lower() in ("jpg", "jpeg"):
                with open(img_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":
                        ImageOps.exif_transpose(Image.open(img_file)).save(
                            img_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {img_file}: corrupt JPEG restored and saved!"
            return img_file, shape, nc, msg

        except Exception as e:
            nc = 1
            msg = f"WARNING: {img_file}: ignoring corrupt image:{e}"
            return img_file, None, nc, msg

    @staticmethod
    def check_label_files(path_args):
        # 分别为单个图片jpg文件的路径，以及单个对应标签txt文件的路径
        img_path, label_path = path_args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing found empty message)
        try:
            if osp.exists(label_path):
                nf = 1  # label found 标记为找到label文件
                with open(label_path, "r") as f:
                    # 读取打开label标签文件，标签文件格式为 [类别，左上角x，左上角y，右下角x，右下角y]
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    # 读取txt文件内容，并将标签label转换成numpy数组格式
                    labels = np.array(labels, dtype=np.float32)

                # 经过处理后的一个标签文件的labels数组中应该是有长度的(>=1)
                if len(labels):
                    """一个标签txt文件产生的labels数组中的每个元素也是数组，并且长度一定要是5"""
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{label_path}: wrong label format."  # 如果labels数组中的元素存在长度不是5的，格式错误

                    """第二步：检查所有的标签值都要大于等于0"""
                    assert (
                            labels >= 0
                    ).all(), f"{label_path}: Label values error: all values in label file must > 0"

                    """第三步：检查所有的四角坐标都要被标准化，取值位于0~1之间"""
                    assert (
                            labels[:, 1:] <= 1
                    ).all(), f"{label_path}: Label values error: all coordinates must be normalized"

                    """第四步：检查有没有标签值重复的情况"""
                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):
                        labels = labels[indices]
                        msg += f"WARNING: {label_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # 标签列表为空 label empty
                    labels = []
            else:
                nm = 1  # 标签列表未找到 label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {label_path}: ignoring invalid labels: {e}"
            return img_path, None, nc, nm, nf, ne, msg

    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        """使用pycocotools构建验证集"""
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategories": ""}
            )
        ann_id = 0
        print("Converting to COCO format dataset...")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            img_w, img_h = info["shape"]
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # 把中心坐标-宽高模式更改为对焦坐标模式([x,y,w,h]-->[x1,y1,x2,y2])
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id 类别标签从 0 开始
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            "segmentation": []
                        }
                    )
                    ann_id += 1
        # 整个coco格式验证数据构建完成后，写入json文件
        with open(save_path, "w") as f:
            json.dump(dataset, f)


class LoadData:
    def __init__(self, path, webcam, webcam_addr):
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        if webcam:
            img_path = []
            video_path = [int(webcam_addr) if webcam_addr.isdigit() else webcam_addr]
        else:
            path = str(Path(path).resolve())
            if os.path.isdir(path):
                files = sorted(glob.glob(os.path.join(path, '**/*.*'), recursive=True))
            elif os.path.isfile(path):
                files = [path]
            else:
                raise FileNotFoundError(f'Invalid path {path}')

            img_path = [i for i in files if i.split('.')[-1] in IMG_FORMATS]
            video_path = [v for v in files if v.split('.')[-1] in VID_FORMATS]

        self.files = img_path + video_path
        self.num_files = len(self.files)
        self.type = 'image'

        self.frames, self.frame = None, None

        if len(video_path) > 0:
            self.add_video(video_path[0])
        else:
            self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.num_files:
            raise StopIteration
        path = self.files[self.count]
        if self.check_ext(path) == 'video':
            self.type = 'video'
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.num_files:
                    raise StopIteration
                path = self.files[self.count]
                self.add_video(path)
                ret_val, img = self.cap.read()
        else:
            self.count += 1
            img = cv2.imread(path)  # cv2的读取顺序是 BGR
        return img, path, self.cap

    def add_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def check_ext(self, path):
        """
        检测文件类型，到底是视频格式、还是图片格式的推理数据
        """
        if self.webcam:
            file_type = 'video'
        else:
            file_type = 'image' if path.split('.')[-1].lower() in IMG_FORMATS else 'video'
        return file_type

