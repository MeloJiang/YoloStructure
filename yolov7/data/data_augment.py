#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import math
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
"""
本模块文件中，所有图片增强函数接受的图片输入都是im形式，
也就是图片经过 cv2.imread 读取后的值 --> im = cv2.imread(img_path)
"""


def augment_hsv(im, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    """HSV 颜色空间图片增强"""
    if h_gain or s_gain or v_gain:
        # 获得随机的hsv色域空间的三个值
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # 图片的数据类型是uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # 更改颜色空间不需要返回


def mix_up(im1, labels1, im2, labels2):
    """
    Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf.
    该图片增强函数的作用是，将两张图片进行混合，混合比例为 r
    """
    # 混合比例r，随机生成一个比例，这个比例服从beta分布
    r = np.random.beta(a=32.0, b=32.0)
    im = (im1 * r + im2 * (1 - r)).astype(np.uint8)  # cv2.imread读取的图片为numpy.ndarray-uint8
    labels = np.concatenate((labels1, labels2), axis=0)  # 图片标签也要拼接起来
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-6):
    """
    本函数的作用是将经过增强变换后的锚框坐标和原始的锚框坐标比对，过滤掉那些已经严重变形的锚框
    :param box1: np.ndarray类型，大小为 [4, n]，n为目标锚框的个数
    :param box2: np.ndarray类型，大小为 [4, n]，n为目标锚框的个数
    :param wh_thr: int类型，表示图片增强后锚框高height和宽width的阈值，任何一个小于阈值都是不正常的锚框
    :param ar_thr: int类型，表示图片增强后锚框的 (宽高比,高宽比) 较大的那个的阈值，如果超过这个阈值，表明锚框长和宽严重失衡
    :param area_thr: float类型，表示增强变换前后锚框面积的比值的阈值，如果小于这个阈值，同样说明前后锚框大小严重失衡
    :param eps: float类型，一个很小的数，用在分母上，防止出现除以0的情况
    :return:
    """
    # 分别取出box1和box2的宽w和高h
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]  # w1，h1的大小尺寸为 (n,) (n,) --> np.ndarray
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    # 求出增强变换后锚框 高宽比 或 宽高比 较大的那个
    aspect_ratio = np.maximum(w2 / (h2+eps), h2 / (w2+eps))

    # 最后得到的结果是：[true, false, ..., false] 长度为n的np数组，数组元素类型为bool型，用于指代在哪个锚框可取哪个锚框不可取
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2*h2 / (w1*h1+eps) > area_thr) & (aspect_ratio < ar_thr)


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    """
    本函数是获取仿射矩阵，以及图片变换的放缩尺度
    """
    new_height, new_width = new_shape
    """
    假设一个仿射变换矩阵为 M , 锚框的四角坐标之一为 P

        |a b c|
    M = |d e f|  P = | x y 1 |
        |0 0 1|

    其中，a,b,c,d,e,f 是变换矩阵的参数，而：
    (1)a 和 e 这两个参数控制了图像的缩放变换，分别代表水平和垂直方向的缩放比例
    (2)b 和 d 这两个参数表示了图像的旋转和错切变换
    (3)c 和 f 这两个参数表示控制平移变换和缩放变换，确定了图像水平和垂直方向的平移距离

                          |a d 0|   | ax+by+c |
    P @ M.T = | x y 1 | @ |b e 0| = | dx+ey+f |
                          |c f 1|   |    1    |  经过矩阵运算后，角坐标点P完成了变换
    """
    # center中心
    c = np.eye(3)
    c[0, 2] = -img_shape[1] / 2
    c[1, 2] = -img_shape[0] / 2

    # rotation旋转和scale缩放
    r = np.eye(3)
    a = random.uniform(-degrees, degrees)
    scaling = random.uniform(1 - scale, 1 + scale)
    r[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=scaling)

    # shear剪切操作
    s = np.eye(3)
    s[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    s[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # translation平移操作
    t = np.eye(3)
    t[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    t[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y translation (pixels)

    # 合并旋转矩阵
    m = t @ s @ r @ c
    return m, scaling


def random_affine(img, labels=(), degrees=10, translate=0.1,
                  scale=0.1, shear=10, new_shape=(640, 640)):
    """
    :param img:
    :param labels: 类型为数组，结构为 [label1, label2, ..., label_n]
    :param degrees:
    :param translate:
    :param scale:
    :param shear:
    :param new_shape:
    :return:
    """
    n = len(labels)
    height, width = new_shape

    # matrix的大小为：3 * 3
    matrix, scaling = get_transform_matrix(
        img_shape=img.shape[:2],
        new_shape=(width, height),
        degrees=degrees,
        scale=scale,
        shear=shear,
        translate=translate
    )
    if not np.all(np.equal(matrix, np.eye(3))):
        img = cv2.warpAffine(img, matrix[:2], dsize=(width, height),
                             borderValue=(114, 114, 114))

    if n:  # 如果标签列表不为空，那么锚框对应的坐标也要变换
        xy = np.ones((n * 4, 3))
        # (x1 y1 x2 y2) | (x1 y2 x2 y1)
        # --reshape-->
        # (x1 y1) | (x2 y2) | (x1 y2) | (x2 y1)
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ matrix.T
        xy = xy[:, :2].reshape(n, 8)  # reshape成每个锚框实体对应的四角xy坐标共计8个值

        # 创建新的，经过仿射变换后的labels的锚框对应的坐标值
        x = xy[:, [0, 2, 4, 6]]  # x1 x2 x1 x2
        y = xy[:, [1, 3, 5, 7]]  # y1 y2 y2 y1
        new_axis = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip，将变换后的锚框的对角坐标的取值限制在 0~width 或 0~height 之内
        new_axis[:, [0, 2]] = new_axis[:, [0, 2]].clip(0, width)
        new_axis[:, [1, 3]] = new_axis[:, [1, 3]].clip(0, height)

        # 过滤掉那些因为增强变换而变得过小、变形、高宽比例严重失衡的锚框
        # 但是要注意，原始的锚框要先按照增强处理的缩放尺度进行缩放后，再进行锚框比对操作
        """ "因为，锚框是用来框住目标实体的，试想一下如果图片经过了缩放，锚框也会同步缩放；
             原始图片没有经过缩放，因此对应的锚框也不会缩放，这两个锚框所代表的目标大小基础不一致，
             两个基础不一的锚框没有比较的意义" 
        """
        # 经过锚框筛选后得到应该选择的锚框的bool型数组
        corrupted_index = box_candidates(box1=labels[:, 1:5].T*scaling,  # box1.shape=[4, n]
                                         box2=new_axis.T, area_thr=0.1)
        labels = labels[corrupted_index]  # 选出没有严重变形的锚框，作为图片增强操作后的labels值
        labels[:, 1:5] = new_axis[corrupted_index]  # 把锚框的坐标值更新为仿射变换后的锚框坐标值
    return img, labels


def mosaic_augmentation(img_size, images, hs, ws, labels, hyp):
    """
    应用马赛克图片增强
    img_size: int类型
    hyp: dict类型，装载着诸如平移、旋转、剪切等图片增强操作的参数
    images: list类型，装着4张图片的列表，用于拼接生成马赛克增强图片
    hs: list类型，装着4张图片每张图片的高height
    ws: list类型，装着4张图片每张图片的宽width
    labels: list类型，装着每张图片的标签，其中，每个标签的样式为：np.ndarray.shape = [n, 5]
            表示每张图片有n个目标，每个目标有五个信息，分别是 类别、左上角x坐标、左上角y坐标、右下角x坐标、右下角y坐标
    """
    assert len(images) == 4, "Mosaic augmentation of current version only supports 4 images."

    s = img_size
    # ------------------------
    # 马赛克的中心点位，取值范围是
    # 横坐标方向 xc = 320~960
    # 纵坐标方向 yc = 320~960
    # ------------------------
    yc, xc = (int(random.uniform(s//2, 3*s//2)) for _ in range(2))

    labels4 = []
    img4 = np.full((s * 2, s * 2, images[0].shape[2]), fill_value=114, dtype=np.uint8)
    for i in range(len(images)):
        # 不断地把图片加载到img4这个大的画板上面
        im, h, w = images[i], hs[i], ws[i]
        if i == 0:  # 左上角的图片
            # 这一行代码表示取出马赛克的第一个片区(左上角片区)的对角坐标值，即左上角xy以及右下角xy
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            # 这一行代码表示截取原图片以放入马赛克左上角的部分，分别代表 原图上 宽的起始index，高的起始index
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # 右上角的图片
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, (x2a - x1a)), h
        elif i == 2:  # 左下角的图片
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * s)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # 右下角的图片
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(yc + h, 2 * s)
            x1b, y1b, x2b, y2b = 0, 0, min(x2a - x1a, w), min(y2a - y1a, h)

        # 每次循环直接将原始图片截取范围对应的内容填入img4中，最终构成马赛克图（4宫格）
        img4[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
        # pad填充值有可能是负数，因为对应了图片被组合边界截掉了一部分的情况
        pad_width = x1a - x1b
        pad_height = y1a - y1b

        # 标签的处理
        labels_per_img = labels[i].copy()
        if labels_per_img.size:
            boxes = np.copy(labels_per_img[:, 1:])
            boxes[:, 0] = w * (labels_per_img[:, 1] - labels_per_img[:, 3] / 2) + pad_width  # 左上角的x坐标
            boxes[:, 1] = h * (labels_per_img[:, 2] - labels_per_img[:, 4] / 2) + pad_height  # 左上角的y坐标
            boxes[:, 2] = w * (labels_per_img[:, 1] + labels_per_img[:, 3] / 2) + pad_width  # 右下角的x坐标
            boxes[:, 3] = h * (labels_per_img[:, 2] + labels_per_img[:, 4] / 2) + pad_height  # 右下角的y坐标
            labels_per_img[:, 1:] = boxes

        labels4.append(labels_per_img)

    # 拼接labels
    labels4 = np.concatenate(labels4, axis=0)
    for x in (labels4[:, 1:]):
        # 把锚框的坐标全部限制在大图片里面(1280 x 1280)
        np.clip(x, 0, 2 * s, out=x)

    # 图片增强处理
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=hyp["degrees"],
                                  translate=hyp["translate"],
                                  scale=hyp["scale"],
                                  shear=hyp['shear'],
                                  new_shape=(img_size, img_size))
    return img4, labels4


def letterbox(im, new_shape=(640, 640), scaleup=True, auto=False,
              stride=32, color=(114, 114, 114), return_int=False):
    """
    本函数的作用是将输入图片填充成正方形（边长是给定的new_shape）
    """
    # 首先获取图片的原始大小 shape[0] = height, shape[1] = width
    shape = im.shape[:2]  # 当前图片的大小 [height, width]，也等于函数外的 h 和 w

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    else:
        new_shape = new_shape

    # 先分别将图片的高、宽单独写出，增加代码可读性
    old_height, old_width = shape[0], shape[1]  # e.g. 640 * 427
    new_height, new_width = new_shape[0], new_shape[1]  # 640 * 640

    # 放缩率 (new / old)
    # -----------------------------------------------------------------------------------------
    # old_height = h, old_width = w;
    ratio = min(new_height / old_height, new_width / old_width)

    # 在构建验证集时，augment=False，因此进入if语句内：
    if not scaleup:
        ratio = min(ratio, 1.0)

    # ----------------------------------------------------------------------------------------
    # 总结一下到目前为止的 shape 信息：
    #       1. new_shape=[672, 512], 即 width=672， height=512
    #       2. shape=[480, 640], 即 width=640， height=480
    #       3. old_shape-[]

    # 计算填充padding，unpad指的是未填充前，原始图片resize到较长边小于等于给定new_shape的尺寸
    new_unpad = (int(round(old_width * ratio)), int(round(old_height * ratio)))  # 427 * 640

    # 分别计算高和宽填充至给定的new_shape所需要的像素大小
    # dw = 640 - 427   , dh = 640 - 640
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高的填充padding值

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2  # 因为padding是两边都要，因此平均分成两份左右填充
    dh /= 2

    if shape[::-1] != new_unpad:  # 如果原图片大小不等于预计未填充的大小
        # 先resize成最长边 <= 640的图片，后续再填充成640*640的正方形图片
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 直接算出上下左右四个方向的填充长度
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 根据四个方向的填充长度直接调用cv2函数填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    if not return_int:
        return im, ratio, (dw, dh)
    else:
        return im, ratio, (left, top)
