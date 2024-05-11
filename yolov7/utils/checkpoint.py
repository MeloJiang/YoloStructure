#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import shutil
import torch
import os.path as osp
from yolov7.utils.torch_utils import fuse_model


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """从checkpoints文件导入模型"""
    print("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:
        print("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """ Save checkpoint to the disk."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + '.pt')
    torch.save(ckpt, filename)
    if is_best:
        best_filename = osp.join(save_dir, 'best_ckpt.pt')
        shutil.copyfile(filename, best_filename)


