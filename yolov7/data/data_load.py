import os
from torch.utils.data import dataloader, distributed, DataLoader
from yolov7.utils.torch_utils import torch_distributed_zero_first
from yolov7.data.datasets import LoadImagesAndLabels
import torch.distributed as dist

import torch
import numpy as np
from yolov7.utils.general import xywh2xyxy, dist2bbox


def create_dataloader(path, img_size, batch_size, stride, hyp=None, augment=False,
                      check_images=False, check_labels=False, pad=0.0, rect=False,
                      rank=-1, workers=8, shuffle=False, data_dict=None, task='val'):
    if rect and shuffle:
        shuffle = False
    print("+------------------------------------------------------------------+")
    print(f"Now creating {'Validation' if task=='val' else 'Train'} dataset...🚀")
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path=path,
            img_size=img_size,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            check_images=check_images,
            check_labels=check_labels,
            stride=int(stride),
            pad=pad,
            rank=rank,
            data_dict=data_dict,
            task=task)
    batch_size = min(batch_size, len(dataset))
    # 多线程创建DataLoader使用的workers
    # 选择：每个线程分得的CPU，批次数目，预设workers 的最小值
    workers = min(
        [
            os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),
            batch_size if batch_size > 1 else 0,
            workers
        ])
    print(f"Using {workers} worker(s) for DataLoader...")
    # 如果使用pytorch.distributed，则需要丢弃尾部数据
    drop_last = rect and dist.is_initialized() and dist.get_world_size() > 1
    sampler = (
        None if rank == -1 else distributed.DistributedSampler(
            dataset=dataset, shuffle=shuffle, drop_last=drop_last
        )
    )
    print(f"✅{'Validation' if task=='val' else 'Train'} dataset has been SUCCESSFULLY CREATED!")
    print("+------------------------------------------------------------------+\n")
    # 返回dataloader 以及 dataset
    return (
        InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=workers,
            sampler=sampler,
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn,
            drop_last=True
        ),
        dataset
    )


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def preprocess(targets, batch_size, scale_tensor):
    targets_list = np.zeros((batch_size, 1, 5)).tolist()
    for i, item in enumerate(targets.cpu().numpy().tolist()):
        # item[0]表示的是这个gt框属于这一批次中的哪张图片
        targets_list[int(item[0])].append(item[1:])
    max_len = max((len(l) for l in targets_list))
    print(max_len)
    targets = torch.from_numpy(
        # 将targets中每张图片锚框数量不足max_len的用[-1, 0, 0, 0, 0]填充直至长度满足max_len
        np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]
    ).to(targets.device)
    batch_target = targets[:, :, 1:5].mul_(scale_tensor)
    targets[..., 1:] = xywh2xyxy(batch_target)
    return targets


def bbox_decode(anchor_points, pred_dist):
    return dist2bbox(pred_dist, anchor_points)


if __name__ == '__main__':
    from yolov7.utils.events import load_yaml
    from yolov7.model.model_yolo import ModelYOLO
    from yolov7.assigners.anchor_generator import generate_anchors
    from yolov7.assigners.atss_assigners import ATSSAssigner
    from yolov7.losses.loss import ComputeLoss
    dl, ds = create_dataloader(path='../../coco/images/val2017',
                               img_size=640, batch_size=16, stride=32, hyp={},
                               data_dict=load_yaml('../../data/coco.yaml'))

    assigner = ATSSAssigner(9, num_classes=80)
    model = ModelYOLO().train()
    for i, data in enumerate(dl):
        outputs = model(data[0])
        targets = data[1]
        batch_size = 16
        compute_loss = ComputeLoss()
        compute_loss(outputs, targets)
        break


