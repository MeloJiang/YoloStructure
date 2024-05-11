#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import time
from torch.cuda import amp
from tqdm import tqdm
from copy import deepcopy
from yolov7.data.data_load import create_dataloader
from yolov7.utils.events import load_yaml
from yolov7.model.model_yolo import ModelYOLO
from yolov7.solver.build import build_optimizer, build_lr_scheduler
from yolov7.losses.loss import ComputeLoss

from torch.nn.parallel import DistributedDataParallel as DDP
from yolov7.utils.checkpoint import *
from yolov7.core.evaluater import Evaler
from yolov7.utils.general import *
import pyfiglet


class Trainer:
    def __init__(self, args, cfg, device, is_half=True):
        self.args = args
        self.cfg = cfg
        self.device = device
        self.max_epoch = args.epochs

        # 决定是否要开启半精度训练
        self.is_half = is_half

        self.start_time = 0
        self.start_epoch = 0

        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir

        print(pyfiglet.Figlet('slant').renderText('YOLO Engine'))
        print("🔥YOLO model training...\n")
        self.data_dict = load_yaml(args.data_path)

        self.train_loader, self.val_loader = self.get_data_loader(
            args=self.args, cfg=self.cfg, data_dict=self.data_dict
        )
        # self.train_loader, self.val_loader = None, None

        self.max_stepnum = len(self.train_loader)

        self.model = self.get_model(args, cfg, device)

        self.optimizer = self.get_optimizer(args, cfg, model=self.model)
        self.scheduler, self.lr = self.get_lr_scheduler(args, cfg, optimizer=self.optimizer)

        self.epoch = 0
        self.step = None
        self.batch_data = None
        self.img_size = args.img_size
        self.batch_size = args.batch_size

        self.write_trainbatch_tb = args.write_trainbatch_tb

        self.loss_num = 3
        self.loss_info = ['Epoch', 'lr', 'iou_loss', 'dfl_loss', 'cls_loss']

        self.ap, self.best_ap = (None, None)
        self.mean_loss, self.accumulate, self.last_opt_step = (None, None, None)
        self.evaluate_results = None

        self.scaler = None

    def train(self):
        try:
            self.before_train_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.train_one_epoch(self.epoch)
                self.after_epoch()

        except Exception as _:
            print('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()

    def get_model(self, cfg, nc, device):
        model = ModelYOLO().to(device)
        return model

    def get_optimizer(self, args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        cfg.solver.lr0 *= args.batch_size / (self.world_size * args.bs_per_gpu)
        optimizer = build_optimizer(cfg, model)
        return optimizer

    def before_train_loop(self):
        print('\nTraining start...')
        self.start_time = time.time()
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.scaler = amp.GradScaler(enabled=self.is_half)

        self.best_ap, self.ap = 0.0, 0.0
        self.best_stop_strong_aug_ap = 0.0
        self.evaluate_results = (0, 0)

        self.compute_loss = ComputeLoss(
            num_classes=self.data_dict['nc'],
            original_img_size=self.img_size,
            iou_type=self.cfg.model.head.iou_type,
            fpn_strides=self.cfg.model.head.strides
        )

    def before_epoch(self):
        # 如果进行到的 epoch=400 - 15，此时停止对训练数据进行图片增强
        if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
            print('♻️Data Augmentation is no longer needed, REBUILD the Dataloader.\n')
            self.train_loader, self.val_loader = self.get_data_loader(
                self.args, self.cfg, self.data_dict)

        print('#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#')
        print('Preparation before EPOCH: \n'
              '(1) Optimizer zero_grad and zero mean_loss.\n'
              '(2) process bar onload.\n')
        self.model.train()
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        self.mean_loss = torch.zeros(self.loss_num, device=self.device)
        self.optimizer.zero_grad()

        self.pbar = enumerate(self.train_loader)
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=80,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    def train_one_epoch(self, epoch_num):
        print(f'🚩Start training in EPOCH: {epoch_num} .')
        print(f'training in {"torch.float16" if self.is_half else "torch.float32"}.')
        time.sleep(1.5)
        try:
            for self.step, self.batch_data in self.pbar:
                self._train_in_steps(epoch_num, self.step)
        except Exception as _:
            print('ERROR in training steps.')
            raise
        print('#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#\n')

    def _train_in_steps(self, epoch_num, step_num):
        images, targets = self.prepro_data(self.batch_data, self.device)

        # forward 环节
        with amp.autocast(enabled=self.is_half):
            _, _, batch_height, batch_width = images.shape
            preds = self.model(images)

            # 那么问题就定位到了标签这块
            total_loss, loss_items = self.compute_loss(
                outputs=preds, targets=targets
            )
            if self.rank != -1:
                total_loss *= self.world_size

        # backward 环节
        self.scaler.scale(total_loss).backward()
        self.loss_items = loss_items
        self.update_optimizer()

    def after_epoch(self):
        # lrs_of_this_epoch = [x['lr'] for x in self.optimizer.param_groups]
        # self.scheduler.step()
        if self.main_process:
            remaining_epochs = self.max_epoch - 1 - self.epoch  # self.epoch 从 0 开始
            eval_interval = self.args.eval_interval if remaining_epochs >= self.args.heavy_eval_range \
                else min(3, self.args.eval_interval)
            is_val_epoch = (remaining_epochs == 0) or ((not self.args.eval_final_only)
                                                       and ((self.epoch + 1) % eval_interval == 0))
            if is_val_epoch:
                print('🧪Start Evaluating YOLO model.')
                self.eval_model()
                self.ap = self.evaluate_results[1]
                self.best_ap = max(self.ap, self.best_ap)

            # 保存 checkpoints
            ckpt = {
                'model': deepcopy(self.model),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'results': self.evaluate_results
            }
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            save_checkpoint(ckpt=ckpt, is_best=is_val_epoch and (self.ap == self.best_ap),
                            save_dir=save_ckpt_dir, model_name='last_ckpt')

            if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
                save_checkpoint(ckpt=ckpt, is_best=False, save_dir=save_ckpt_dir,
                                model_name=f'{self.epoch}_ckpt')

            del ckpt
            self.evaluate_results = list(self.evaluate_results)

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))

        # if curr_step - self.last_opt_step >= self.accumulate:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.last_opt_step = curr_step

    def train_after_loop(self):
        if self.device != 'cpu':
            print('⬇️Flushing CUDA Memory...')
            torch.cuda.empty_cache()

    def eval_model(self):
        results = self.run_eval(
            data=self.data_dict,
            batch_size=self.batch_size // self.world_size * 2,
            image_size=self.img_size, task='train',
            model=self.model, conf_thres=0.03,
            dataloader=self.val_loader,
            save_dir=self.save_dir,
        )
        print(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
        self.evaluate_results = results[:2]

    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf

    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        print('Building Training Validation Dataloader for YOLO model training:')
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = int(data_dict['nc'])
        class_names = data_dict['names']

        grid_size = max(int(max(cfg.model.head.strides)), 32)

        # 创建训练数据集 train_data_loader:
        train_loader = create_dataloader(
            path=train_path,
            img_size=args.img_size,
            batch_size=args.batch_size // args.world_size,
            stride=grid_size,
            hyp=dict(cfg.data_aug),
            augment=True,
            rect=args.rect,
            rank=args.local_rank,
            workers=args.workers,
            shuffle=False,
            check_images=args.check_images,
            check_labels=args.check_labels,
            data_dict=data_dict,
            task='train'
        )[0]

        # 创建验证数据集 val_data_loader:
        val_loader = None
        if args.rank in [-1, 0]:
            val_loader = create_dataloader(
                path=val_path,
                img_size=args.img_size,
                batch_size=args.batch_size // args.world_size * 2,
                stride=grid_size,
                hyp=dict(cfg.data_aug),
                rect=True,
                rank=-1,
                pad=0.5,
                workers=args.workers,
                check_images=args.check_images,
                check_labels=args.check_labels,
                data_dict=data_dict,
                task='val',
                shuffle=False
            )[0]

        return train_loader, val_loader

    @staticmethod
    def prepro_data(batch_data, device):
        images = batch_data[0].to(device) / 255.0
        labels = batch_data[1].to(device)
        return images, labels

    @staticmethod
    def parallel_model(args, model, device):
        # 如果是DP模式
        dp_mode = device.type != 'cpu' and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            print('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)

        # 如果是DDP模式
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

        return model

    @torch.no_grad()
    def run_eval(self, data, weights=None, batch_size=32, image_size=640,
                 conf_thres=0.03, iou_thres=0.65, device='cuda', task='val',
                 half=False, model=None, dataloader=None, save_dir='',
                 name='', shrink_size=640, infer_on_rect=False, verbose=False,
                 do_coco_metric=True, do_pr_metric=False, plot_curve=False,
                 plot_confusion_matrix=False, specific_shape=False,
                 height=640, width=640,):
        """
        Run the Evaluation process
            This function is the main process of evaluation, supporting image file and dir containing images.
            It has tasks of 'val', 'train' and 'speed'. Task 'train' processes the evaluation during training phase.
            Task 'val' processes the evaluation purely and return the mAP of model.pt. Task 'speed' processes the
            evaluation of inference speed of model.pt.
        """
        Evaler.check_task(task=task)
        if task == 'train':
            save_dir = save_dir
        else:
            save_dir = str(increment_name(save_dir))
            os.makedirs(save_dir, exist_ok=True)

        Evaler.check_threshold(conf_thres, iou_thres, task)

        data = Evaler.reload_dataset(data, task) if isinstance(data, str) else data

        val = Evaler(data=data, batch_size=batch_size, img_size=image_size,
                     conf_threshold=conf_thres, iou_threshold=iou_thres, device=device,
                     half=half, save_dir=save_dir, shrink_size=shrink_size, verbose=verbose,
                     do_coco_metric=do_coco_metric, do_pr_metric=do_pr_metric,
                     plot_curve=plot_curve, plot_confusion_matrix=plot_confusion_matrix,
                     specific_shape=specific_shape, height=height, width=width,
                     infer_on_rect=infer_on_rect)
        model = val.init_model(model, weights=weights, task=task)
        dataloader = val.init_dataloader(dataloader, task=task)

        # 开始验证模型：
        model.eval()
        pred_result = val.predict_model(model=model, dataloader=dataloader, task=task)
        eval_result = val.eval_model(pred_results=pred_result, model=model,
                                     dataloader=dataloader, task=task)
        return eval_result



