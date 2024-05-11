import os
import cv2
import time
import math
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from collections import deque

from yolov7.layers.common import DetectBackend
from yolov7.utils.events import load_yaml
from yolov7.data.datasets import LoadData
from yolov7.data.data_augment import letterbox
from yolov7.utils.nms import non_max_suppression


class Inference:
    def __init__(self, device, source, webcam, webcam_addr,
                 weights, yaml, img_size, half):
        self.__dict__.update(locals())

        # 初始化模型
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')

        self.model = DetectBackend(weights, device=self.device)

        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']

        self.img_size = self.check_img_size(img_size, s=self.stride)
        self.half = half

        # 切换模型至deploy模式
        self.model_switch(self.model.model, self.img_size)

        # 数据加载...
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        self.files = LoadData(source, webcam, webcam_addr)
        self.source = source

        # 是否保存完成目标检测的图片的变量：
        self.save_txt, self.save_img = None, None
        self.hide_labels, self.hide_conf = None, None

    def infer(self, conf_thres, iou_thres, classes, agnostic_nms, max_det,
              save_dir, save_txt, save_img, hide_labels, hide_conf, view_img=False):
        """Model Inference and results visualization """
        vid_path, vid_writer, windows = None, None, []
        self.save_img, self.save_txt, self.hide_labels, self.hide_conf =\
            save_img, save_txt, hide_labels, hide_conf

        fps_calculator = CalFPS()
        for img_src, img_path, vid_cap in tqdm(self.files):
            img, img_src = self.process_image(img_src, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]

            # YOLOModel推理并执行极大值抑制算法NMS的过程：
            # --------------------------------------------------------------------------------
            # 1. 首先在代码块开始时记录一下开始时间：
            #       t1 = time.time() 表示开始推理的时间
            # 2. 然后对输入图片进行YOLO模型推理，得到预测结果：
            #       pred_results = self.model(img) 表示得到推理计算后的预测结果pred_results
            # 3. 接着对初步的推理预测结果执行极大值抑制算法(NMS), 得到取出重复锚框的预测结果
            #       pred_results => NMS => det, 并且 det.shape = [min(max_det, num_NMS), 6]，表
            #       示有多少个预测锚框，每个锚框由 4 个坐标信息，1 个置信度信息以及 1 个类别信息构成
            # 4. 最后记录一下完成推理和NMS的时间，以便后续计算 FPS
            #       t2 = time.time() 表示结束推理时间
            # --------------------------------------------------------------------------------
            t1 = time.time()
            pred_results = self.model(img)
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes,
                                      agnostic_nms, max_det=max_det)[0]
            t2 = time.time()

            save_path, txt_path = self._inference_saved_path_generator(save_dir, img_path)

            assert img_src.data.contiguous, \
                ('Image needs to be contiguous. Please apply to input images with '
                 'np.ascontiguousarray(im).')
            self.font_check()

            if len(det):
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    self._save_img_and_text(
                        img_src=img_src, xyxy=xyxy, cls=cls, conf=conf, txt_save_path=txt_path,
                        img_save_path=img_path)

            # 计算图片处理的速度FPS
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()

            if self.files.type == 'video':
                self.draw_text(
                    img=img_src, text=f"FPS: {avg_fps:0.1f}", pos=(20, 20),
                    text_color=(204, 85, 17), text_color_bg=(255, 255, 255),
                    font_thickness=2, font_scale=1)

            if view_img:
                if img_path not in windows:
                    windows.append(img_path)
                    cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
                cv2.imshow(str(img_path), img_src)
                cv2.waitKey(1)

            if save_img:
                if self.files.type == 'image':
                    cv2.imwrite(filename=str(save_path), img=img_src)
                else:  # 摄像头webcam流或者视频video流
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # 把之前的video保存writer先释放掉
                        if vid_cap:
                            # 如果是video类型的输入，获取视频文件的帧率
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            # 如果是摄像头webcam流的输入
                            fps, w, h = 30, img_src.shape[1], img_src.shape[0]
                        save_path = str(Path(str(save_path)).with_suffix('.mp4'))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*'mp4v'),
                                                     fps, (w, h))
                    vid_writer.write(img_src)

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in
        each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size, list) else [new_size]*2

    def _inference_saved_path_generator(self, save_dir, img_path):
        if self.webcam:
            save_path = osp.join(str(save_dir), str(self.webcam_addr))
            txt_path = osp.join(str(save_dir), str(self.webcam_addr))
        else:
            rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(self.source))
            save_path = osp.join(save_dir, rel_path, osp.basename(img_path))
            txt_path = osp.join(save_dir, rel_path, 'labels', osp.splitext(osp.basename(img_path))[0])
            os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)

        return save_path, txt_path

    def _save_img_and_text(self, img_src, xyxy, cls, conf, img_save_path, txt_save_path):
        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]
        if self.save_txt:
            xywh = list((self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1))
            line = (cls, *xywh, conf)
            with open(txt_save_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

        if self.save_img:
            class_num = int(cls)
            label = None \
                if self.hide_labels \
                else (self.class_names[class_num]
                      if self.hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
            self.plot_box_and_label(
                image=img_src, lw=max(round(sum(img_src.shape)/2*0.003), 2), box=xyxy,
                label=label, color=self.generate_colors(class_num, True)
            )

    @staticmethod
    def model_switch(model, img_size):
        """Model switch to deploy status """
        from yolov7.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif (isinstance(layer, torch.nn.Upsample)
                  and not hasattr(layer, 'recompute_scale_factor')):
                layer.recompute_scale_factor = None
        print("Switch model to deploy modality.")

    @staticmethod
    def make_divisible(x, divisor):
        """"""
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def process_image(img_src, img_size, stride, half):
        """Process image before image inference."""
        image = letterbox(img_src, img_size, stride=stride)[0]

        image = image.transpose(2, 0, 1)[::-1]  # 读取图片的顺序改变：HWC-->CHW, BGR-->RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()
        image /= 255.0

        return image, img_src

    @staticmethod
    def font_check(font='', size=10):
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if osp.exists(font) else osp.basename(font), size)
        except FileNotFoundError as e:
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def rescale(original_shape, boxes, target_shape):
        """Rescale the output to the original image shape"""
        ratio = min(original_shape[0] / target_shape[0], original_shape[1] / target_shape[1])
        padding = ((original_shape[1] - target_shape[1] * ratio) / 2,
                   (original_shape[0] - target_shape[0] * ratio) / 2)

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        # -------------------------------------------------------------
        # boxes[:, 0].clamp_(0, target_shape[1])
        # boxes[:, 1].clamp_(0, target_shape[0])
        # boxes[:, 2].clamp_(0, target_shape[1])
        # boxes[:, 3].clamp_(0, target_shape[0])
        # 上述四行代码可以使用一个 for-loop 来代替：
        # --------------------------------------------------------------
        for i in range(4):
            boxes[:, i].clamp_(0, target_shape[(i + 1) % 2])

        return boxes

    @staticmethod
    def box_convert(x):
        """
        把形状为: shape=[n, 4] 的锚框从 [x1, y1, x2, y2] 转换为 [x, y, w, h] 其中
        x1y1 为左上角的横纵坐标，x2y2为右下角的横纵坐标
        """
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17',
               '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF', '344593', '6473FF',
               '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            # 将十六进制的颜色表示两个两个地取出，并转化为十进制的数字，用以构建(R, G, B)格式的颜色表示
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color

    @staticmethod
    def draw_text(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0),
                  font_scale=1, font_thickness=2, text_color=(0, 255, 0),
                  text_color_bg=(0, 0, 0)):
        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        # --------------------------------------------------------------------------
        # cv2.rectangle() 的参数含义：
        #   1. img: 所接受的输入是经过读取后的图片，也就是需要被绘制的图片
        #   2. pt1: 需要绘制的矩形框的左上角的坐标点 (x1, y1)
        #   3. pt2: 需要绘制的矩形框的右下角的坐标点 (x2, y2)
        #   4. color: 矩形边框或填充的颜色、亮度
        #   5. thickness: 矩形边框的粗细，负值表示使用color填充整个矩形框
        #   6. lineType: 矩形边框的线形
        # --------------------------------------------------------------------------
        cv2.rectangle(
            img=img, pt1=rec_start, pt2=rec_end, color=text_color_bg, thickness=-1
        )

        # --------------------------------------------------------------------------
        # cv2.putText() 的参数含义：
        #   1. img: 所接受的输入是经过读取后的图片，这也是把文本往上面写的目标图片
        #   2. text: 往图片上面写入的文本的内容
        #   3. org: 文本的左下角的坐标 (x_left_bottom, y_left_bottom)
        #   4. fontFace: 所使用的文本字体
        #   5. fontScale: 诚意特定字体基本大小的比例因子
        # --------------------------------------------------------------------------
        cv2.putText(
            img=img, text=text, org=(x, int(y + text_h + font_scale - 1)),
            fontFace=font, fontScale=font_scale, color=text_color,
            thickness=font_thickness, lineType=cv2.LINE_AA
        )
        return text_size

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128),
                           text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
        # 首先先把xyxy格式的锚框，取出其左上角、右下角的坐标的横纵坐标的大小
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        # -------------------------------------------------------------------------------
        # cv2.rectangle() 的作用是将输入的图片，根据box的坐标信息画出对应的方框
        #   1. img=image, 即接受输入的图片，在其上画出方框
        #   2. pt1=p1, 即锚框的左上角的坐标信息 (x_left_top, y_left_top)
        #   3. pt2=p2, 即锚框的右下角的坐标信息 (x_right_bottom, y_right_bottom)
        #   4. color=color, 即所绘制锚框的边框颜色
        #   5. thickness=lw, 即绘制的锚框的线的粗细(line-weight)
        #   6. lineType=cv2.LINE_AA, 绘制的锚框的线的类型
        # --------------------------------------------------------------------------------
        cv2.rectangle(
            img=image, pt1=p1, pt2=p2, color=color, thickness=lw, lineType=cv2.LINE_AA
        )

        if label:
            # 首先先确定文本的内容的字体的线粗细，不能比锚框粗太多
            tf = max(lw - 1, 1)

            # ----------------------------------------------------------------------------
            # cv2.getTextSize() 函数的作用是计算并返回即将填入的文本的宽和高
            #   1. text=label, 即将填入的文本的内容，如锚框识别的类别 person, bicycle, car
            #   2. fontFace=0, 表示填入的文本所要使用的字体
            #   3. fontScale=lw / 3, 乘以特定字体基本大小的比例因子
            #   4. thickness=tf, 所要填入的文本的线的粗细
            # ----------------------------------------------------------------------------
            w, h = cv2.getTextSize(
                text=label, fontFace=0, fontScale=lw / 3, thickness=tf
            )[0]

            # outside表示：如果锚框的上端离图片上边缘的位置不够填入标签文本了，那就填在锚框的内部
            outside = p1[1] - h - 3 >= 0

            # 计算文本标签框的p2(框的对角坐标)
            # -------------------------------------------------------------------------------
            # p2[0] = p1[0] + w, 文本标签框的右下角的横坐标直接用锚框的左上角横坐标加上文本框宽度 w 即可
            # p2[1] 即文本标签框的右下角的纵坐标就会受到 outside 的影响：如果文本标签框应该绘制在检测锚框
            #       的外侧，那么纵坐标相当于要对应 p1 去减掉文本框的高 h，得到文本框右上角坐标 p2，此时传入
            #       cv2.rectangle() 函数的对角坐标为左下角和右上角; 而当应该将文本框绘制到锚框内部时，纵
            #       坐标相当于要对应 p1 加上文本框的高h，得到文本框的右下角坐标 p2，此时传入cv2.rectangle()
            #       函数的对角坐标为左上角和右下角
            #
            # thickness=-1，因为绘制线框的线条粗细参数设置为负数，因此结果为用相应的颜色填充这个线框
            # -------------------------------------------------------------------------------
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(
                img=image, pt1=p1, pt2=p2, color=color, thickness=-1, lineType=cv2.LINE_AA
            )

            # --------------------------------------------------------------------------------
            # cv2.putText() 函数的作用是将指定文本写入指定的图片之中
            #   1. img=image, 表示即将写入文本的图片
            #   2. text-label, 表示需要被写入的文本内容
            #   3. org, 表示文本的左下角坐标，如果文本标签要写在锚框的外侧，那么文本的左下角坐标就等于锚框的
            #           左上角坐标; 如果文本标签要写在锚框的内侧，那么文本的左下角的横坐标还是等于锚框的左上
            #           角坐标的横坐标值，但是纵坐标值就等于锚框左上角的纵坐标加上文本框的高度
            # --------------------------------------------------------------------------------
            cv2.putText(
                img=image, text=label, org=(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                fontFace=font, fontScale=lw / 3, color=text_color, thickness=tf, lineType=cv2.LINE_AA
            )

    @staticmethod
    def draw_text(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), font_scale=1,
                  font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0)):
        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text=text, fontFace=font, fontScale=font_scale,
                                       thickness=font_thickness)
        text_w, text_h = text_size
        rect_start = tuple(x - y for x, y in zip(pos, offset))
        rect_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img=img, pt1=rect_start, pt2=rect_end, color=text_color_bg, thickness=-1)
        cv2.putText(
            img=img, text=text, org=(x, int(y + text_h + font_scale - 1)), fontFace=font,
            fontScale=font_scale, color=text_color, thickness=font_thickness, lineType=cv2.LINE_AA
        )


class CalFPS:
    def __init__(self, n_samples=50):
        """
        deque 是一个线程安全的队列，适用于快速在其两端添加元素
        """
        self.framerate = deque(maxlen=n_samples)

    def update(self, duration):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
