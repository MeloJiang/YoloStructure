from yolov7.model.head import Head
from yolov7.model.neck import RepPANNeck
from yolov7.model.efficient_rep import EfficientRep
from torch import nn


class ModelYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # -----------------------------
        # 首先先创建模型的 backbone 部分
        # -----------------------------
        self.backbone = EfficientRep(channels_list=[64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512],
                                     num_repeats=[1, 6, 12, 18, 6, 12, 12, 12, 12])

        # -----------------------------
        # 其次再创建模型的 neck 部分
        # -----------------------------
        self.neck = RepPANNeck(channels_list=[64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512],
                               num_repeats=[1, 6, 12, 18, 6, 12, 12, 12, 12])

        # -----------------------------
        # 最后再创建模型的 head 部分
        # -----------------------------
        self.head = Head(channels_list=[64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512])
        self.stride = self.head.stride

    def forward(self, x):
        outs1 = self.backbone(x)
        outs2 = self.neck(outs1)
        outs3 = self.head(outs2)

        return outs3


if __name__ == '__main__':
    model = ModelYOLO().train().cuda()
    import torch
    inputs = torch.rand(1, 3, 640, 640).cuda()
    outputs = model(inputs)
    for i in outputs[2]:
        print(i.shape)
