from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


anno_json = 'coco/annotations/instances_val2017.json'
pred_json = 'predictions.json'

anno = COCO(anno_json)
pred = anno.loadRes(pred_json)
coco_eval = COCOeval(anno, pred, 'bbox')

coco_eval.evaluate()
coco_eval.accumulate()
val_dataset_img_count = coco_eval.cocoGt.imgToAnns.__len__()
coco_precision = coco_eval.eval['precision']
print(coco_precision[0, :, :, 0, 2].shape)
coco_precision_iou50 = coco_precision[0, :, :, 0, 2]
map50 = np.mean(coco_precision_iou50[coco_precision_iou50 > -1])
mean_precision = np.array([
    np.mean(coco_precision_iou50[k][coco_precision_iou50[k] > -1])  # 对所有类别的这个维度求平均
    for k in range(coco_precision_iou50.shape[0])
])
print(mean_precision.shape)
coco_eval.summarize()
print(coco_eval.stats)