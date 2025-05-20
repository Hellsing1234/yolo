import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/home/m608/project/yolo/ultralytics-main/runs/detect/0-yolov8+MBConv/weights/best.pt')
    model.val(data='/home/m608/project/yolo/ultralytics-main/datasets/VOC2007/images/val',
              split='val',
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )