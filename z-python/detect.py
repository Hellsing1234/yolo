import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'F:\work\ultralytics-main\runs\detect\train9\weights\best.pt') # select your model.pt path
    model.predict(source=r'F:\work\paper\dataset\pests_single.v1i.yolov8\test\images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp-1-yolov8',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # visualize=True # visualize model features maps
                )