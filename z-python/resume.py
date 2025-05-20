from ultralytics import YOLO
if __name__ == '__main__':

    model = YOLO("/home/m608/project/yolo/ultralytics-main/runs/detect/train45/weights/last.pt")
    results = model.train(data="/home/m608/project/yolo/ultralytics-main/VOC.yaml", epochs=300, batch=4, workers=2, resume=True, device=0)

