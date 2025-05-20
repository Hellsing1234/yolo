import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# if __name__ == '__main__':
#     # 加载模型配置
#     model = YOLO(r'F:\work\ultralytics-main\yamls\yolov8-C2f-MLCA-CSFCN-TADDH-DyS.yaml')
#     #model.load('yolov8n.pt')
#     model.train(data=r"F:\work\paper\dataset\denseFinal\densepset.v2i.yolov8\data.yaml", epochs=200,batch=16,pretrained = False)
if __name__ == '__main__':
    model = YOLO(r'F:\work\ultralytics-main\yamls\yolov8-C2f_DSCA-CSFCN-AYHead-Dy.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'F:\work\paper\dataset\apid.v2i.yolov8\data.yaml',
                cache=False, 
                imgsz=640,
                pretrained = True,               epochs=220,
                batch=16,
                close_mosaic=0, 
                workers=4,# device='0',x
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )