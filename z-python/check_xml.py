import xml.etree.ElementTree as ET
import os

# 假设您的 XML 文件都在一个目录下：
xml_dir = '/home/m608/project/yolo/ultralytics-main/datasets/VOC2007/Annotations'

for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        try:
            in_file = os.path.join(xml_dir, xml_file)
            tree = ET.parse(in_file)
            # ... 进行您的处理逻辑
        except ET.ParseError as e:
            print(f"Error in file: {in_file}")
            print(f"Error message: {e}")