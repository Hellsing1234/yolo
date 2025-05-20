import os

# 设置label文件夹的路径
labels_path = '/media/m608/文档/dataset/my_yolo_data/labels/train'

# 确认路径存在
if not os.path.isdir(labels_path):
    print(f"路径不存在: {labels_path}")
else:
    print(f"正在扫描路径: {labels_path}")

# 创造一个集合来保存所有唯一的类别取值
unique_classes = set()

# 遍历label文件夹下的所有文件
for label_file in os.listdir(labels_path):
    file_path = os.path.join(labels_path, label_file)
    # 确保仅处理文本文件
    if label_file.endswith('.txt'):
        print(f"正在处理文件: {file_path}")
        with open(file_path, 'r') as file:
            # 遍历文件的每一行
            for line in file:
                if line.strip():  # 确保不是空行
                    # 获取标签行中的类别部分（第一个元素）
                    class_id = line.split()[0]
                    unique_classes.add(class_id)

# 打印出所有的唯一类别
if unique_classes:
    print("所有标签文件中的唯一类别取值有：")
    print(unique_classes)
else:
    print("没有发现任何类别标签。")