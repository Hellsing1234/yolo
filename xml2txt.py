import xml.etree.ElementTree as ET
import pickle
import os
import shutil

from os import listdir, getcwd
from os.path import join
from shutil import copyfile
# 获取全部类别标签

classes = [str(i) for i in range(102)] 
def gen_classes(image_id):
    in_file = open('%s/Annotations/%s.xml'%(path,image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name in classes:
            pass
        else:
            classes.append(cls_name)
    return classes

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_set,image_id):
    in_file = open('%s/Annotations/%s.xml'%(path,image_id))
    out_file = open('%s/%s'%(path,image_set)+'labels/%s.txt'%image_id, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

path = r'/home/m608/project/yolo/ultralytics-main/datasets/VOC2007'
imagetype='.jpg'
sets = ['train','test','val']
for image_set in sets:
    if not os.path.exists('%s/%s'%(path,image_set)+'labels/'):
        os.makedirs('%s/%s'%(path,image_set)+'labels/')
    if not os.path.exists('%s/%s'%(path,image_set)+'imgs/'):
        os.makedirs('%s/%s'%(path,image_set)+'imgs/')
    image_ids = open('%s/ImageSets/Main/%s.txt'%(path,image_set)).read().strip().split()
    for image_id in image_ids:
        gen_classes(image_id)
        convert_annotation(image_set,image_id)
        #copyfile('%s/JPEGImages/%s'%(path,image_id)+'.png','%s/JPEGImages/%s'%(path,image_set)+'imgs/')

        shutil.copy('%s/JPEGImages/%s'%(path,image_id)+imagetype,'%s/%s'%(path,image_set)+'imgs/')

    classes_file = open('%s/%s'%(path,image_set)+'labels/classes.txt','w')
    classes_file.write("\n".join([a for a in classes]))
    classes_file.close()