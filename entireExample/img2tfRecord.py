# -*-coding:utf-8-*-
import numpy as np
import os
import tensorflow as tf
from PIL import Image

classes=["wood"]

def convert(box):
    """
    args:
    size,the direct image size ,yolo v3 [416,416]
    box,[xmin,ymin,xmax,ymax,c]
    return the relative value of [x,y,w,h,c]
    """
    size=[416,416]
    dw=1./size[0]
    dh=1./size[1]
    x=(box[0]+box[2])/2.0
    y=(box[1]+box[3])/2.0
    w=box[2]-box[0]
    h=box[3]-box[1]
    x=x*dw
    w=w*dw
    y=y*dh
    h=h*dh
    c=box[-1]
    return [x,y,w,h,c]
def convert_img(img_path):
    """
    args:
    img_path:input image_path
    """
    image=Image.open(img_path)
    resized_image=image.resize((416,416),Image.BICUBIC)
    image_data=np.array(resized_image,dtype="float32")/255
    img_raw=image_data.tobytes()
    return img_raw
def convert_annocation(boxes):
    """
    args:[x,y,w,h,c]
    make sure the len of the boxes is the same. so can deal this with map
    """

    if len(boxes)<30:
        boxes=boxes+[0,0,0,0,0]*[30-len(boxes)]
    return np.array(boxes,dtype=np.float32).flatten().tolist()
filename=os.path.join("dataset.tfRecord")
writer=tf.io.TFRecordWriter(filename)
with open("./data.txt","r") as f:
    lines=f.readlines()
for line in lines:
    infos=line.split(" ")
    img_raw=convert_img("./dataset/"+infos[0])
    boxes=infos[1:]
    boxes=np.array(boxes,dtype=np.float32)
    boxes=boxes.reshape(-1,5)
    boxes=np.array(list(map(convert,boxes)))
    boxes=boxes.flatten().tolist()
    if len(boxes)<30*5:
        boxes=boxes+[0]*(30*5-len(boxes))
    example=tf.train.Example(features=tf.train.Features(feature={
        'xywhc':tf.train.Feature(float_list=tf.train.FloatList(value=boxes)),
        "img":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())
writer.close()





