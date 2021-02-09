# -*-oding:utf-8-*-
# produce dataset of custom
'''
use labelme label the datasets
produce data.txt with boxes and classes information in line :imgpath,xmin0,ymin0,xmax0,ymax0, cls,xmin1,ymin1,xmax1,ymax1,cls,...
'''
import json
import os
path='./dataset'
filenames=os.listdir(path)
with open('data.txt','a',encoding='utf-8') as f:
    for file in filenames:
        print(file)
        filename=file.split('.')
        if filename[1] == 'json':
            with open('./dataset/'+file, 'r', encoding='utf8')as fp:
                json_data = json.load(fp)
                shapes=json_data['shapes']
                img_name=filename[0]+'.jpg'
                f.write('\n'+img_name)
                for shape in shapes:
                    points=shape['points']
                    pointstring=' '+str(int(points[0][0]))+' '+str(int(points[0][1]))+' '+str(int(points[1][0]))+' '+str(int(points[1][1]))+' 1'
                    f.write(pointstring)