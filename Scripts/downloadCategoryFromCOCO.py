from pycocotools.coco import COCO
coco = COCO('C:/Users/aryan/Desktop/annotations/instances_train2014.json')

import os
import urllib.request as request
from urllib.request import urlopen
import pandas as pd

testSize = 200
trainDF = pd.DataFrame(columns=['filename', 'width', 'height', 'class','xmin','ymin','xmax','ymax'])
testDF = pd.DataFrame(columns=['filename', 'width', 'height', 'class','xmin','ymin','xmax','ymax'])

categories = ['person','bicycle','motorcycle','car','truck','bus','cow','dog']

for category in categories[:]:
    count = 1
    categoryID = coco.getCatIds(catNms=[category])
    imageIDs = coco.getImgIds(catIds=categoryID )
    images = coco.loadImgs(imageIDs)
    imagesLen = len(images)
    
    for i,image in enumerate(images[:imagesLen]):
        imageID = str(image['id'])

        label = 'test' if (imagesLen-count) < testSize else 'train'
        request.urlretrieve(image['coco_url'], f'C:/Users/aryan/Desktop/objectDetection/images/{label}/'+f'{imageID}.jpg')
                
        annotationsID = coco.getAnnIds(imgIds=image['id'], catIds=categoryID)
        annotations = coco.loadAnns(annotationsID)
            
        print(f'{count}. INFO => Category - {category} | Image ID - {imageID} | #Annotations = {len(annotations)} | Label = {label} | Completion = {i+1}/{len(images)}.')
            
        for a in annotations:
            data = {
                'filename' : imageID+'.jpg',
                'width' : image['width'],
                'height' : image['height'],
                'class' : category,
                'xmin' : int(a['bbox'][0]),
                'ymin' : int(a['bbox'][1]),
                'xmax' : int(abs(int(a['bbox'][0])+int(a['bbox'][2]))),
                'ymax' : int(abs(int(a['bbox'][1])+int(a['bbox'][3])))
            }
            if (imagesLen-count) < testSize:
                testDF = testDF.append(data, ignore_index=True)
            else:
                trainDF = trainDF.append(data, ignore_index=True)
        count+=1


testDF.to_csv('C:/Users/aryan/Desktop/objectDetection/data/test_labels.csv', index=False)
trainDF.to_csv('C:/Users/aryan/Desktop/objectDetection/data/train_labels.csv', index=False)