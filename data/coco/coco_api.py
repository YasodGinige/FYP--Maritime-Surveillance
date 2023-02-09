from pycocotools.coco import COCO
import numpy as np
import json
import random
import os
import cv2

### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



dataDir='/home/fyp3/Desktop/Batch18/Obj_Track/TraDeS/data/coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# Initialize the COCO api for instance annotations
coco=COCO(annFile)

# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"


filterClasses = ['boat']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses) 
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))


for k in imgIds:

    img = coco.loadImgs(k)[0]
    print(img)
    I = cv2.imread('{}/{}/{}'.format(dataDir,dataType,img['file_name']))

    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

    mask = np.zeros((img['height'],img['width']))
    try:
        for i in range(len(anns)):
            mask = np.maximum(coco.annToMask(anns[i]), mask)
    except:
        continue

    mask = mask.astype("uint8")
    res = cv2.bitwise_and(I,I,mask = mask)
    cv2.imwrite('/home/fyp3/Desktop/Batch18/Obj_Track/TraDeS/data/coco/masked_boat_images/'+img['file_name'],res)
    