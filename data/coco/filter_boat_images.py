import cv2
import json
import shutil
  

f = open("/home/fyp3/Desktop/Batch18/Obj_Track/TraDeS/data/coco/annotations-boat/instances_train2017.json")
data = json.load(f)
file_names=[]

for i in data['images']:
    file_names.append(i["file_name"])
f.close()

for i in file_names:
    shutil.copy("/home/fyp3/Desktop/Batch18/Obj_Track/TraDeS/data/coco/train2017/"+i,"/home/fyp3/Desktop/Batch18/Obj_Track/TraDeS/data/coco/boat_images/"+i)

print('Process completed')


