from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import copy

from ..generic_dataset import GenericDataset

class SMD(GenericDataset):
  default_resolution = [544, 960]
  num_categories = 3
  class_name = [
    'person','boat','ship']
  _valid_ids = [
      1, 2, 3]
  cat_ids = {v: i + 1 for i, v in enumerate(_valid_ids)}
  num_joints = 17
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
           [4, 6], [3, 5], [5, 6], 
           [5, 7], [7, 9], [6, 8], [8, 10], 
           [6, 12], [5, 11], [11, 12], 
           [12, 14], [14, 16], [11, 13], [13, 15]]
  max_objs = 128
  def __init__(self, opt, split):
    super(SMD, self).__init__()
    # load annotations
    data_dir = os.path.join(opt.data_dir, 'smd')
    img_dir = os.path.join(data_dir, 'SMD_{}'.format(split))
    #if opt.trainval:
    #  split = 'test'
    #  ann_path = os.path.join(
    #      data_dir, 'annotations', 
    #      'image_info_test-dev2017.json')
    #else:
    ann_path = os.path.join(
          data_dir, 'annotations', 
          'SMD_{}.json').format(split)

    self.images = None
    # load image list and coco
    super(SMD, self).__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      if type(all_bboxes[image_id]) != type({}):
        # newest format
        for j in range(len(all_bboxes[image_id])):
          item = all_bboxes[image_id][j]
          cat_id = item['class'] - 1
          category_id = self._valid_ids[cat_id]
          bbox = item['bbox']
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(item['score']))
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results_smd.json'.format(save_dir), 'w'))
  
  def _save_results(self, records, fpath):
    with open(fpath,'w') as fid:
      for record in records:
        line = json.dumps(record)+'\n'
        fid.write(line)
    return fpath
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results_smd.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()