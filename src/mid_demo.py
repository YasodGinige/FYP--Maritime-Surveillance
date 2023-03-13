# from _future_ import absolute_import
# from _future_ import division
# from _future_ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector
import math


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    detector = Detector(opt)
    feedViaWifi = False

    if feedViaWifi:
        cap = cv2.VideoCapture("rtsp://192.168.8.175:8554/ir")
    else:
        cap = cv2.VideoCapture("rtsp://169.254.219.121:8554/ir")

    out = None
    out_name = opt.demo[opt.demo.rfind('/') + 1:]
    if opt.save_video:
      if not os.path.exists('../results_mid_demo'):
          os.mkdir('../results_mid_demo')
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      out = cv2.VideoWriter('../results_mid_demo/{}'.format(
        opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
          opt.input_w, opt.input_h))
  
    if opt.debug < 5:
      detector.pause = False

    cnt = 0
    results = {}
    while True:
        rectangle, img = cap.read()

        if opt.resize_video:
          img = cv2.resize(img, (opt.input_w, opt.input_h))

        ret = detector.run(img)

        time_str = 'frame {} |'.format(cnt)
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)

        results[cnt] = ret['results']

        if opt.save_video:
          cv2.imshow('image', ret['generic'])
          out.write(ret['generic'])
          # cv2.imwrite('../results_mid_demo/demo{}.jpg'.format(cnt), ret['generic'])

        cnt += 1
        if cv2.waitKey(1) == 27:
          save_and_exit(opt, out, results, out_name)
    return 


def save_and_exit(opt, out=None, results=None, out_name=''):
  if opt.save_results and (results is not None):
    save_dir =  '../results_mid_demo/{}results.json'.format(opt.exp_id + '' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)