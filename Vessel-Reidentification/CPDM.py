import os
import numpy as np
from PIL import Image


class CPDM:
    def __init__(self, mask_root='./PartAttMask/image_query'):
        self.mask_root = mask_root

    def get_area_ratios(self, image_name):
        image = os.path.join(self.mask_root, image_name)
        front = Image.open(image.replace('.jpg', '_front.jpg'))
        front_area = np.sum(np.array(front) / 255)
        rear = Image.open(image.replace('.jpg', '_rear.jpg'))
        rear_area = np.sum(np.array(rear) / 255)
        side = Image.open(image.replace('.jpg', '_side.jpg'))
        side_area = np.sum(np.array(side) / 255)
        global_area = front_area + rear_area + side_area
        front_area /= global_area
        rear_area /= global_area
        side_area /= global_area
        global_area /= global_area
        # print('global: {} \nfront: {} \nrear:{} \nside: {}'.format(global_area, front_area, rear_area, side_area))
        area_ratios = np.array([global_area, front_area, rear_area, side_area])
        return area_ratios

    def cooccurence_attention(self, image_1, image_2):
        image1_area_ratios = self.get_area_ratios(image_1)
        image2_area_ratios = self.get_area_ratios(image_2)
        # print(image1_area_ratios, image2_area_ratios)
        cam = image1_area_ratios*image2_area_ratios
        # print(cam)
        normalized_cam = cam/np.sum(cam)
        # print(normalized_cam)
        # print(np.sum(normalized_cam))
        return normalized_cam
