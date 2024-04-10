import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as F
import sys

csv_path = './test_images/identities_train/train_data.csv'

train_data = pd.read_csv(csv_path)


class ContrastStretch(nn.Module):
    def forward(self, image):
        # Get minimum and maximum pixel values
        image = F.to_tensor(image)
        min_val = torch.min(image)
        max_val = torch.max(image)

        # Perform contrast stretching
        stretched = (image - min_val) * (1.0 / (max_val - min_val))
        stretched_image = F.to_pil_image(stretched)
        return stretched_image


class ImageMasksTriplet(Dataset):
    def __init__(self, df, image_path, mask_path, train=True):

        self.data_csv = df
        self.is_train = train
        # self.img_transform = transforms.Compose([transforms.Resize([192, 192]), transforms.ToTensor()])
        # self.mask_transform = transforms.Compose([transforms.Resize([24, 24]), transforms.ToTensor()])
        transform1 = T.Resize(size=(192, 192))
        transform2 = T.Resize(size=(24, 24))
        stretch= ContrastStretch()
        self.img_transform = transforms.Compose([stretch, transform1, transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transform2, transforms.ToTensor()])
        self.img_path = image_path
        self.mask_path = mask_path
        if self.is_train:
            self.images = df['filename'].values
            self.labels = df['id'].values
            self.area_ratios = df['area_ratios']
            # self.ops = df['ops'].values
            self.index = df.index.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_image_name = self.images[item]
        anchor_image_path = self.img_path + '/' + self.labels[item] + '/' + anchor_image_name

        anchor_img = self.img_transform(Image.open(anchor_image_path).convert('RGB'))
        anchor_label = self.labels[item]

        target = int(anchor_label)

        anchor_area_ratios = np.array(self.area_ratios[item])
        anchor_image_masks = self.get_masks(anchor_image_name)
        if self.is_train:
            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
            positive_item = random.choice(positive_list)
            positive_label = self.labels[positive_item]
            positive_image_name = self.images[positive_item]
            positive_image_path = self.img_path + '/' + self.labels[positive_item] + '/' + positive_image_name
            positive_img = self.img_transform(Image.open(positive_image_path).convert('RGB'))
            positive_img_masks = self.get_masks(positive_image_name)
            positive_area_ratios = np.array(self.area_ratios[positive_item])

            # negative_list=[]
            # negative_list_int = self.index[self.index != item]#[self.labels[self.index != item] in self.ops[item] ]
            # for i in negative_list_int:
            #     if self.labels[i] in self.ops[item]:
            #         negative_list.append(i)
            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_label = self.labels[negative_item]
            negative_image_name = self.images[negative_item]
            negative_image_path = self.img_path + '/' + self.labels[negative_item] + '/' + negative_image_name
            negative_img = self.img_transform(Image.open(negative_image_path).convert('RGB'))
            negative_img_masks = self.get_masks(negative_image_name)
            negative_area_ratios = np.array(self.area_ratios[negative_item])

            return anchor_img, anchor_image_masks, anchor_area_ratios, positive_img, positive_img_masks, \
                positive_area_ratios, negative_img, negative_img_masks, negative_area_ratios, target, positive_label, negative_label
        else:
            return anchor_img, anchor_image_masks, anchor_area_ratios, target

    def get_masks(self, image_name):
        front_mask = self.mask_transform(Image.open(self.mask_path + '/' + image_name.replace('.jpg', '_front.jpg')))
        rear_mask = self.mask_transform(Image.open(self.mask_path + '/' + image_name.replace('.jpg', '_rear.jpg')))
        side_mask = self.mask_transform(Image.open(self.mask_path + '/' + image_name.replace('.jpg', '_side.jpg')))
        return front_mask, rear_mask, side_mask


if __name__ == '__main__':
    mask_path = './PartAttMask/image_train'
    csv_path = 'test_images/identities_train/train_data.csv'
    train_data_path = './test_images/identities_train'

    train_data = pd.read_csv(csv_path)
    names = train_data['filename']
    train_data['area_ratios'] = train_data[['global', 'front', 'rear', 'side']].values.tolist()

    print(train_data.index.dtype)
