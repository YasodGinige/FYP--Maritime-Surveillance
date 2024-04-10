import os
import random
import shutil

import numpy as np
import torch
from torch.backends import cudnn

import PartAttGen
import BGRemove_DL
# from visualize import visualize
import model


def pipeline_span(Or_image_root, mask_dl_ckpt, part_att_ckpt, target_dir):

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_list = os.listdir(Or_image_root)
    temp_dataset_root = Or_image_root + '/temp_data'
    if not os.path.isdir(temp_dataset_root):
        os.mkdir(temp_dataset_root)


    for i in f_list:
        shutil.copytree(Or_image_root + '/' + i, temp_dataset_root + '/' + i)

    image_root = temp_dataset_root

    Folder_list = os.listdir(image_root)
    for k in Folder_list:

        for (root, dirs, files) in os.walk(image_root + '/' + k, topdown=True):
            if len(dirs) == 0:
                for i in files:
                    shutil.copy(root + '/' + i, image_root + '/' + k + '/' + i)
                shutil.rmtree(root)

    tar_dir_name = target_dir + '/attention_masks_gen'
    if not os.path.isdir(tar_dir_name):
        os.mkdir(tar_dir_name)

    part_att_root = tar_dir_name

    print("\n### STEP 3 : Generate foreground mask by deep generator ###")
    checkpoint = os.path.join(mask_dl_ckpt, '5.ckpt')
    BGRemove_DL.implement(image_root=image_root,
                              mask_root=part_att_root,
                              model=model.Foreground_Generator().to(device),
                              device=device,
                              checkpoint=checkpoint)

    print("\n### Generate part attention mask ###")
    checkpoint = os.path.join(part_att_ckpt, '10.ckpt')
    PartAttGen.implement(image_root=image_root,
                         mask_root=part_att_root,
                         model=model.PartAtt_Generator().to(device),
                         device=device,
                         checkpoint=checkpoint)

    shutil.rmtree(temp_dataset_root)

    return 0


if __name__ == '__main__':
    pipeline_span(Or_image_root="/home/fyp3-2/Desktop/BATCH18/ReID_check/Val_data/",
                  mask_dl_ckpt= "/home/fyp3-2/Desktop/BATCH18/FYP-SPAN/mask_dl_chckpt/",
                  part_att_ckpt="/home/fyp3-2/Desktop/BATCH18/Grayscale_Mask_Images/part_attention_chckpt/",
                  target_dir="/home/fyp3-2/Desktop/BATCH18/ReID_check/")