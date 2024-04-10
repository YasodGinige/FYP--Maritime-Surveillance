import argparse
import os
#
#import numpy as np
#import torch
#
#import data_preparation
#import train
## import evaluate
#import evaluate_old
#import evaluate_efficient
#
#if __name__ == '__main__':
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#    parser = argparse.ArgumentParser(description='Re-ID using SPAN',
#                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    parser.add_argument('--mode', required=True,
#                        help='Select training or implementation mode; option: ["train", "implement"]')
#    parser.add_argument('--dataset', required=True, help='path to Re-ID dataset')
#    parser.add_argument('--part_att_ckpt', required=True, help='path to part attention mask model checkpoint')
#    parser.add_argument('--mask_dl_ckpt', required=True, help='path to mask dl model checkpoint')
#    parser.add_argument('--mask_dir', required=True, help='path to target part attention mask dir')
#    parser.add_argument('--train_csv_path', required=True, help='path to csv file with train data')
#
#    args = parser.parse_args()
#
#    reid_model_path = "/home/fyp3/Desktop/Batch18/Re_ID/model_save_final_2024_02_09/Weligama_cp.pth"
#    NB_cls_train = 500
#    NB_cls_eval = 250
#
#    print("### Mask Preparation ###")
#    # data_preparation.pipeline_span(Or_image_root=args.dataset,
#    #                               mask_dl_ckpt=args.mask_dl_ckpt,
#    #                                part_att_ckpt=args.part_att_ckpt,
#    #                                 target_dir=args.mask_dir)
#
#    print("### ReID Train ###")
#    train_data_path = args.dataset + '/train'
#    train_mask_path = args.mask_dir + '/attention_masks_gen/train'
#    train.reid_train(csv_path_val="/import argparse
import os

import numpy as np
import torch

import data_preparation
import train
# import evaluate
import evaluate_old
import evaluate_efficient

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Re-ID using SPAN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', required=True,
                        help='Select training or implementation mode; option: ["train", "implement"]')
    parser.add_argument('--dataset', required=True, help='path to Re-ID dataset')
    parser.add_argument('--part_att_ckpt', required=True, help='path to part attention mask model checkpoint')
    parser.add_argument('--mask_dl_ckpt', required=True, help='path to mask dl model checkpoint')
    parser.add_argument('--mask_dir', required=True, help='path to target part attention mask dir')
    parser.add_argument('--train_csv_path', required=True, help='path to csv file with train data')

    args = parser.parse_args()

    reid_model_path = "/home/fyp3/Desktop/Batch18/Re_ID/Model_save/thermal.pth"
    NB_cls_train = 23
    NB_cls_eval = 22

    print("### Mask Preparation ###")
    # data_preparation.pipeline_span(Or_image_root=args.dataset,
    #                               mask_dl_ckpt=args.mask_dl_ckpt,
    #                                part_att_ckpt=args.part_att_ckpt,
    #                                 target_dir=args.mask_dir)

    print("### ReID Train ###")
    train_data_path = args.dataset + '/train'
    train_mask_path = args.mask_dir + '/attention_masks_gen/train'
    train.reid_train(csv_path_val="/home/fyp3/Desktop/Batch18/Re_ID/Weligama_data/final_results_thermal_dataset/valid_data.csv",
                     csv_path_train=args.train_csv_path,
                     train_data_path=train_data_path, mask_path_train=train_mask_path,
                     mask_path_val=args.mask_dir + '/attention_masks_gen/valid',
                     val_data_path=args.dataset + '/valid', NB_classes=NB_cls_train, reid_model_path=reid_model_path,
                     af='enable')

    print('### Evaluate ###')
    train_mask_path = args.mask_dir + '/attention_masks_gen'
    evaluate_efficient.reid_evaluation(
        csv_path_query="/home/fyp3/Desktop/Batch18/Re_ID/Weligama_data/final_results_thermal_dataset/query_data.csv",
        csv_path_gallery="/home/fyp3/Desktop/Batch18/Re_ID/Weligama_data/final_results_thermal_dataset/gallery_data.csv", root_dir=args.dataset,
        mask_dir=args.mask_dir + '/attention_masks_gen', NB_classes=NB_cls_eval, reid_model_path=reid_model_path,
        af='enable')

    # evaluate.reid_evaluation(root_dir=args.dataset, mask_dir=train_mask_path)
    # evaluate_old.reid_evaluation( root_dir=args.dataset, mask_dir=train_mask_path,NB_classes=NB_cls,reid_model_path=reid_model_path)valid_data.csv",