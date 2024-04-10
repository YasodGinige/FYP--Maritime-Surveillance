import os
from CPDM import CPDM
import csv


def create_folder_name(name):
    folder_name = '_'.join(name.split('_')[:-1])
    return folder_name


# def create_csv_with_area_ratios(image_path, mask_path, csv_path, save_path):
#     #creating folders with car name as identity
#     if not os.path.isdir(save_path):
#         os.mkdir(save_path)
#     for root, dirs, files in os.walk(image_path, topdown=True):
#         for name in files:
#             if not name[-3:] == 'jpg':
#                 continue
#             folder_name = create_folder_name(name)
#             src_path = paths[0] + '/' + name
#             src_label_path = paths[1] + '/' + name[:-3] + 'txt'
#             dst_path = save_path + '/' + folder_name
#             if not os.path.isdir(dst_path):
#                 os.mkdir(dst_path)
#             save_grayscale(src_path, dst_path + '/' + name[:-3] + 'jpg')

#data_type = 'train'
#data_type = 'valid'
#data_type = 'query'
data_type = 'gallery'


def create_csv_with_area_ratios(image_path, mask_path, csv_path):
    first = True
    with open(csv_path + '/'+ data_type + '_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'id', 'global', 'front', 'rear', 'side'])
        no_of_identities = 0
        for root, dirs, files in os.walk(image_path, topdown=True):
            if first:
                first = False
                no_of_identities = len(dirs)
                continue
            else:
                for image_name in files:
                    cpdm = CPDM(mask_root=mask_path)
                    area_ratios = cpdm.get_area_ratios(image_name=image_name)
                    # area_ratios_csv = ','.join(area_ratios)
                    digits = len(str(no_of_identities))
                    ID = root[-digits:]
                    if ID[0] == '/':
                      ID = ID[1:]
                    elif ID[1] == '/':
                      ID = ID[2:]
                    writer.writerow([image_name, ID, area_ratios[0], area_ratios[1], area_ratios[2], area_ratios[3]])


if __name__ == '__main__':
    create_csv_with_area_ratios(image_path='/home/fyp3/Desktop/Batch18/Re_ID/vehicleID_data/'+ data_type, mask_path='/home/fyp3/Desktop/Batch18/Re_ID/Weligama_data/masks/attention_masks_gen/'+ data_type,
                                csv_path='/home/fyp3/Desktop/Batch18/Re_ID/vehicleID_results/')

    # csv_path = 'test_images/identities_train/train_data.csv'
