import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchsummary import summary

from logit_mappings import ArcFace
import sys
import random
from torch.utils.data import DataLoader
import pandas as pd
import json 
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms as T
import model as reidmodels


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

arcface = ArcFace(n_classes=575,m=0.50,mode='evaluate')



class ImageMasks(Dataset):
    def __init__(self, df, image_path, mask_path, train=True):

        self.data_csv = df
        self.is_train = train
        #self.img_transform = transforms.Compose([transforms.Resize([192, 192]), transforms.ToTensor()])
        #self.mask_transform = transforms.Compose([transforms.Resize([24, 24]), transforms.ToTensor()])
        transform1 = T.Resize(size = (192,192))
        transform2 = T.Resize(size = (24,24))
        self.img_transform = transforms.Compose([transform1, transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transform2, transforms.ToTensor()])
        self.img_path = image_path
        self.mask_path = mask_path
        
        self.images = df['filename'].values
        self.labels = df['id'].values
        self.area_ratios = df['area_ratios']
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
        
        return anchor_image_name,anchor_img, anchor_image_masks, anchor_area_ratios, target

    def get_masks(self, image_name):
        front_mask = self.mask_transform(Image.open(self.mask_path + '/' + image_name.replace('.jpg', '_front.jpg')))
        rear_mask = self.mask_transform(Image.open(self.mask_path + '/' + image_name.replace('.jpg', '_rear.jpg')))
        side_mask = self.mask_transform(Image.open(self.mask_path + '/' + image_name.replace('.jpg', '_side.jpg')))
        return front_mask, rear_mask, side_mask



def cosine_similarity(v1, v2):
    v1=v1.detach().to('cpu').numpy()
    v2=v2.detach().to('cpu').numpy()
    dot_product = np.dot(v1, np.transpose(v2))
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def calc_euclidean(x1, x1_area_ratio, x2, x2_area_ratio):
#    global_feature_size = 256
#    part_feature_size = 128
    cam = x1_area_ratio.detach().to('cpu').numpy() * x2_area_ratio.detach().to('cpu').numpy()
    normalized_cam = cam / np.sum(cam)
    normalized_cam = torch.from_numpy(normalized_cam).float().to(device)

    #distance= 1 - (torch.nn.functional.cosine_similarity(x1, x2,dim=1))
    distance=torch.cdist(x1, x2, p=2.0) #euclidean
    # distance = cosine_similarity(x1, x2) #cosine_similarity
    # distance = torch.sum(abs(x1 - x2)).item() #manhatton
    # distance = (x1 - x2).pow(2).to(device)  # athal ekata wadi kala
    # distance[:global_feature_size] = distance[:global_feature_size]
    # distance[global_feature_size: global_feature_size + part_feature_size] = distance[
    #                                                                          global_feature_size: global_feature_size + part_feature_size]
    # distance[global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] = distance[
    #                                                                                                  global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size]
    # distance[global_feature_size + 2 * part_feature_size:] = distance[
    #                                                          global_feature_size + 2 * part_feature_size:]

    # weighted_distance = [global_distance, front_distance, rear_distance, side_distance]
    return torch.sum(distance)


def get_area_ratios(image_name, mask_root):
    image = os.path.join(mask_root, image_name)
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
    area_ratios = np.array([global_area, front_area, rear_area, side_area])
    return area_ratios

def calc_cosface(y_, features,nb_classes,area_ratios):
    weight_decay = 1e-4
    #features_2=features.detach().to('cpu').numpy()

    y_bar=np.array([y_])
    #output = CosFace(nb_classes, regularizer=regularizers.l2(weight_decay))([features_2, y_bar])
    logits_front,logits_side,logits_rare = arcface(torch.from_numpy(features),area_ratios)


    
    return logits_front,logits_side,logits_rare

def compare(query_img_features,gallery_img_features, query_area_ratio,gallery_area_ratio):


    # query_area_ratios = 1
    # gallery_area_ratios = 1
    # weighted_distance = calc_euclidean(np.array([2,3,3,4,6]), query_area_ratios, np.array([1,4,5,7,8]), gallery_area_ratios)
    weighted_distance = calc_euclidean(torch.from_numpy(query_img_features), torch.from_numpy(np.array(query_area_ratio)).to('cuda'), torch.from_numpy(gallery_img_features), torch.from_numpy(np.array(gallery_area_ratio)).to('cuda'))
    #weighted_distance = 1 - (torch.nn.functional.cosine_similarity(torch.from_numpy(query_img_features), torch.from_numpy(gallery_img_features), dim=1))
    #print(weighted_distance)
    return weighted_distance

def compare_arcface(query_img_features,gallery_img_features,NB_classes,gallery_image_id,query_area_ratio,gallery_area_ratio):
    
    # ################################################################################################

    # query_area_ratios = get_area_ratios(query_image, query_mask_dir)
    # gallery_area_ratios = get_area_ratios(gallery_image,gallery_mask_dir)
    cam = np.array(query_area_ratio)* np.array(gallery_area_ratio)
    normalized_cam = cam / np.sum(cam)
    normalized_cam = torch.from_numpy(normalized_cam).float().to(device)

    query_img_features = query_img_features.astype('float32')
    gallery_img_features = gallery_img_features.astype('float32')
    # Find the maximum value in each tensor
    max_query = max(query_area_ratio)
    max_gallery = max(gallery_area_ratio)
    
    # Set the maximum value to 1 and others to 0 (only for the first occurrence)
    query_area_ratio = [1 if x == max_query and not found_query else 0 for x, found_query in zip(query_area_ratio, [False] + [True] * (len(query_area_ratio) - 1))]
    gallery_area_ratio = [1 if x == max_gallery and not found_gallery else 0 for x, found_gallery in zip(gallery_area_ratio, [False] + [True] * (len(gallery_area_ratio) - 1))]
    q_im_f,q_im_s,q_im_r=calc_cosface(int(gallery_image_id), query_img_features,NB_classes,query_area_ratio)
    g_im_f,g_im_s,g_im_r=calc_cosface(int(gallery_image_id), gallery_img_features,NB_classes,gallery_area_ratio)

    del query_img_features,gallery_img_features

    # ################################################################################################

    query_area_ratios = 1
    gallery_area_ratios = 1
    # weighted_distance = calc_euclidean(np.array([2,3,3,4,6]), query_area_ratios, np.array([1,4,5,7,8]), gallery_area_ratios)
    #
    weighted_distance_f = 1 - (torch.nn.functional.cosine_similarity(q_im_f, g_im_f, dim=1))
    weighted_distance_s = 1 - (torch.nn.functional.cosine_similarity(q_im_s, g_im_s, dim=1))
    weighted_distance_r = 1 - (torch.nn.functional.cosine_similarity(q_im_r, g_im_r, dim=1))
    weighted_distance=weighted_distance_f*normalized_cam[1] + weighted_distance_r*normalized_cam[2] + weighted_distance_s*normalized_cam[3]

    del weighted_distance_f,weighted_distance_s,weighted_distance_r,q_im_f,q_im_s,q_im_r,g_im_f,g_im_s,g_im_r
    torch.cuda.empty_cache()

    return weighted_distance


def check_ratio(query_area_ratio,gallery_area_ratio):
    return query_area_ratio.index(max(query_area_ratio))==gallery_area_ratio.index(max(gallery_area_ratio))

def accuracy(query_features,gallery_features, query_images, query_images_ids,NB_classes,num_of_ids,af = 'enable'):
    
    final_accuracy1 = 0
    final_accuracy5 = 0
    final_accuracy10 = 0
    Tot = 0
    pbar = tqdm(total=len(query_images))
    for i in query_features.keys():
        query_img_ID,query_area_ratio,query_img_feature=query_features[i][0],query_features[i][1],np.array(query_features[i][2])
        query_img_feature=np.expand_dims(query_img_feature, 0)

        first_round_distances=[]
        first_round_IDs=[]

        for j in gallery_features.keys():
            gallery_img_ID,gallery_area_ratio,gallery_img_feature=gallery_features[j][0],gallery_features[j][1],np.array(gallery_features[j][2])
            Is_matching_ratio=check_ratio(query_area_ratio,gallery_area_ratio)
            if Is_matching_ratio:
                first_round_IDs.append(gallery_img_ID)
                gallery_img_feature=np.expand_dims(gallery_img_feature, 0)
                first_round_distances.append(compare(query_img_feature,gallery_img_feature,query_area_ratio,gallery_area_ratio))

        min_id=first_round_distances.index(min(first_round_distances))
        first_round_distances[min_id]=np.Inf
        ID1=first_round_IDs[min_id]
        min_id=first_round_distances.index(min(first_round_distances))
        first_round_distances[min_id]=np.Inf
        ID2=first_round_IDs[min_id]
        min_id=first_round_distances.index(min(first_round_distances))
        first_round_distances[min_id] = np.Inf
        ID3=first_round_IDs[min_id]
        min_id = first_round_distances.index(min(first_round_distances))
        first_round_distances[min_id] = np.Inf
        ID4 = first_round_IDs[min_id]
        min_id = first_round_distances.index(min(first_round_distances))
        first_round_distances[min_id] = np.Inf
        ID5 = first_round_IDs[min_id]

        second_round_distances={}
        
        for j in gallery_features.keys():
            gallery_img_ID,gallery_area_ratio,gallery_img_feature=gallery_features[j][0],gallery_features[j][1],gallery_features[j][2]
            if gallery_img_ID == ID1 or gallery_img_ID == ID2 or gallery_img_ID == ID3:
                gallery_img_feature=np.expand_dims(gallery_img_feature, 0)
                if af == 'disable':
                    second_round_distances[str(gallery_img_ID)+'_'+j] = compare(query_img_feature,gallery_img_feature,query_area_ratio,gallery_area_ratio)
                else:
                    second_round_distances[str(gallery_img_ID)+'_'+j]=compare_arcface(query_img_feature,gallery_img_feature, NB_classes, gallery_img_ID,query_area_ratio,gallery_area_ratio)

        sorted_distances = sorted(second_round_distances.items(), key=lambda x: x[1])

        instances = []
        for k in sorted_distances:
            instances.append(int(k[0].split('_')[0]))

        correct_instances = 0
        precision_total = 0
        for x in range(1, len(instances) + 1):
            if correct_instances == num_of_ids[query_img_ID]:
                break
            elif instances[x - 1] == query_img_ID:
                correct_instances += 1
                precision_total += correct_instances / x
        Tot += precision_total / num_of_ids[query_img_ID]

        keys = instances
        # for k in sorted_distances:
        #     keys.append(k[0])

        correct_instances1 = keys[:1].count(query_img_ID)
        
        if keys[:5].count(query_img_ID) >= 1:
            correct_instances5 = 1
        else:
            correct_instances5 = 0
        if keys[:10].count(query_img_ID) >= 1:
            correct_instances10 = 1
        else:
            correct_instances10 = 0
        final_accuracy1 += correct_instances1
        final_accuracy5 += correct_instances5
        final_accuracy10 += correct_instances10
        pbar.update(1)
    pbar.close()
    return final_accuracy1 / len(query_images), final_accuracy5 / len(query_images), final_accuracy10 / len(
        query_images), Tot / len(query_images)


# def mAP(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images, query_images_ids, num_of_ids, gallery_images, gallery_images_ids,NB_classes):
#     total = 0
#     pbar = tqdm(total=len(query_images))
#     for i in range(len(query_images)):
#         distances = {}
#         for j in range(len(gallery_images)):
#             weighted_distance = compare(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images[i], query_images_ids[i], gallery_images[j], gallery_images_ids[j],NB_classes)
#             distances[gallery_images_ids[j] + gallery_images[j]] = weighted_distance
#         sorted_distances = sorted(distances.items(), key=lambda x: x[1])
#         instances = []
#         for k in sorted_distances:
#             instances.append(k[0][:3])
#         correct_instances = 0
#         precision_total = 0
#         for x in range(1, len(instances) + 1):
#             if correct_instances == num_of_ids[i]:
#                 break
#             elif instances[x - 1] == query_images_ids[i]:
#                 correct_instances += 1
#                 precision_total += correct_instances / x
#         total += precision_total / num_of_ids[i]
#         pbar.update(1)
#     pbar.close()
#     return total / len(query_images)

def get_features(csv_path_query, csv_path_gallery, query_dir, gallery_dir, query_mask_dir, mask_path_gallery,
                 NB_classes, reid_model_path="/home/fyp3/Desktop/Batch18/Re_ID/RGB_data/temp.pth"):
    query_features = {}
    gallery_features = {}

    # model = torch.load(reid_model_path)
    # if torch.cuda.is_available():
    #     model.cuda()
    #     model = torch.nn.DataParallel(model, device_ids=[0, 1])

    types_dict = {'filename': str, 'id': str, 'global': float, 'front': float, 'rear': float, 'side': float}
    dataframe_train = pd.read_csv(csv_path_query, dtype=types_dict)
    dataframe_train['area_ratios'] = dataframe_train[['global', 'front', 'rear', 'side']].values.tolist()

    dataframe_val = pd.read_csv(csv_path_gallery, dtype=types_dict)  # val
    dataframe_val['area_ratios'] = dataframe_val[['global', 'front', 'rear', 'side']].values.tolist()  # val

    dataset_gallery = ImageMasks(df=dataframe_val, image_path=gallery_dir, mask_path=mask_path_gallery)
    dataloader_gal = DataLoader(dataset_gallery, batch_size=64, shuffle=False, num_workers=1, drop_last=True)

    dataset_query = ImageMasks(df=dataframe_train, image_path=query_dir, mask_path=query_mask_dir)
    dataloader_query = DataLoader(dataset_query, batch_size=22, shuffle=False, num_workers=1, drop_last=True)

    
    for batch_idx, data in enumerate(dataloader_query):
        query_image_name, query_img, query_image_masks, query_area_ratios, target = data

        query_img_features = model(query_img.to(device))


        for i in range(len(query_image_name)):
            query_features[query_image_name[i]] = [target[i].detach().to('cpu').numpy().tolist(), query_area_ratios[i].detach().to('cpu').numpy().tolist(), query_img_features[i].detach().to('cpu').numpy().tolist()]

    
    for batch_idx, data in enumerate(dataloader_gal):
        gallery_image_name, gallery_img, gallery_image_masks, gallery_area_ratios, target = data

        gallery_img_features = model(gallery_img.to(device))

        for i in range(len(gallery_image_name)):
            gallery_features[gallery_image_name[i]] = [target[i].detach().to('cpu').numpy().tolist(), gallery_area_ratios[i].detach().to('cpu').numpy().tolist(), gallery_img_features[i].detach().to('cpu').numpy().tolist()]


    with open("/home/fyp3/Desktop/Batch18/Re_ID/efficient_feature_gallery/query_features.json", "w") as outfile:
        json.dump(query_features, outfile)

    with open("/home/fyp3/Desktop/Batch18/Re_ID/efficient_feature_gallery/gallery_features.json", "w") as outfile2:
        json.dump(gallery_features, outfile2)


    return query_features, gallery_features

def reid_evaluation(csv_path_query, csv_path_gallery,root_dir,mask_dir,NB_classes=23,reid_model_path="/home/fyp3/Desktop/Batch18/Re_ID/RGB_data/temp.pth",af = 'enable'):

    global model
    # model = torch.load(reid_model_path)
    model = reidmodels.Dino_VIT16()
    if torch.cuda.is_available():
        model.to(device)
        #model=torch.nn.DataParallel(model, device_ids=[0, 1])

    query_dir = root_dir + "/query"
    gallery_dir = root_dir + "/gallery"
    query_mask_dir = mask_dir + "/query"
    gallery_mask_dir = mask_dir + "/gallery"
    query_images = []
    query_images_ids = []
    gallery_images = []
    gallery_images_ids = []
    num_of_ids = {}
    print(query_dir)

    for root, query_dirs, query_images_names in os.walk(query_dir, topdown=True):
        if len(query_images_names) != 0:
            for i in range(len(query_images_names)):
                if query_images_names[i].split('.')[-1] == 'jpg':
                    query_images.append(query_images_names[i])
                query_images_ids.append(root.split('/')[-1])

    for root, gallery_dirs, gallery_images_names in os.walk(gallery_dir, topdown=True):
        if len(gallery_images_names) != 0:
            for i in range(len(gallery_images_names)):
                if gallery_images_names[i].split('.')[-1] == 'jpg':
                    gallery_images.append(gallery_images_names[i])
                gallery_images_ids.append(root.split('/')[-1])
            num_of_ids[int(root.split('/')[-1])]=len(gallery_images_names)

    
    query_features,gallery_features=get_features(csv_path_query, csv_path_gallery,query_dir,gallery_dir, query_mask_dir,gallery_mask_dir, NB_classes,reid_model_path=reid_model_path)

    print("### Calculating Accuracy & MAP ###")

    
    top1, top5, top10, MAP = accuracy(query_features,gallery_features, query_images, query_images_ids,NB_classes,num_of_ids,af = af)

    # print("### Calculating MAP ###")
    # MAP = mAP(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images, query_images_ids, num_of_ids, gallery_images, gallery_images_ids,NB_classes)

    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (top1, top5, top10, MAP))





            
            

            
