import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from torchsummary import summary
#from tensorflow.keras import regularizers
#from logit_mappings import CosFace
from logit_mappings import ArcFace
import sys
import pandas as pd
import model as reidmodels
import gc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def cosine_similarity(v1, v2):
    v1=v1.detach().to('cpu').numpy()
    v2=v2.detach().to('cpu').numpy()
    dot_product = np.dot(v1, np.transpose(v2))
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def calc_euclidean(x1, x1_area_ratio, x2, x2_area_ratio):
    global_feature_size = 256
    part_feature_size = 128
    cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
    normalized_cam = cam / np.sum(cam)
    normalized_cam = torch.from_numpy(normalized_cam).float().to(device)
    
    #distance=torch.cdist(x1, x2, p=2.0) #euclidean
    # distance = cosine_similarity(x1, x2) #cosine_similarity
    # distance = torch.sum(abs(x1 - x2)).item() #manhatton       
    distance = (x1 - x2).pow(2)  # athal ekata wadi kala
    distance[:global_feature_size] = distance[:global_feature_size] * normalized_cam[0:1]
    distance[global_feature_size: global_feature_size + part_feature_size] = distance[
                                                                              global_feature_size: global_feature_size + part_feature_size] * normalized_cam[
                                                                                                                                              1:2]
    distance[global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] = distance[
                                                                                                      global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] * normalized_cam[
                                                                                                                                                                                             2:3]
    distance[global_feature_size + 2 * part_feature_size:] = distance[
                                                             global_feature_size + 2 * part_feature_size:] * normalized_cam[
                                                                                                             3:]

    #weighted_distance = [global_distance, front_distance, rear_distance, side_distance]
    return torch.sum(distance)
    #return distance


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
    #features=features.detach().to('cpu').numpy()

    y_bar=np.array([y_])
    arcface=ArcFace(n_classes=nb_classes, s=30.0, m=0.80,mode='eval',num_features=384)
    #output = CosFace(nb_classes, regularizer=regularizers.l2(weight_decay))([features_2, y_bar])
    #output = ArcFace_tf(nb_classes,mode='evaluate')([features, y_bar])
    logits_front,logits_side,logits_rare = arcface(features,area_ratios)
    return logits_front,logits_side,logits_rare

def compare(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_image, query_image_id, gallery_image,
            gallery_image_id,NB_classes):
    img_transform = transforms.Compose([transforms.Resize([192, 192]), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize([24, 24]), transforms.ToTensor()])

    imageQueryT = img_transform(Image.open(query_dir + '/' + query_image_id + '/' + query_image).convert('RGB'))
    imageQuery = torch.unsqueeze(imageQueryT, 0)
    frontQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_image[:-4] + '_front.jpg'))
    frontQuery = torch.unsqueeze(frontQueryT, 0)
    rearQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_image[:-4] + '_rear.jpg'))
    rearQuery = torch.unsqueeze(rearQueryT, 0)
    sideQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_image[:-4] + '_side.jpg'))
    sideQuery = torch.unsqueeze(sideQueryT, 0)

    imageGalleryT = img_transform(Image.open(gallery_dir + '/' + gallery_image_id + '/' + gallery_image).convert('RGB'))
    imageGallery = torch.unsqueeze(imageGalleryT, 0)
    frontGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_image[:-4] + '_front.jpg'))
    frontGallery = torch.unsqueeze(frontGalleryT, 0)
    rearGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_image[:-4] + '_rear.jpg'))
    rearGallery = torch.unsqueeze(rearGalleryT, 0)
    sideGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_image[:-4] + '_side.jpg'))
    sideGallery = torch.unsqueeze(sideGalleryT, 0)

    # print(imageQuery.shape)
    query_img_features = model(imageQuery.to(device))
    gallery_img_features = model(imageGallery.to(device))

    del imageQuery,frontQuery,rearQuery,sideQuery,imageGallery,frontGallery,rearGallery,sideGallery
    ################################################################################################

    # query_img_features=calc_cosface(int(query_image_id[1:]), query_img_features,NB_classes)
    # gallery_img_features=calc_cosface(int(gallery_image_id[1:]), gallery_img_features,NB_classes)

    # query_img_features_bar=query_img_features.detach().to('cpu').numpy()
    # gallery_img_features_bar=gallery_img_features.detach().to('cpu').numpy()
    # del query_img_features,gallery_img_features
    # query_img_features=torch.from_numpy(query_img_features_bar)
    # gallery_img_features=torch.from_numpy(gallery_img_features_bar)
    ################################################################################################

    query_area_ratios = get_area_ratios(query_image, query_mask_dir)
    gallery_area_ratios = get_area_ratios(gallery_image,gallery_mask_dir)

    # weighted_distance = calc_euclidean(np.array([2,3,3,4,6]), query_area_ratios, np.array([1,4,5,7,8]), gallery_area_ratios)
    #
    weighted_distance = calc_euclidean(query_img_features, query_area_ratios, gallery_img_features, gallery_area_ratios)
    # print(weighted_distance)
    # num1 = random.randint(0, 99)
    return weighted_distance


def compare_arcface(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_image, query_image_id, gallery_image,
            gallery_image_id,NB_classes):
    img_transform = transforms.Compose([transforms.Resize([192, 192]), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize([24, 24]), transforms.ToTensor()])

    imageQueryT = img_transform(Image.open(query_dir + '/' + query_image_id + '/' + query_image).convert('RGB'))
    imageQuery = torch.unsqueeze(imageQueryT, 0)
    frontQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_image[:-4] + '_front.jpg'))
    #frontQuery = torch.unsqueeze(frontQueryT, 0)
    #rearQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_image[:-4] + '_rear.jpg'))
    #rearQuery = torch.unsqueeze(rearQueryT, 0)
    #sideQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_image[:-4] + '_side.jpg'))
    #sideQuery = torch.unsqueeze(sideQueryT, 0)

    imageGalleryT = img_transform(Image.open(gallery_dir + '/' + gallery_image_id + '/' + gallery_image).convert('RGB'))
    imageGallery = torch.unsqueeze(imageGalleryT, 0)
    #frontGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_image[:-4] + '_front.jpg'))
    #frontGallery = torch.unsqueeze(frontGalleryT, 0)
    #rearGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_image[:-4] + '_rear.jpg'))
    #rearGallery = torch.unsqueeze(rearGalleryT, 0)
    #sideGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_image[:-4] + '_side.jpg'))
    #sideGallery = torch.unsqueeze(sideGalleryT, 0)

    # print(imageQuery.shape)
    query_img_features = model(imageQuery.to(device))
    gallery_img_features = model(imageGallery.to(device))
    del imageQuery,imageGallery

    query_area_ratios = get_area_ratios(query_image, query_mask_dir)
    gallery_area_ratios = get_area_ratios(gallery_image,gallery_mask_dir)
    
    ################################################################################################

    g_img_f_front,g_img_f_side,g_img_f_rare=calc_cosface(int(gallery_image_id[1:]), gallery_img_features,NB_classes,gallery_area_ratios)
    q_img_f_front,q_img_f_side,q_img_f_rare=calc_cosface(int(gallery_image_id[1:]), query_img_features,NB_classes,query_area_ratios)
    del gallery_image_id,gallery_img_features

    #query_img_features=calc_cosface(int(query_image_id[1:]), query_img_features,NB_classes)
    # query_img_features_bar=query_img_features.detach().to('cpu').numpy()
    # gallery_img_features_bar=gallery_img_features.detach().to('cpu').numpy()
    # del query_img_features,gallery_img_features
    # query_img_features=torch.from_numpy(query_img_features_bar)
    # gallery_img_features=torch.from_numpy(gallery_img_features_bar)
    ################################################################################################

    # weighted_distance = calc_euclidean(np.array([2,3,3,4,6]), query_area_ratios, np.array([1,4,5,7,8]), gallery_area_ratios)
    #
    weighted_distance_front = calc_euclidean(q_img_f_front, query_area_ratios, g_img_f_front, gallery_area_ratios)
    weighted_distance_side = calc_euclidean(q_img_f_side, query_area_ratios, g_img_f_side, gallery_area_ratios)
    weighted_distance_rare = calc_euclidean(q_img_f_rare, query_area_ratios, g_img_f_rare, gallery_area_ratios)
    del g_img_f_front, g_img_f_side, g_img_f_rare, q_img_f_front, q_img_f_side, q_img_f_rare

    torch.cuda.empty_cache()
    gc.collect()

    cam = np.array(query_area_ratios) * np.array(gallery_area_ratios)
    normalized_cam = cam / np.sum(cam)

    return weighted_distance_front*normalized_cam[1] + weighted_distance_side*normalized_cam[2] + weighted_distance_rare*normalized_cam[3]


def accuracy(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images, query_images_ids, gallery_images, gallery_images_ids,NB_classes):
    global sel_key_list
    sel_key_list=[]
    final_accuracy1 = 0
    final_accuracy5 = 0
    final_accuracy10 = 0
    pbar = tqdm(total=len(query_images))
    
    ID=[]
    label=[]
    before_1=[]
    sortedlist=[]
    before_2=[]
    after_2=[]
    before_3=[]
    after_3=[]
    key_bef=[]
    key_aff=[]

    for i in range(len(query_images)):
        ID.append(query_images[i])
        label.append(query_images_ids[i])
        # print("step" + str(i+1))
        distances = {}
        for j in range(len(gallery_images)):
            # print(j)
            weighted_distance_1 = compare(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images[i], query_images_ids[i], gallery_images[j],
                                        gallery_images_ids[j],NB_classes)
            #weighted_distance_2 = compare_arcface(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images[i], query_images_ids[i], gallery_images[j],
                                        #gallery_images_ids[j],NB_classes)
            distances[gallery_images_ids[j] + gallery_images[j]] = weighted_distance_1
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])

        ######
        keys = []
        Values = []
        # print(sorted_distances[0])
        for q in sorted_distances:
            keys.append(q[0][:3])
            Values.append(q[1])

        before_1.append(Values[0].item())
        before_2.append(Values[1].item())
        before_3.append(Values[2].item())
        key_bef.append(keys[:3])
        # print(Keys)
        # Values = tormodel = model.Dino_VIT16()ch.FloatTensor(Values)

        # mean = torch.mean(Values)
        # std = torch.std(Values)
        # #print((Values-mean)/std)
        # print(torch.nn.functional.softmax(Values))
        ######

        # keys = []
        
        # for k in sorted_distances:
        #     keys.append(k[0][:3])
        #
        # del sorted_distances,distances
        # sel_keys=keys[:5]
        # # sel_keys = list(sel_keys)[:5]
        # sel_key_list.append(sel_keys)
        
        
        # distances = {}
        # for j in range(len(gallery_images)):
        #     if gallery_images_ids[j] in sel_keys:
        #         #print(gallery_images_ids[j])
        #         weighted_distance = compare_arcface(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images[i], query_images_ids[i], gallery_images[j],
        #                                     gallery_images_ids[j],NB_classes)
        #         distances[gallery_images_ids[j] + gallery_images[j]] = weighted_distance

        # # # query_area_ratios = get_area_ratios(query_images[i], query_mask_dir)
        # # # distances = {}
        # # # for j in sel_keys:
        # # #     temp_dist=0
        # # #     count=0
        # # #     for l in range(len(gallery_images)):
        # # #         if gallery_images_ids[l] == j:
        # # #             gallery_area_ratios = get_area_ratios(gallery_images[l],gallery_mask_dir)
        # # #             temp_dist+=compare(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images[i], query_images_ids[i], gallery_images[l],
        # # #                                        gallery_images_ids[l],NB_classes)*np.sum(query_area_ratios*gallery_area_ratios)
        # # #             count+=1
        # # #     distances[j]=temp_dist/count

        # sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        # #
        # keys = []
        # Values = []
        # # print(sorted_distances[0])
        # for q in sorted_distances:
        #     keys.append(q[0][:3])
        #     Values.append(q[1])

        # after_1.append(Values[0].item())
        # after_2.append(Values[1].item())
        # after_3.append(Values[2].item())
        # key_aff.append(keys[:3])


        correct_instances1 = keys[:1].count(query_images_ids[i])
        # temp = keys[:10].count(query_images_ids[i])
        # print(temp)
        if keys[:5].count(query_images_ids[i]) >= 1:
            correct_instances5 = 1
        else:
            correct_instances5 = 0
        # if keys[:10].count(query_images_ids[i]) >= 1:
        #     correct_instances10 = 1
        # else:
        #     correct_instances10 = 0
        final_accuracy1 += correct_instances1
        final_accuracy5 += correct_instances5
        #final_accuracy10 += correct_instances10
        pbar.update(1)
        sortedlist.append(Values)

        del sorted_distances,keys,Values

    pbar.close()

    np_array=np.transpose(np.array([ID,label,sortedlist,key_bef]))
    pd.DataFrame(np_array).to_csv("effect_of_arcface_distance_comparison.csv")

    return final_accuracy1 / len(query_images), final_accuracy5 / len(query_images), 0 #, final_accuracy10 / len(query_images)


def mAP(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images, query_images_ids, num_of_ids, gallery_images, gallery_images_ids,NB_classes):
    total = 0
    pbar = tqdm(total=len(query_images))
    for i in range(len(query_images)):
        distances = {}
        for j in range(len(gallery_images)):
            weighted_distance = compare(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images[i], query_images_ids[i], gallery_images[j], gallery_images_ids[j],NB_classes)
            distances[gallery_images_ids[j] + gallery_images[j]] = weighted_distance
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        instances = []
        for k in sorted_distances:
            instances.append(k[0][:3])
        correct_instances = 0
        precision_total = 0
        for x in range(1, len(instances) + 1):
            if correct_instances == num_of_ids[i]:
                break
            elif instances[x - 1] == query_images_ids[i]:
                correct_instances += 1
                precision_total += correct_instances / x
        total += precision_total / num_of_ids[i]
        pbar.update(1)
        del distances,sorted_distances
    pbar.close()
    return total / len(query_images)


def reid_evaluation(root_dir, mask_dir,NB_classes=23,reid_model_path="/home/fyp3/Desktop/Batch18/Re_ID/RGB_data/temp.pth"):

    global model
    #model = torch.load(reid_model_path)
    model = reidmodels.Dino_VIT16()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    # if torch.cuda.is_available():
    #     model.cuda()
    #     model=torch.nn.DataParallel(model, device_ids=[0, 1])

    query_dir = root_dir + "/query"
    gallery_dir = root_dir + "/gallery"
    query_mask_dir = mask_dir + "/query"
    gallery_mask_dir = mask_dir + "/gallery"
    query_images = []
    query_images_ids = []
    gallery_images = []
    gallery_images_ids = []
    num_of_ids = []

    for root, query_dirs, query_images_names in os.walk(query_dir, topdown=True):
        if len(query_images_names) != 0:
            for i in range(len(query_images_names)):
                if query_images_names[i][-3:] == 'jpg':
                    query_images.append(query_images_names[i])
                query_images_ids.append(root[-3:])

    for root, gallery_dirs, gallery_images_names in os.walk(gallery_dir, topdown=True):
        if len(gallery_images_names) != 0:
            for i in range(len(gallery_images_names)):
                if gallery_images_names[i][-3:] == 'jpg':
                    gallery_images.append(gallery_images_names[i])
                gallery_images_ids.append(root[-3:])
            num_of_ids.append(len(gallery_images_names))
    print("### Calculating Accuracy ###")
    top1, top5, top10 = accuracy(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images, query_images_ids, gallery_images,
                                 gallery_images_ids,NB_classes)
    print("### Calculating MAP ###")
    MAP = mAP(query_dir, query_mask_dir, gallery_dir, gallery_mask_dir, query_images, query_images_ids, num_of_ids, gallery_images, gallery_images_ids,NB_classes)

    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (top1, top5, top10, MAP))
