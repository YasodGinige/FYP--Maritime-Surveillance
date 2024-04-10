import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from logit_mappings import ArcFace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("/home/fyp3/Desktop/Batch18/Re_ID/model_save_yasod/thermal_test.pth")
if torch.cuda.is_available():
    model.cuda()

def cosine_similarity(v1, v2):
    v1 = v1.cpu().detach().numpy()
    v2 = v2.cpu().detach().numpy()
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    epsilon = 1*10**(-10)
    return dot_product / (norm_v1 * norm_v2 + epsilon)


def calc_euclidean(x1, x1_area_ratio, x2, x2_area_ratio):
    global_feature_size = 1024
    part_feature_size = 512
    cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
    normalized_cam = cam / np.sum(cam)
    normalized_cam = torch.from_numpy(normalized_cam).float().to(device)
    #distance = (x1 - x2).pow(2) #euclidean
    distance=torch.cdist(x1, x2, p=2.0)
    #dist=torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    #distance = cosine_similarity(x1, x2) #cosine_similarity
    #distance = abs(x1 - x2) #manhatton
    #distance = (x1 - x2).pow(4)  # athal ekata wadi kala
    # distance[:global_feature_size] = distance[:global_feature_size] * normalized_cam[0:1]
    # distance[global_feature_size: global_feature_size + part_feature_size] = distance[
    #                                                                           global_feature_size: global_feature_size + part_feature_size] * normalized_cam[
    #                                                                                                                                           1:2]
    # distance[global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] = distance[
    #                                                                                                   global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] * normalized_cam[
    #                                                                                                                                                                                          2:3]
    # distance[global_feature_size + 2 * part_feature_size:] = distance[
    #                                                          global_feature_size + 2 * part_feature_size:] * normalized_cam[
    #                                                                                                          3:]

    #weighted_distance = [global_distance, front_distance, rear_distance, side_distance]
    #return torch.sum(distance).item()
    return distance
    
def calc_cosine(x1, x1_area_ratio, x2, x2_area_ratio):
    global_feature_size = 1024
    part_feature_size = 512
    cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
    normalized_cam = cam / np.sum(cam)
    normalized_cam = torch.from_numpy(normalized_cam).float().to(device)
    
    x1[:global_feature_size] = x1[:global_feature_size] * normalized_cam[0:1]
    x1[global_feature_size: global_feature_size + part_feature_size] = x1[
                                                                             global_feature_size: global_feature_size + part_feature_size] * normalized_cam[
                                                                                                                                             1:2]
    x1[global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] = x1[
                                                                                                     global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] * normalized_cam[
                                                                                                                                                                                             2:3]
    x1[global_feature_size + 2 * part_feature_size:] = x1[
                                                             global_feature_size + 2 * part_feature_size:] * normalized_cam[
                                                                                                             3:]
                                                                                                             
    x2[:global_feature_size] = x2[:global_feature_size] * normalized_cam[0:1]
    x2[global_feature_size: global_feature_size + part_feature_size] = x2[
                                                                             global_feature_size: global_feature_size + part_feature_size] * normalized_cam[
                                                                                                                                             1:2]
    x2[global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] = x2[
                                                                                                     global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] * normalized_cam[
                                                                                                                                                                                             2:3]
    x2[global_feature_size + 2 * part_feature_size:] = x2[
                                                             global_feature_size + 2 * part_feature_size:] * normalized_cam[
                                                                                                             3:]
    distance = cosine_similarity(x1, x2) #cosine_similarity
    return -distance[0][0]


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



def compare(query_mask_dir, gallery_mask_dir, query_image, query_img_features, gallery_image, gallery_img_features):
    query_area_ratios = get_area_ratios(query_image, query_mask_dir)
    gallery_area_ratios = get_area_ratios(gallery_image, gallery_mask_dir)

    #weighted_distance = calc_euclidean(query_img_features, query_area_ratios, gallery_img_features, gallery_area_ratios)
    
    weighted_distance = calc_euclidean(query_img_features, query_area_ratios, gallery_img_features, gallery_area_ratios)
    return weighted_distance


def accuracy(query_mask_dir, gallery_mask_dir, query_images, query_images_ids, query_images_features, gallery_images, gallery_images_ids, gallery_images_features):

    final_accuracy1 = 0
    final_accuracy5 = 0
    final_accuracy10 = 0
    pbar = tqdm(total=len(query_images))
    for i in range(len(query_images)):

        distances = {}
        for j in range(len(gallery_images)):
            weighted_distance = compare(query_mask_dir, gallery_mask_dir, query_images[i], query_images_features[i], gallery_images[j], gallery_images_features[j])

            distances[gallery_images_ids[j] + gallery_images[j]] = weighted_distance
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        keys = []
        
        for k in sorted_distances:
            keys.append(k[0][:3])

        sel_keys=set(keys)
        sel_keys = list(sel_keys)[:5]
        print(sel_keys)

        correct_instances1 = keys[:1].count(query_images_ids[i])
        if keys[:5].count(query_images_ids[i]) >= 1:
            correct_instances5 = 1
        else:
            correct_instances5 = 0
        if keys[:10].count(query_images_ids[i]) >= 1:
            correct_instances10 = 1
        else:
            correct_instances10 = 0
        final_accuracy1 += correct_instances1
        final_accuracy5 += correct_instances5
        final_accuracy10 += correct_instances10
        pbar.update(1)
    pbar.close()

    return final_accuracy1 / len(query_images), final_accuracy5 / len(query_images), final_accuracy10 / len(query_images)


def mAP(query_mask_dir, gallery_mask_dir, query_images, query_images_ids, query_images_features, num_of_ids, gallery_images, gallery_images_ids, gallery_images_features):

    total = 0
    pbar = tqdm(total=len(query_images))
    for i in range(len(query_images)):
        distances = {}
        for j in range(len(gallery_images)):

            weighted_distance = compare(query_mask_dir, gallery_mask_dir, query_images[i], query_images_features[i], gallery_images[j], gallery_images_features[j])

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
    pbar.close()
    return total / len(query_images)


def reid_evaluation(root_dir, mask_dir):
    print(root_dir)
    query_dir = root_dir + "/query"
    gallery_dir = root_dir + "/gallery"
    query_mask_dir = mask_dir + "/query"
    gallery_mask_dir = mask_dir + "/gallery"
    query_images = []
    query_images_ids = []

    query_images_features = []
    gallery_images = []
    gallery_images_ids = []
    gallery_images_features = []
    num_of_ids = []
    img_transform = transforms.Compose([transforms.Resize([192, 192]), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize([24, 24]), transforms.ToTensor()])

    arcface=ArcFace(mode='evaluate')
    for root, query_dirs, query_images_names in os.walk(query_dir, topdown=True):
        if len(query_images_names) != 0:
            for i in range(len(query_images_names)):
                if query_images_names[i][-3:] == 'jpg':

                    imageQueryT = img_transform(Image.open(query_dir + '/' + root[-3:] + '/' + query_images_names[i]).convert('RGB'))
                    imageQuery = torch.unsqueeze(imageQueryT, 0)
                    frontQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_images_names[i][:-4] + '_front.jpg'))
                    frontQuery = torch.unsqueeze(frontQueryT, 0)
                    rearQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_images_names[i][:-4] + '_rear.jpg'))
                    rearQuery = torch.unsqueeze(rearQueryT, 0)
                    sideQueryT = mask_transform(Image.open(query_mask_dir + '/' + query_images_names[i][:-4] + '_side.jpg'))
                    sideQuery = torch.unsqueeze(sideQueryT, 0)
                    query_img_features = model(imageQuery.to(device), frontQuery.to(device), rearQuery.to(device), sideQuery.to(device))
                    # query_img_features=arcface(query_img_features)
                    # print(query_img_features.size())
                    query_images_features.append(query_img_features)
                    query_images.append(query_images_names[i])
                query_images_ids.append(root[-3:])
    

    for root, gallery_dirs, gallery_images_names in os.walk(gallery_dir, topdown=True):
        if len(gallery_images_names) != 0:
            for i in range(len(gallery_images_names)):
                if gallery_images_names[i][-3:] == 'jpg':
                    imageGalleryT = img_transform(Image.open(gallery_dir + '/' + root[-3:] + '/' + gallery_images_names[i]).convert('RGB'))
                    imageGallery = torch.unsqueeze(imageGalleryT, 0)
                    frontGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_images_names[i][:-4] + '_front.jpg'))
                    frontGallery = torch.unsqueeze(frontGalleryT, 0)
                    rearGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_images_names[i][:-4] + '_rear.jpg'))
                    rearGallery = torch.unsqueeze(rearGalleryT, 0)
                    sideGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_images_names[i][:-4] + '_side.jpg'))
                    sideGallery = torch.unsqueeze(sideGalleryT, 0)
                    gallery_img_features = model(imageGallery.to(device), frontGallery.to(device), rearGallery.to(device), sideGallery.to(device))
                    # gallery_img_features=arcface(gallery_img_features)
                    gallery_images_features.append(gallery_img_features)
                    gallery_images.append(gallery_images_names[i])
                gallery_images_ids.append(root[-3:])
            num_of_ids.append(len(gallery_images_names))


    print("### Calculating Accuracy ###")
    top1, top5, top10 = accuracy(query_mask_dir, gallery_mask_dir, query_images, query_images_ids, query_images_features, gallery_images, gallery_images_ids, gallery_images_features)
    print("### Calculating MAP ###")
    MAP = mAP(query_mask_dir, gallery_mask_dir, query_images, query_images_ids, query_images_features, num_of_ids, gallery_images, gallery_images_ids, gallery_images_features)

    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (top1, top5, top10, MAP))
