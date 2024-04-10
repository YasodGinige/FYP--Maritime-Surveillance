import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from logit_mappings import CosFace

import model as reidmodels
from ImageMasksDataset import ImageMasksTriplet
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TripletLossWithCPDM(nn.Module):
    def __init__(self, margin=1.0, global_feature_size=1024, part_feature_size=512):
        super(TripletLossWithCPDM, self).__init__()
        self.margin = margin
        self.global_feature_size = global_feature_size
        self.part_feature_size = part_feature_size

    def calc_distance_vector(self, x1, x1_area_ratio, x2, x2_area_ratio):
        cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
        normalized_cam = cam / np.sum(cam, axis=1, keepdims=True)
        normalized_cam = torch.from_numpy(normalized_cam).float().to(device)
        weighted_distance=torch.cdist(x1, x2, p=2.0)
        # distance = (x1 - x2).pow(2)
        # global_distance = distance[:, :self.global_feature_size] * normalized_cam[:, 0:1]
        # front_distance = distance[:,
        #                  self.global_feature_size: self.global_feature_size + self.part_feature_size] * normalized_cam[
        #                                                                                                 :, 1:2]
        # rear_distance = distance[:,
        #                 self.global_feature_size + self.part_feature_size: self.global_feature_size + 2 * self.part_feature_size] * normalized_cam[
        #                                                                                                                             :,
        #                                                                                                                             2:3]
        # side_distance = distance[:, self.global_feature_size + 2 * self.part_feature_size:] * normalized_cam[:, 3:]

        # weighted_distance = torch.cat((global_distance, front_distance, rear_distance, side_distance), 1).sum(1)
        return weighted_distance

    def forward(self, anchor, anchor_area_ratio, positive, positive_area_ratio, negative, negative_area_ratio):
        distance_positive = self.calc_distance_vector(anchor, anchor_area_ratio, positive, positive_area_ratio)
        distance_negative = self.calc_distance_vector(anchor, anchor_area_ratio, negative, negative_area_ratio)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


def calc_cosface(y, features,nb_classes):
    weight_decay = 1e-4
    features_2=features.detach().to('cpu').numpy()
    y_bar=y.detach().to('cpu').numpy()
    output = CosFace(nb_classes, regularizer=regularizers.l2(weight_decay))([features_2, y_bar])
    return output

def reid_train(csv_path_train, csv_path_val,train_data_path,val_data_path, mask_path_train,mask_path_val, NB_classes,model=reidmodels):
    # transform = T.Compose([T.Resize([192, 192]),
    #                        T.ToTensor(),
    #                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    types_dict = {'filename': str, 'id': str, 'global': float, 'front': float, 'rear': float, 'side': float}
    dataframe_train = pd.read_csv(csv_path_train, dtype=types_dict)
    dataframe_train['area_ratios'] = dataframe_train[['global', 'front', 'rear', 'side']].values.tolist()

    dataframe_val = pd.read_csv(csv_path_val, dtype=types_dict)                                                  #val
    dataframe_val['area_ratios'] = dataframe_val[['global', 'front', 'rear', 'side']].values.tolist()            #val

    dataset_val = ImageMasksTriplet(df=dataframe_val, image_path=val_data_path, mask_path=mask_path_val)
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=1)
   

    dataset_train = ImageMasksTriplet(df=dataframe_train, image_path=train_data_path, mask_path=mask_path_train)
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=1)

    
    # dataloader_train = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=2,
    #                         # persistent_workers=True)

    classifier = model.BoatIDClassifier(num_of_classes=NB_classes)
    model = model.Second_Stage_Extractor()

    if torch.cuda.is_available():
        model.cuda()
        classifier.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.1)
    epoch = 30

    triplet_loss_bucket = []
    CE_loss_bucket = []
    total_loss_bucket = []
    triplet_loss_bucket_val = []
    CE_loss_bucket_val = []
    total_loss_bucket_val = []
    for ep in range(epoch):
        model.train()
        print('\nStarting epoch %d / %d :' % (ep + 1, epoch))
        pbar = tqdm(total=len(dataloader_train))
        for batch_idx, data in enumerate(dataloader_train):
            anchor_img, anchor_image_masks, anchor_area_ratios, positive_img, \
                positive_img_masks, positive_area_ratios, negative_img, negative_img_masks, \
                negative_area_ratios, target = data
            
            val_data=next(iter(dataloader_val))
            anchor_img_val, anchor_image_masks_val, anchor_area_ratios_val, positive_img_val, \
                positive_img_masks_val, positive_area_ratios_val, negative_img_val, negative_img_masks_val, \
                negative_area_ratios_val, target_val = val_data                                                     #val

            anchor_img_features = model(anchor_img.to(device), anchor_image_masks[0].to(device),
                                        anchor_image_masks[1].to(device), anchor_image_masks[2].to(device))
            positive_img_features = model(positive_img.to(device), positive_img_masks[0].to(device),
                                          positive_img_masks[1].to(device), positive_img_masks[2].to(device))
            negative_img_features = model(negative_img.to(device), negative_img_masks[0].to(device),
                                          negative_img_masks[1].to(device), negative_img_masks[2].to(device))

            
            anchor_img_features_val = model(anchor_img_val.to(device), anchor_image_masks_val[0].to(device),
                                        anchor_image_masks_val[1].to(device), anchor_image_masks_val[2].to(device))
            positive_img_features_val = model(positive_img_val.to(device), positive_img_masks_val[0].to(device),
                                          positive_img_masks_val[1].to(device), positive_img_masks_val[2].to(device))
            negative_img_features_val = model(negative_img_val.to(device), negative_img_masks_val[0].to(device),
                                          negative_img_masks_val[1].to(device), negative_img_masks_val[2].to(device))         #val


            prediction = classifier(anchor_img_features)
            prediction_val = classifier(anchor_img_features_val)                                                            #val
            
            prediction=calc_cosface(target, prediction,NB_classes)
            prediction_val=calc_cosface(target_val, prediction_val,NB_classes)


            prediction_bar=prediction.numpy()
            prediction_val_bar=prediction_val.numpy()
            del prediction,prediction_val
            prediction=torch.from_numpy(prediction_bar)
            prediction_val=torch.from_numpy(prediction_val_bar)

            criterion1 = nn.CrossEntropyLoss()
            criterion2 = TripletLossWithCPDM()

            cross_entropy_loss = criterion1(prediction.to(device), target.to(device))
            triplet_loss = criterion2(anchor_img_features, anchor_area_ratios, positive_img_features,
                                      positive_area_ratios, negative_img_features, negative_area_ratios)

            cross_entropy_loss_val = criterion1(prediction_val.to(device), target_val.to(device))
            triplet_loss_val = criterion2(anchor_img_features_val, anchor_area_ratios_val, positive_img_features_val,
                                      positive_area_ratios_val, negative_img_features_val, negative_area_ratios_val)           #val

            lambda_ID = 1
            lambda_triplet = 1
            loss = lambda_ID * cross_entropy_loss + lambda_triplet * triplet_loss
            loss_val = lambda_ID * cross_entropy_loss_val + lambda_triplet * triplet_loss_val                                #val

            triplet_loss_bucket.append(triplet_loss)
            CE_loss_bucket.append(cross_entropy_loss)
            total_loss_bucket.append(loss)

            triplet_loss_bucket_val.append(triplet_loss_val)
            CE_loss_bucket_val.append(cross_entropy_loss_val)
            total_loss_bucket_val.append(loss_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Triplet_loss_train': ' {0:1.6f}'.format(triplet_loss / len(data)),
                              'ID_loss_train': ' {0:1.6f}'.format(cross_entropy_loss / len(data)),
                              'Triplet_loss_val': ' {0:1.6f}'.format(triplet_loss_val / len(val_data)),
                              'ID_loss_val': ' {0:1.6f}'.format(cross_entropy_loss_val / len(val_data))})
            pbar.update(1)
        pbar.close()
    torch.save(model, "/home/fyp3/Desktop/Batch18/Re_ID/RGB_data//temp.pth")
