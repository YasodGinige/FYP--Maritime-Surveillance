import array
import numpy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from logit_mappings import ArcFace
import collections

import model as reidmodels
from ImageMasksDataset import ImageMasksTriplet
from torch.optim.lr_scheduler import StepLR
import sys

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


class TripletLossWithCPDM(nn.Module):
    def __init__(self, margin=1.0, global_feature_size=1024, part_feature_size=512):
        super(TripletLossWithCPDM, self).__init__()
        self.margin = margin
        self.global_feature_size = global_feature_size
        self.part_feature_size = part_feature_size

    def calc_distance_vector(self, x1, x1_area_ratio, x2, x2_area_ratio, lable_1, label_2, NB_Classes):
        try:
            cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
        except:
            cam = np.array(x1_area_ratio.detach().to('cpu').numpy()) * np.array(
                x2_area_ratio.detach().to('cpu').numpy())

        normalized_cam = cam  # / np.sum(cam, axis=1, keepdims=True)
        normalized_cam = torch.from_numpy(normalized_cam).float().to(device)

       
        weighted_distance = torch.cdist(x1, x2, p=2)
        return weighted_distance

    def forward(self, anchor, anchor_area_ratio, positive, positive_area_ratio, negative, negative_area_ratio,
                pos_label, neg_label, NB_Classes):
        distance_positive = self.calc_distance_vector(anchor, anchor_area_ratio, positive, positive_area_ratio,
                                                      pos_label, pos_label, NB_Classes)
        distance_negative = self.calc_distance_vector(anchor, anchor_area_ratio, negative, negative_area_ratio,
                                                      pos_label, neg_label, NB_Classes)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class ArcFaceLoss(nn.Module):
    def __init__(self, n_classes=22, s=30.0, m=0.80, mode='train', num_features=600):
        super(ArcFaceLoss, self).__init__()
        self.n_classes = n_classes
        self.s = s
        self.mode = mode
        self.m = m
        self.mapping_dimensions= 600
        self.dino_feature_size = 2400
        self.num_features = num_features
        
        if self.mode == 'train':
          self.W_front = nn.Parameter(torch.Tensor(self.n_classes, self.num_features)).to(device)
          self.W_side = nn.Parameter(torch.Tensor(self.n_classes, self.num_features)).to(device)
          self.W_rear = nn.Parameter(torch.Tensor(self.n_classes, self.num_features)).to(device)
          self.W_global = nn.Parameter(torch.Tensor(self.n_classes, self.num_features)).to(device)
          
          self.W_front_mapping = nn.Parameter(torch.Tensor(self.mapping_dimensions, self.dino_feature_size)).to(device)
          self.W_side_mapping = nn.Parameter(torch.Tensor(self.mapping_dimensions, self.dino_feature_size)).to(device)
          self.W_rear_mapping = nn.Parameter(torch.Tensor(self.mapping_dimensions,self.dino_feature_size)).to(device)
          self.W_global_mapping = nn.Parameter(torch.Tensor(self.mapping_dimensions, self.dino_feature_size)).to(device)
          
          nn.init.xavier_uniform_(self.W_front)
          nn.init.xavier_uniform_(self.W_side)
          nn.init.xavier_uniform_(self.W_rear)
          nn.init.xavier_uniform_(self.W_global)
          
          
          
          nn.init.xavier_uniform_(self.W_front_mapping)
          nn.init.xavier_uniform_(self.W_side_mapping)
          nn.init.xavier_uniform_(self.W_rear_mapping)
          nn.init.xavier_uniform_(self.W_global_mapping)
        else:

                
          self.W_front_mapping = torch.load('./mapping_weights/Mapping_weights_front.pt').to(device)
          self.W_side_mapping = torch.load('./mapping_weights/Mapping_weights_side.pt').to(device)
          self.W_rear_mapping = torch.load('./mapping_weights/Mapping_weights_rear.pt').to(device)
          self.W_global_mapping = torch.load('./mapping_weights/Mapping_weights_global.pt').to(device)
          
          
          self.W_front = torch.load('./mapping_weights/ArcFace_weights_front.pt').to(device)
          self.W_side = torch.load('./mapping_weights/ArcFace_weights_side.pt').to(device)
          self.W_rear = torch.load('./mapping_weights/ArcFace_weights_rear.pt').to(device)
          self.W_global = torch.load('./mapping_weights/ArcFace_weights_global.pt').to(device)
          

    def calc_loss(self, logits, y):
        y = torch.nn.functional.one_hot(y, num_classes=self.n_classes).to(device)
        logits = logits * (1 - y) + self.s * y

        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y

        # out = torch.nn.functional.softmax(logits, dim=-1)
        # loss = torch.nn.functional.cross_entropy(out, torch.argmax(y,dim=-1))

        return logits

    def forward(self, x, area_ratios, y=None):
        
        
        area_ratios = area_ratios.to(device)
        # normalize weights
        
        ## To map the Dino features to four different spaces
        global_features = torch.matmul(x, torch.transpose(self.W_front_mapping, 0, 1))
        front_features = torch.matmul(x, torch.transpose(self.W_side_mapping, 0, 1))
        rear_features = torch.matmul(x, torch.transpose(self.W_rear_mapping, 0, 1))
        side_features = torch.matmul(x, torch.transpose(self.W_global_mapping, 0, 1))
        
        
        # normalize feature
        front_features = nn.functional.normalize(front_features, dim=1).to(device)
        rear_features = nn.functional.normalize(rear_features, dim=1).to(device)
        side_features = nn.functional.normalize(side_features, dim=1).to(device)
        global_features = nn.functional.normalize(global_features, dim=1).to(device)
    
    
        W_front = nn.functional.normalize(self.W_front, dim=0).to(device)
        W_side = nn.functional.normalize(self.W_side, dim=0).to(device)
        W_rear = nn.functional.normalize(self.W_rear, dim=0).to(device)
        W_global = nn.functional.normalize(self.W_global, dim=0).to(device)

           
        # dot product

        logits_front = torch.matmul(front_features, torch.transpose(self.W_front, 0, 1))
        logits_side = torch.matmul(side_features, torch.transpose(self.W_side, 0, 1))
        logits_rear = torch.matmul(rear_features, torch.transpose(self.W_rear, 0, 1))
        logits_global = torch.matmul(global_features, torch.transpose(self.W_global, 0, 1))

        if y is not None:
            # cross-entropy loss
            logits_front = self.calc_loss(logits_front, y).detach().to('cpu').numpy()
            logits_side = self.calc_loss(logits_side, y).detach().to('cpu').numpy()
            logits_rear = self.calc_loss(logits_rear, y).detach().to('cpu').numpy()
            logits_global = self.calc_loss(logits_global, y).detach().to('cpu').numpy()

            # area_ratios=area_ratios.detach().to('cpu').numpy()
            # arcface_vector=area_ratios[:,1,np.newaxis]*loss_front+area_ratios[:,2,np.newaxis]*loss_side+area_ratios[:,3,np.newaxis]*loss_rare
            return logits_front, logits_side, logits_rear, logits_global
        else:
            # vector comparison
            return logits_front, logits_side, logits_rear, logits_global


def plot_losses(triplet_train_loss, triplet_val_loss, CE_train_loss, CE_val_loss, total_train_loss, total_val_loss,
                test_name, loss_save_path):
    # Create a new figure with multiple subplots
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # Set the space between the subplots
    fig.subplots_adjust(hspace=0.4)

    # Plot the training loss on the first subplot
    axs[0].set_title('Triplet Loss')
    axs[0].plot(triplet_train_loss, label='train')
    axs[0].plot(triplet_val_loss, label='valid')
    axs[0].legend()

    axs[1].set_title('Cross Entropy Loss')
    axs[1].plot(CE_train_loss, label='train')
    axs[1].plot(CE_val_loss, label='valid')
    axs[1].legend()

    axs[2].set_title('Total Loss')
    axs[2].plot(total_train_loss, label='train')
    axs[2].plot(total_val_loss, label='valid')
    axs[2].legend()

    # Set the x-axis label for all subplots
    for ax in axs.flat:
        ax.set(xlabel='Epoch')

    # Display the plot
    plt.savefig(loss_save_path + '/' + test_name + '.jpg', dpi=900)


def reid_train(csv_path_train, csv_path_val, train_data_path, val_data_path, mask_path_train, mask_path_val, NB_classes,
               reid_model_path, model=reidmodels,af ='disabled'):
    # transform = T.Compose([T.Resize([192, 192]),
    #                        T.ToTensor(),
    #                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    types_dict = {'filename': str, 'id': str, 'global': float, 'front': float, 'rear': float, 'side': float, 'ops': str}
    dataframe_train = pd.read_csv(csv_path_train, dtype=types_dict)
    dataframe_train['area_ratios'] = dataframe_train[['global', 'front', 'rear', 'side']].values.tolist()

    dataframe_val = pd.read_csv(csv_path_val, dtype=types_dict)  # val
    dataframe_val['area_ratios'] = dataframe_val[['global', 'front', 'rear', 'side']].values.tolist()  # val

    dataset_val = ImageMasksTriplet(df=dataframe_val, image_path=val_data_path, mask_path=mask_path_val)
    dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=2)

    dataset_train = ImageMasksTriplet(df=dataframe_train, image_path=train_data_path, mask_path=mask_path_train)
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=2)

    # dataloader_train = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=2,
    #                         # persistent_workers=True)

    classifier = model.BoatIDClassifier(input_size=NB_classes, num_of_classes=NB_classes)
    # model = model.Second_Stage_Extractor()
    model = model.Dino_VIT16()
    # model = model.MAE()

    if torch.cuda.is_available():
        model.to(device)
        classifier.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
    epoch = 20

    train_triplet_loss_mean = []
    valid_triplet_loss_mean = []
    train_CE_loss_mean = []
    valid_CE_loss_mean = []
    train_total_loss_mean = []
    valid_total_loss_mean = []
    #

    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = TripletLossWithCPDM(global_feature_size=256, part_feature_size=128)

    # model=torch.nn.Sequential(model,
    #                             torch.nn.Linear(384, 600, bias=True, device=device))

    model.train()

    # model_extend = torch.nn.Sequential(
    #     torch.nn.Linear(600, 3, bias=True, device=device),
    #     torch.nn.ReLU(inplace=False),
    #     torch.nn.LayerNorm([3]))
    # model_extend.to(device)
    arcface = ArcFaceLoss(n_classes=NB_classes, s=30.0, m=0.50, mode='train')
    
    for ep in range(epoch):

        print('\nStarting epoch %d / %d :' % (ep + 1, epoch))
        pbar = tqdm(total=len(dataloader_train))

        triplet_loss_bucket = []
        CE_loss_bucket = []
        total_loss_bucket = []
        triplet_loss_bucket_val = []
        CE_loss_bucket_val = []
        total_loss_bucket_val = []
        
        

        for batch_idx, data in enumerate(dataloader_train):
            anchor_img, anchor_image_masks, anchor_area_ratios, positive_img, \
                positive_img_masks, positive_area_ratios, negative_img, negative_img_masks, \
                negative_area_ratios, target, positive_label, negative_label = data

            positive_label = torch.from_numpy(np.asarray(positive_label, dtype=int))
            negative_label = torch.from_numpy(np.asarray(negative_label, dtype=int))

            val_data = next(iter(dataloader_val))
            anchor_img_val, anchor_image_masks_val, anchor_area_ratios_val, positive_img_val, \
                positive_img_masks_val, positive_area_ratios_val, negative_img_val, negative_img_masks_val, \
                negative_area_ratios_val, target_val, positive_label_val, negative_label_val = val_data  # val

            positive_label_val = torch.from_numpy(np.asarray(positive_label_val, dtype=int))
            negative_label_val = torch.from_numpy(np.asarray(negative_label_val, dtype=int))

            # train_features = model(anchor_img.to(device), anchor_image_masks[0].to(device),
            #                             anchor_image_masks[1].to(device), anchor_image_masks[2].to(device))

            # val_features = model(anchor_img_val.to(device), anchor_image_masks_val[0].to(device),
            #                             anchor_image_masks_val[1].to(device), anchor_image_masks_val[2].to(device))

            anchor_img_features = model(anchor_img.to(device))
            positive_img_features = model(positive_img.to(device))
            negative_img_features = model(negative_img.to(device))

            anchor_img_features_val = model(anchor_img_val.to(device))
            positive_img_features_val = model(positive_img_val.to(device))
            negative_img_features_val = model(negative_img_val.to(device))  # val

            #########################################################################################################

            # pred_area_ratios = model_extend(anchor_img_features.to(device))
            # pred_area_ratios_val = model_extend(anchor_img_features_val.to(device))

            # prediction = classifier(anchor_img_features)
            # prediction_val = classifier(anchor_img_features_val)  # val

            
            anchor_img_features_f, anchor_img_features_s, anchor_img_features_r, anchor_img_features_g = arcface(
                anchor_img_features, anchor_area_ratios, y=target)
            anchor_img_features_f_val, anchor_img_features_s_val, anchor_img_features_r_val, anchor_img_features_g_val = arcface(
                anchor_img_features_val, anchor_area_ratios_val, y=target_val)

            prediction_f = classifier(torch.from_numpy(anchor_img_features_f).to(device))
            prediction_s = classifier(torch.from_numpy(anchor_img_features_s).to(device))
            prediction_r = classifier(torch.from_numpy(anchor_img_features_r).to(device))
            prediction_g = classifier(torch.from_numpy(anchor_img_features_g).to(device))
            anchor_area_ratios = anchor_area_ratios.to(device)
            anchor_area_ratios_val = anchor_area_ratios_val.to(device)
            prediction = (prediction_f.T * anchor_area_ratios[:, 1] + prediction_r.T * anchor_area_ratios[:,
                                                                                       2] + prediction_s.T * anchor_area_ratios[
                                                                                                             :,
                                                                                                             3] + prediction_g.T).T
            # prediction = classifier(anchor_img_features.to(device))

            prediction_f_val = classifier(torch.from_numpy(anchor_img_features_f_val).to(device))
            prediction_s_val = classifier(torch.from_numpy(anchor_img_features_s_val).to(device))
            prediction_r_val = classifier(torch.from_numpy(anchor_img_features_r_val).to(device))
            prediction_g_val = classifier(torch.from_numpy(anchor_img_features_g_val).to(device))
            prediction_val = (
                    prediction_f_val.T * anchor_area_ratios_val[:, 1] + prediction_r_val.T * anchor_area_ratios_val[
                                                                                             :,
                                                                                             2] + prediction_s_val.T * anchor_area_ratios_val[
                                                                                                                       :,
                                                                                                                       3] + prediction_g_val.T).T  # val

            anchor_area_ratios = anchor_area_ratios.to(torch.float32)
            anchor_area_ratios_val = anchor_area_ratios_val.to(torch.float32)

            triplet_loss = criterion3(anchor_img_features, anchor_area_ratios, positive_img_features,
                                      positive_area_ratios, negative_img_features, negative_area_ratios, positive_label,
                                      negative_label, NB_classes) * 10

            triplet_loss_val = criterion3(anchor_img_features_val, anchor_area_ratios_val, positive_img_features_val,
                                          positive_area_ratios_val, negative_img_features_val, negative_area_ratios_val,
                                          positive_label_val, negative_label_val, NB_classes) * 10

            # MSE_loss = criterion2(pred_area_ratios.to(device), anchor_area_ratios[:, 1:].to(device))
            # MSE_loss_val = criterion2(pred_area_ratios_val.to(device), anchor_area_ratios_val[:, 1:].to(device))
            # sum_loss = torch.sum((1 - torch.sum(pred_area_ratios)) ** 2)
            # sum_loss_val = torch.sum((1 - torch.sum(pred_area_ratios_val)) ** 2)

            # AR_loss = MSE_loss + sum_loss
            # AR_loss_val = MSE_loss_val + sum_loss_val

            cross_entropy_loss = criterion1(prediction, target.to(device))
            # arcface_loss = criterion2(train_features, anchor_area_ratios, target)

            cross_entropy_loss_val = criterion1(prediction_val, target_val.to(device))
            # arcface_loss_val = criterion2(val_features, anchor_area_ratios_val, target_val)  # val

            lambda_ID = 10
            lambda_triplet = 10
            lambda_AR = 0.1

            # if ep<11:
            #     loss = lambda_AR * AR_loss
            #     loss_val = lambda_AR * AR_loss_val
            # else:
            loss = lambda_ID * cross_entropy_loss + lambda_triplet * triplet_loss # +lambda_arc*arcface_loss
            loss_val = lambda_ID * cross_entropy_loss_val + lambda_triplet * triplet_loss_val # +lambda_arc*arcface_loss_val # val

            triplet_loss_bucket.append(triplet_loss.item())
            CE_loss_bucket.append(cross_entropy_loss.item())
            total_loss_bucket.append(loss.item())

            triplet_loss_bucket_val.append(triplet_loss_val.item())
            CE_loss_bucket_val.append(cross_entropy_loss_val.item())
            total_loss_bucket_val.append(loss_val.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Tpt_l_t': ' {0:1.6f}'.format(triplet_loss / len(data)),
                              'ID_l_t': ' {0:1.6f}'.format(cross_entropy_loss / len(data)),
                              'Tpt_l_v': ' {0:1.6f}'.format(triplet_loss_val / len(val_data)),
                              'ID_l_v': ' {0:1.6f}'.format(cross_entropy_loss_val / len(val_data)),
                              })
            pbar.update(1)

        train_triplet_loss_mean.append(numpy.mean(triplet_loss_bucket))
        train_CE_loss_mean.append(numpy.mean(CE_loss_bucket))
        train_total_loss_mean.append(numpy.mean(total_loss_bucket))
        valid_triplet_loss_mean.append(numpy.mean(triplet_loss_bucket_val))
        valid_CE_loss_mean.append(numpy.mean(CE_loss_bucket_val))
        valid_total_loss_mean.append(numpy.mean(total_loss_bucket_val))
        pbar.close()
        scheduler.step()


    plot_losses(train_triplet_loss_mean, valid_triplet_loss_mean, train_CE_loss_mean, valid_CE_loss_mean,
                train_total_loss_mean, valid_total_loss_mean, "ArcFace3", '/home/fyp3/Desktop/Batch18/Re_ID/Model_save')

    torch.save(model, reid_model_path)
    torch.save(arcface.W_front, './mapping_weights/ArcFace_weights_front.pt')
    torch.save(arcface.W_side, './mapping_weights/ArcFace_weights_side.pt')
    torch.save(arcface.W_rear, './mapping_weights/ArcFace_weights_rear.pt')
    torch.save(arcface.W_global, './mapping_weights/ArcFace_weights_global.pt')
    
    torch.save(arcface.W_front_mapping, './mapping_weights/Mapping_weights_front.pt')
    torch.save(arcface.W_side_mapping, './mapping_weights/Mapping_weights_side.pt')
    torch.save(arcface.W_rear_mapping, './mapping_weights/Mapping_weights_rear.pt')
    torch.save(arcface.W_global_mapping, './mapping_weights/Mapping_weights_global.pt')

    # Identify the last layer
    # last_layer = model.fc_last  # Example: saving the last fully connected layer named 'fc'
    # torch.save(last_layer.state_dict(), '/home/fyp3/Desktop/Batch18/Re_ID/model_save_yasod/last_layer_weights.pth')
