#import tensorflow as tf
#from tensorflow.keras import backend as K
#from tensorflow.keras.layers import Layer
#from tensorflow.keras import regularizers
import numpy as np

import torch
import torch.nn as nn
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
#class ArcFace(nn.Module):
#    def __init__(self, n_classes=23, s=30.0, m=0.50,mode='train',num_features=640):
#        super(ArcFace, self).__init__()
#        self.n_classes = n_classes
#        self.s = s
#        self.mode=mode
#        self.m = m
#        self.num_features=num_features
#        self.W_front = nn.Parameter(torch.Tensor(self.n_classes, self.num_features))
#        self.W_side = nn.Parameter(torch.Tensor(self.n_classes, self.num_features))
#        self.W_rear = nn.Parameter(torch.Tensor(self.n_classes, self.num_features))
#        nn.init.xavier_uniform_(self.W_front)
#        nn.init.xavier_uniform_(self.W_side)
#        nn.init.xavier_uniform_(self.W_rear)
#
#    def calc_loss(self,logits,y):
#        y = torch.nn.functional.one_hot(y, num_classes=self.n_classes).to('cuda')
#        logits = logits * (1 - y) + self.s * y
#            
#            # add margin
#        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
#        target_logits = torch.cos(theta + self.m)
#        logits = logits * (1 - y) + target_logits * y
#            
#        out = torch.nn.functional.softmax(logits, dim=-1)
#        loss = torch.nn.functional.cross_entropy(out, torch.argmax(y,dim=-1))
#
#        return loss
#
#    def forward(self, x, area_ratios,y=None):
#        # normalize feature
#
#        x = nn.functional.normalize(x, dim=1).to('cuda')
#        #x.to('cuda')
#        #area_ratios=area_ratios.to('cuda')
#        # normalize weights
#        if self.mode=='train':
#            W_front = nn.functional.normalize(self.W_front, dim=0).to('cuda')
#            W_side = nn.functional.normalize(self.W_side, dim=0).to('cuda')
#            W_rear = nn.functional.normalize(self.W_rear, dim=0).to('cuda')
#        else:
#            W_front = torch.load('./mapping_weights/ArcFace_weights_front.pt').to('cuda')
#            W_side = torch.load('./mapping_weights/ArcFace_weights_side.pt').to('cuda')
#            W_rear = torch.load('./mapping_weights/ArcFace_weights_rear.pt').to('cuda')
#        # dot product
#
#        logits_front = torch.matmul(x, torch.transpose(W_front, 0, 1))
#        logits_side = torch.matmul(x, torch.transpose(W_side, 0, 1))
#        logits_rare = torch.matmul(x, torch.transpose(W_rear, 0, 1))
#        
#        if y is not None:
#            # cross-entropy loss
#            loss_front=self.calc_loss(self,logits_front,y)
#            loss_side=self.calc_loss(self,logits_side,y)
#            loss_rare=self.calc_loss(self,logits_rare,y)
#            
#            loss=loss_front*area_ratios[:,1]+loss_side*area_ratios[:,2]+loss_rare*area_ratios[:,3]
#            return loss
#        else:
#            # vector comparison
#            return logits_front,logits_side,logits_rare
class ArcFace(nn.Module):
    def __init__(self, n_classes=575, s=30.0, m=0.80, mode='train', num_features=600):
        super(ArcFace, self).__init__()
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

                
          self.W_front_mapping = torch.load('./mapping_weights/Mapping_weights_front.pt', map_location=torch.device('cpu') ).to(device)
          self.W_side_mapping = torch.load('./mapping_weights/Mapping_weights_side.pt', map_location=torch.device('cpu') ).to(device)
          self.W_rear_mapping = torch.load('./mapping_weights/Mapping_weights_rear.pt', map_location=torch.device('cpu') ).to(device)
          self.W_global_mapping = torch.load('./mapping_weights/Mapping_weights_global.pt', map_location=torch.device('cpu') ).to(device)
          
          
          self.W_front = torch.load('./mapping_weights/ArcFace_weights_front.pt', map_location=torch.device('cpu') ).to(device)
          self.W_side = torch.load('./mapping_weights/ArcFace_weights_side.pt', map_location=torch.device('cpu') ).to(device)
          self.W_rear = torch.load('./mapping_weights/ArcFace_weights_rear.pt', map_location=torch.device('cpu') ).to(device)
          self.W_global = torch.load('./mapping_weights/ArcFace_weights_global.pt', map_location=torch.device('cpu') ).to(device)
          

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
        
        
#        area_ratios = area_ratios.to(device)
        # normalize weights
        x= x.to(device)
        ## To map the Dino features to four different spaces
        
        
        front_features = torch.matmul(x, torch.transpose(self.W_side_mapping, 0, 1))
        rear_features = torch.matmul(x, torch.transpose(self.W_rear_mapping, 0, 1))
        side_features = torch.matmul(x, torch.transpose(self.W_side_mapping, 0, 1))
        
        
        # normalize feature
        front_features = nn.functional.normalize(front_features, dim=1).to(device)
        rear_features = nn.functional.normalize(rear_features, dim=1).to(device)
        side_features = nn.functional.normalize(side_features, dim=1).to(device)

    
        W_front = nn.functional.normalize(self.W_front, dim=0).to(device)
        W_side = nn.functional.normalize(self.W_side, dim=0).to(device)
        W_rear = nn.functional.normalize(self.W_rear, dim=0).to(device)
       
           
        # dot product

        logits_front = torch.matmul(front_features, torch.transpose(self.W_front, 0, 1))
        logits_side = torch.matmul(side_features, torch.transpose(self.W_side, 0, 1))
        logits_rear = torch.matmul(rear_features, torch.transpose(self.W_rear, 0, 1))

        if y is not None:
            # cross-entropy loss
            logits_front = self.calc_loss(logits_front, y).detach().to('cpu').numpy()
            logits_side = self.calc_loss(logits_side, y).detach().to('cpu').numpy()
            logits_rear = self.calc_loss(logits_rear, y).detach().to('cpu').numpy()

            # area_ratios=area_ratios.detach().to('cpu').numpy()
            # arcface_vector=area_ratios[:,1,np.newaxis]*loss_front+area_ratios[:,2,np.newaxis]*loss_side+area_ratios[:,3,np.newaxis]*loss_rare
            return logits_front, logits_side, logits_rear
        else:
            # vector comparison
            return logits_front, logits_side, logits_rear

# class SphereFace(Layer):
#     def __init__(self, n_classes=10, s=30.0, m=1.35, regularizer=None, **kwargs):
#         super(SphereFace, self).__init__(**kwargs)
#         self.n_classes = n_classes
#         self.s = s
#         self.m = m
#         self.regularizer = regularizers.get(regularizer)

#     def build(self, input_shape):
#         super(SphereFace, self).build(input_shape[0])
#         self.W = self.add_weight(name='W',
#                                  shape=(input_shape[0][-1], self.n_classes),
#                                  initializer='glorot_uniform',
#                                  trainable=True,
#                                  regularizer=self.regularizer)

#     def call(self, inputs):
#         x, y = inputs
#         c = K.shape(x)[-1]
#         # normalize feature
#         x = tf.nn.l2_normalize(x, axis=1)
#         # normalize weights
#         W = tf.nn.l2_normalize(self.W, axis=0)
#         # dot product
#         logits = x @ W
#         # add margin
#         # clip logits to prevent zero division when backward
#         theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
#         target_logits = tf.cos(self.m * theta)
#         #
#         logits = logits * (1 - y) + target_logits * y
#         # feature re-scale
#         logits *= self.s
#         out = tf.nn.softmax(logits)

#         return out

#     def compute_output_shape(self, input_shape):
#         return (None, self.n_classes)


# class CosFace(Layer):
#     def __init__(self, n_classes=23, s=30.0, m=0.1, regularizer=None, **kwargs):
#         super(CosFace, self).__init__(**kwargs)
#         self.n_classes = n_classes
#         self.s = s
#         self.m = m
#         self.regularizer = regularizers.get(regularizer)

#     def build(self, input_shape):
#         super(CosFace, self).build(input_shape[0])
#         self.W = self.add_weight(name='W',
#                                  shape=(input_shape[0][-1], self.n_classes),
#                                  initializer='glorot_uniform',
#                                  trainable=True,
#                                  regularizer=self.regularizer)

#     def call(self, inputs):
#         n_classes = 23
#         x, y = inputs
#         # x=x_.detach().to('cpu').numpy()
#         y = tf.one_hot(y, n_classes)
#         c = K.shape(x)[-1]
#         # normalize feature
#         x = tf.nn.l2_normalize(x, axis=1)
#         # normalize weights
#         W = tf.nn.l2_normalize(self.W, axis=0)
#         # dot product
#         logits = x @ W
#         # add margin
#         target_logits = logits - self.m
#         #
#         # print('#################',logits.shape,target_logits.shape,y.shape)
#         logits = logits * (1 - y) + target_logits * y
#         # feature re-scale

#         logits *= self.s
#         out = tf.nn.softmax(logits)
#         return logits

#     def compute_output_shape(self, input_shape):
#         return (None, self.n_classes)
