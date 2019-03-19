# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import ReverseLayerF

import os

from deeplabv3 import DeepLabV3

class DeepLab_Plus(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeepLab_Plus, self).__init__()
        
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.network=DeepLabV3(self.model_id, self.project_dir)
        
        self.network.load_state_dict(torch.load("pretrained_models/model_13_2_2_2_epoch_580.pth",map_location='cpu'))
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=1296, out_features=2)

    def forward(self, x,alpha):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]
      
        feature_map = self.network.get_feature(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
        
        output = self.network(x) # (shape: (batch_size, num_classes, h/16, w/16))
       

        reverse_feature = ReverseLayerF.apply(feature_map, alpha)
        print(reverse_feature.shape)
        temp = self.conv1(reverse_feature)
        #
        # Your code here
        print(temp.shape)
        temp=F.max_pool2d(temp,(7,7))
        print(temp.shape)
        temp = temp.view(temp.size(0), -1)
        print(temp.shape)
        temp=self.fc(temp)
        print(temp.shape)
        temp=F.log_softmax(temp, dim=1)
        print(temp.shape)
        return output, temp

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
