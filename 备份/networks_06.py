# encoding: utf-8
import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, resnet50
from torchvision.transforms import functional

from .resnet import ResNet

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)


class PBN(nn.Module):
    def __init__(self, input_dim, num_classes, do_reduction=False, num_reduction=512):
        super(PBN, self).__init__()
        self.do_reduction = do_reduction
        self.input_dim = input_dim
        self.num_reduction = num_reduction

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))

        self.reduction = nn.Linear(self.input_dim, self.num_reduction) # 别忘了赋予参数
        self.reduction.apply(weights_init_classifier)
        # self.reduction = reduction

        self.bn = nn.BatchNorm1d(self.num_reduction)

        self.classifier = nn.Linear(self.num_reduction, num_classes)
        self.classifier.apply(weights_init_classifier)
        # self.classifier = classifier

    def forward(self, x):
        # GAP + GMP
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x3 = x1 + x2
        x4 = torch.squeeze(x3)
        # Reduction
        if self.do_reduction:
            x5 = self.reduction(x4) 
        else:
            x5 = x4
            self.num_reduction = self.input_dim
        # BNNeck
        x6 = self.bn(x5)
        # Classification
        x7 = self.classifier(x6)
        return x5, x6, x7  # x5 for triplet; x6 for inference; x7 for softmax


class PBN_modify(nn.Module):
    def __init__(self, input_dim, num_classes, do_reduction=False, num_reduction=512):
        super(PBN_modify, self).__init__()
        self.do_reduction = do_reduction
        self.input_dim = input_dim
        self.num_reduction = num_reduction

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))

        # self.reduction1 = nn.Linear(self.input_dim, self.num_reduction) # 别忘了赋予参数
        # self.reduction1.apply(weights_init_kaiming)

        # self.reduction2 = nn.Linear(self.input_dim, self.num_reduction) # 别忘了赋予参数
        # self.reduction2.apply(weights_init_kaiming)
        # self.reduction = reduction
        reduction1 = []
        reduction1 += [nn.Conv2d(input_dim, num_reduction, kernel_size=1, bias=True)]
        reduction1 += [nn.BatchNorm2d(num_reduction)]
        reduction1 += [nn.LeakyReLU(0.1)]
        reduction1 = nn.Sequential(*reduction1)
        reduction1.apply(weights_init_kaiming)
        self.reduction1 = reduction1
        reduction2 = []
        reduction2 += [nn.Conv2d(input_dim, num_reduction, kernel_size=1, bias=True)]
        reduction2 += [nn.BatchNorm2d(num_reduction)]
        reduction2 += [nn.LeakyReLU(0.1)]
        reduction2 = nn.Sequential(*reduction2)
        reduction2.apply(weights_init_kaiming)
        self.reduction2 = reduction2        

        self.bn1 = nn.BatchNorm1d(self.num_reduction)
        self.bn2 = nn.BatchNorm1d(self.num_reduction)

        self.classifier1 = nn.Linear(self.num_reduction, num_classes)
        self.classifier1.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(self.num_reduction, num_classes)
        self.classifier2.apply(weights_init_classifier)
        # self.classifier = classifier

    def forward(self, x):
        # GAP + GMP
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        
        # ------ for x1 -------
        x1_1 = self.reduction1(x1) 
        x1_2 = torch.squeeze(x1_1)
        x1_3 = self.bn1(x1_2)
        x1_4 = self.classifier1(x1_3)

        # ------ for x2 -------
        x2_1 = self.reduction2(x2) 
        x2_2 = torch.squeeze(x2_1)
        x2_3 = self.bn2(x2_2)
        x2_4 = self.classifier2(x2_3)        

        return x1_2, x1_3, x1_4, x2_2, x2_3, x2_4  # x5 for triplet; x6 for inference; x7 for softmax

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x

class BatchCrop(nn.Module):
    def __init__(self, ratio):
        super(BatchCrop, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rw = int(self.ratio * w)
            start = random.randint(0, h-1)
            if start + rw > h:
                select = list(range(0, start+rw-h)) + list(range(start, h))
            else:
                select = list(range(start, start+rw))
            mask = x.new_zeros(x.size())
            mask[:, :, select, :] = 1
            x = x * mask
        return x

class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=None, last_stride=1, pretrained=False):
        super().__init__()
        self.base = ResNet(last_stride)
        if pretrained:
            model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            self.base.load_param(model_zoo.load_url(model_url))

        self.num_classes = num_classes
        if num_classes is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.in_planes, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5)
            )
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(512, self.num_classes)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.base(x)
        global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.training and self.num_classes is not None:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return [global_feat], [cls_score]
        else:
            return global_feat

    def get_optim_policy(self):
        base_param_group = self.base.parameters()
        if self.num_classes is not None:
            add_param_group = itertools.chain(self.bottleneck.parameters(), self.classifier.parameters())
            return [
                {'params': base_param_group},
                {'params': add_param_group}
            ]
        else:
            return [
                {'params': base_param_group}
            ]

class BFE(nn.Module):
    def __init__(self, num_classes, width_ratio=0.5, height_ratio=0.5):
        super(BFE, self).__init__()
        resnet = resnet50(pretrained=True)
        resnet.fc = nn.Sequential()
        self.model = resnet
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        # layer3 branch
        self.bottleneck1 = Bottleneck(1024, 256)
        self.PBN1 = PBN_modify(1024, num_classes, do_reduction=True, num_reduction=512)

        # global branch
        self.bottleneck_g = Bottleneck(2048, 512)
        self.PBN2 = PBN_modify(2048, num_classes, do_reduction=True, num_reduction=512) # 到这来

        # part1 branch
        self.bottleneck2 = Bottleneck(2048, 512)
        self.PBN3 = PBN_modify(2048, num_classes, do_reduction=True, num_reduction=256)

        # part2 branch
        self.bottleneck3 = Bottleneck(2048, 512)
        self.PBN4 = PBN_modify(2048, num_classes, do_reduction=True, num_reduction=256)

        # # part3 branch
        # self.bottleneck4 = Bottleneck(2048, 512)
        # self.PBN5 = PBN_modify(2048, num_classes, do_reduction=True, num_reduction=256)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x3 = self.model.layer3(x)
        x = self.model.layer4(x3)

        # --- layer3 分支 ---
        x3_1 = self.bottleneck1(x3)
        x3_2, x3_3, x3_4, x3_5, x3_6, x3_7 = self.PBN1(x3_1)

        # --- global 分支 ---
        x = self.bottleneck_g(x)
        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = self.PBN2(x)

        # --- divide x into two parts ---
        # _, _,height, weight = x.shape # [N, C, H, W]
        height = x.shape[2]
        x_p1 = x[:, :, :int(height/2), :]
        x_p2 = x[:, :, int(height/2):, :]
        # x_p3 = x[:, :, int(2*height/3):, :]

        # --- part1 分支 ---
        x_p1_1 = self.bottleneck2(x_p1)
        x_p1_2, x_p1_3, x_p1_4, x_p1_5, x_p1_6, x_p1_7 = self.PBN3(x_p1_1)

        # --- part2 分支 ---
        x_p2_1 = self.bottleneck3(x_p2)
        x_p2_2, x_p2_3, x_p2_4, x_p2_5, x_p2_6, x_p2_7 = self.PBN4(x_p2_1)

        # # --- part3 分支 ---
        # x_p3_1 = self.bottleneck4(x_p3)
        # x_p3_2, x_p3_3, x_p3_4, x_p3_5, x_p3_6, x_p3_7 = self.PBN5(x_p3_1)

        predict = []
        triplet_features = []
        softmax_features = []

        # add layer3 feature
        softmax_features.append(x3_4)
        triplet_features.append(x3_2)
        predict.append(x3_3)
        softmax_features.append(x3_7)
        triplet_features.append(x3_5)
        predict.append(x3_6)

       
       # add global feature
        softmax_features.append(x4_3)
        triplet_features.append(x4_1)
        predict.append(x4_2)
        softmax_features.append(x4_6)
        triplet_features.append(x4_3)
        predict.append(x4_5)      
       # add part1 feature
        softmax_features.append(x_p1_4)
        triplet_features.append(x_p1_2)
        predict.append(x_p1_3)
        softmax_features.append(x_p1_7)
        triplet_features.append(x_p1_5)
        predict.append(x_p1_6)
       # add part2 feature
        softmax_features.append(x_p2_4)
        triplet_features.append(x_p2_2)
        predict.append(x_p2_3)    
        softmax_features.append(x_p2_7)
        triplet_features.append(x_p2_5)
        predict.append(x_p2_6)
       # # add part3 feature
       #  softmax_features.append(x_p3_4)
       #  triplet_features.append(x_p3_2)
       #  predict.append(x_p3_3)    
       #  softmax_features.append(x_p3_7)
       #  triplet_features.append(x_p3_5)
       #  predict.append(x_p3_6)

        if self.training:
            return triplet_features, softmax_features
        else:
            return torch.cat(predict, 1)

    # def forward(self, x):
    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.relu(x)
    #     x = self.model.maxpool(x)

    #     x = self.model.layer1(x)
    #     x = self.model.layer2(x)
    #     x3 = self.model.layer3(x)
    #     x = self.model.layer4(x3)

    #     # --- layer3 分支 ---
    #     x3_1 = self.bottleneck1(x3)
    #     x3_2,x3_3,x3_4 = self.PBN1(x3_1)

    #     # --- global 分支 ---
    #     x4_1, x4_2, x4_3 = self.PBN2(x)

    #     # --- divide x into two parts ---
    #     # _, _,height, weight = x.shape # [N, C, H, W]
    #     height = x.shape[2]
    #     x_p1 = x[:, :, :int(height/2), :]
    #     x_p2 = x[:, :, int(height/2):, :]

    #     # --- part1 分支 ---
    #     x_p1_1 = self.bottleneck2(x_p1)
    #     x_p1_2, x_p1_3, x_p1_4 = self.PBN3(x_p1_1)

    #     # --- part2 分支 ---
    #     x_p2_1 = self.bottleneck3(x_p2)
    #     x_p2_2, x_p2_3, x_p2_4 = self.PBN4(x_p2_1)

    #     predict = []
    #     triplet_features = []
    #     softmax_features = []

    #     # add layer3 feature
    #     softmax_features.append(x3_4)
    #     triplet_features.append(x3_2)
    #     predict.append(x3_3)
       
    #    # add global feature
    #     softmax_features.append(x4_3)
    #     triplet_features.append(x4_1)
    #     predict.append(x4_2)
      
    #    # add part1 feature
    #     softmax_features.append(x_p1_4)
    #     triplet_features.append(x_p1_2)
    #     predict.append(x_p1_3)

    #    # add part2 feature
    #     softmax_features.append(x_p2_4)
    #     triplet_features.append(x_p2_2)
    #     predict.append(x_p2_3)    

    #     if self.training:
    #         return triplet_features, softmax_features
    #     else:
    #         return torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.global_softmax.parameters()},
            {'params': self.res_part2.parameters()},
            {'params': self.reduction.parameters()},
            {'params': self.softmax.parameters()},
        ]
        return params

class Resnet(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(Resnet, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        x = self.global_avgpool(x).squeeze()
        feature = self.softmax(x)
        if self.training:
            return [], [feature]
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()

class IDE(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(IDE, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AvgPool2d(kernel_size=(12, 4))

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        feature = self.global_avgpool(x).squeeze()
        if self.training:
            return [feature], []
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()