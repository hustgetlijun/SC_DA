# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.da_faster_rcnn_change_test.faster_rcnn import _fasterRCNN
from model.utils.config import cfg
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class netD_pixel(nn.Module):
    def __init__(self, n_class,context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = conv1x1(256+n_class, 256+n_class)
        # self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv1x1(256+n_class, 128)
        # self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv1x1(128, 1)

        self.context = context
        self._init_weights()

    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)


    def forward(self, x):
        x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            # feat = x
            x = F.sigmoid(self.conv3(x))
            return x.view(-1, 1), feat  # torch.cat((feat1,feat2),1)#F
        else:
            x = F.sigmoid(self.conv3(x))
            return x.view(-1, 1)  # F.sigmoid(x)


class netD_pixel_mid(nn.Module):
    def __init__(self,n_class,context=False):
        super(netD_pixel_mid, self).__init__()
        self.conv1 = conv1x1(512+n_class, 512+n_class)
        self.conv1_1 = conv1x1(512+n_class, 256)
        # self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv1x1(256, 256)
        self.conv2_1 = conv1x1(256, 128)
        # self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv1x1(128, 1)

        self.context = context
        self._init_weights()

    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv1_1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv2_1, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_1(x))

        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            # feat = x
            x = F.sigmoid(self.conv3(x))
            return x.view(-1, 1), feat  # torch.cat((feat1,feat2),1)#F
        else:
            x = F.sigmoid(self.conv3(x))
            return x.view(-1, 1)  # F.sigmoid(x)


class netD(nn.Module):
    def __init__(self,n_class, context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(512+n_class, 512+n_class, stride=2)
        self.bn1 = nn.BatchNorm2d(512+n_class)
        self.conv2 = conv3x3(512+n_class, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 2)
        self.context = context

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        if self.context:
            feat = x
        x = self.fc(x)
        if self.context:
            return x, feat  # torch.cat((feat1,feat2),1)#F
        else:
            return x

class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), training=self.training)
        x = self.fc3(x)
        return x


class vgg16(_fasterRCNN):
    def __init__(
        self,
        classes,
        pretrained_path,
        pretrained=False,
        class_agnostic=False,
        lc=False,
        gc=False,
    ):
        self.model_path = pretrained_path
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.lc = lc
        self.gc = gc

        _fasterRCNN.__init__(self,classes, class_agnostic, self.lc, self.gc)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict(
                {k: v for k, v in state_dict.items() if k in vgg.state_dict()}
            )

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # print(vgg.features)
        
        self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])

        self.RCNN_base_mid = nn.Sequential(*list(vgg.features._modules.values())[14:21])

        self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])


        self.netD_pixel = netD_pixel(128, context=self.lc)
        self.netD_pixel_mid = netD_pixel_mid(128,context=self.gc)
        self.netD = netD(0, context=self.gc)

        feat_d = 4096
        if self.lc:
            feat_d += 128
        if self.gc:
            feat_d += 128
        # Fix the layers before conv3t()
        for layer in range(10):
            for p in self.RCNN_base1[layer].parameters():
                p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
