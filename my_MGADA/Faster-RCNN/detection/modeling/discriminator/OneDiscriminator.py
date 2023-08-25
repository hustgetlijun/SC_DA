from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from .layer import GradientReversal

########################## local domain and midlle domain ############################
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
    def __init__(self, n_class, context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = conv1x1(256+n_class, 256+n_class)
        # self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv1x1(256+n_class, 128)
        # self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv1x1(128, 1)
        self.grad_reverse = GradientReversal(1.0)
        self.context = context
        self.loss_fn = nn.BCEWithLogitsLoss()

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

    def forward(self, x, target, pix_weight=[]):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9

        x = self.grad_reverse(x)
        # x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            # feat = x
            x = self.conv3(x)
            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn(x, target)
            return loss, feat  # torch.cat((feat1,feat2),1)#F
        else:
            x = self.conv3(x)
            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn(x,target)
            #loss = self.loss_fn(x*(1+pix_weight),target)
            #loss = torch.mean(((target-F.sigmoid(x)) ** 2)*(1+pix_weight))
            return loss  # F.sigmoid(x)


class netD_pixel_mid(nn.Module):
    def __init__(self,n_class,context=False):
        super(netD_pixel_mid, self).__init__()
        self.conv1 = conv1x1(512+n_class, 512+n_class)
        self.conv2 = conv1x1(512+n_class, 256)
        # self.bn1 = nn.BatchNorm2d(256)
        self.conv3 = conv1x1(256, 256)
        self.conv4 = conv1x1(256, 128)
        # self.bn2 = nn.BatchNorm2d(128)
        self.conv5 = conv1x1(128, 1)
        self.grad_reverse = GradientReversal(1.0)
        self.context = context
        self.loss_fn = nn.BCEWithLogitsLoss()

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
      normal_init(self.conv4, 0, 0.01)
      normal_init(self.conv5, 0, 0.01)

    def forward(self, x, target, pix_weight=[]):
        x = self.grad_reverse(x)
        # x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = self.conv5(x)
            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn(x, target)
            return loss, feat  # torch.cat((feat1,feat2),1)#F
        else:
            x = self.conv5(x)
            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn(x,target)
            
            #p = 2-pix_weight.values
            #p = p.unsqueeze(1)
            #loss = self.loss_fn(x*p,target)
            #loss = torch.mean(((target-F.sigmoid(x)) ** 2)*p)
            return loss

########################## local domain and midlle domain ############################


class OneFCOSDiscriminator(nn.Module):
    def __init__(self, cfg, in_channels=256,  grl_applied_domain='both'):
        """
        Arguments:
           in_channels (int): number of channels of the input feature
        """
        super(OneFCOSDiscriminator, self).__init__()
        num_convs = cfg.MODEL.ROI_ONE_DIS.DOM.NUM_CONVS  #3
        grad_reverse_lambda = cfg.MODEL.ROI_ONE_DIS.DOM.GRL_LAMBDA  #0.01
        
        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain
        
    def forward(self, feature, target, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'
        
        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
        
        x = self.dis_tower(feature)
        x = self.cls_logits(x)

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        loss = self.loss_fn(x, target)

        return loss


class OneFCOSDiscriminator_cc(nn.Module):
    def __init__(self, cfg, in_channels=256, grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(OneFCOSDiscriminator_cc, self).__init__()
        num_convs = cfg.MODEL.ROI_ONE_DIS.CLS.NUM_CONVS
        grad_reverse_lambda = cfg.MODEL.ROI_ONE_DIS.CLS.GRL_LAMBDA
        
        self.loss_direct_w = cfg.MODEL.ROI_ONE_DIS.CLS.LOSS_DIRECT_W #1.0
        self.loss_grl_w = cfg.MODEL.ROI_ONE_DIS.CLS.LOSS_GRL_W #0.1
        self.samples_thresh = cfg.MODEL.ROI_ONE_DIS.CLS.SAMPLES_THRESH #0.8

        self.num_classes = cfg.MODEL.ROI_ONE_DIS.CLS.NUM_CLASSES #2
        self.out_classes = cfg.MODEL.ROI_ONE_DIS.CLS.NUM_CLASSES * 2

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.out_classes, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_direct_f = nn.CrossEntropyLoss()#
        self.loss_grl_f = nn.BCELoss()#

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain


    def forward(self, feature, target, pred_dict, groundtruth, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        pred_dict = pred_dict.sigmoid()
        scores_mx = pred_dict.max()
        
        loss_direct = self.forward_direct(feature, target, pred_dict, groundtruth, domain, scores_mx)
        
        loess_grl = self.forward_grl(feature, target, pred_dict, groundtruth, domain, scores_mx)
        
        loss = self.loss_direct_w * loss_direct + self.loss_grl_w * loess_grl
        
        return loss
    

    def forward_direct(self, feature, target, pred_cls, groundtruth, domain='source', scores_mx = 1.0):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        x = self.dis_tower(feature)
        x = self.cls_logits(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes, 2).sum(dim=2)


        nb, nc, nh, nw = pred_cls.size()
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(-1, nc)
        pred_cls_v, pred_cls_index = pred_cls.max(dim=1)

        gt_mask = (pred_cls_v > scores_mx * self.samples_thresh).long()

        loss = self.loss_direct_f(x, gt_mask)
        return loss
    

    def forward_grl(self, feature, target, pred_cls, groundtruth, domain='source', scores_mx = 1.0):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
        x = self.dis_tower(feature)
        x = self.cls_logits(x)
        x = F.softmax(x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes, 2), 2).view(-1, self.out_classes)
        
        nb, nc, nh, nw = pred_cls.size()
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(-1, nc)
        pred_cls_v, pred_cls_index = pred_cls.max(dim=1)
        pred_cls_index = (pred_cls_v > scores_mx * self.samples_thresh).long()

        loss = 0.0 * torch.sum(x)
        for ii in range(1, self.num_classes):
            cls_idxs = pred_cls_index == ii
            pred_cls_idx = pred_cls_v[cls_idxs]
            if pred_cls_idx.size(0) == 0:
                continue

            dx_cls_idx = x[cls_idxs,:]
            
            cls_idxs = pred_cls_idx > self.samples_thresh * scores_mx
            pred_cls_idx = pred_cls_idx[cls_idxs]
            if pred_cls_idx.size(0) == 0:
                continue
            if domain == 'target':
                dx_cls_idx = dx_cls_idx[cls_idxs,ii*2]
            elif domain == 'source':
                dx_cls_idx = dx_cls_idx[cls_idxs,ii*2+1]

            target_idx = torch.full(dx_cls_idx.shape, 1.0, dtype=torch.float, device=dx_cls_idx.device)
            loss += self.loss_grl_f(dx_cls_idx, target_idx)
                
        return loss

