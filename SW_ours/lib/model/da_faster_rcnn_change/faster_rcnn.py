import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
    grad_reverse,
)
from torch.autograd import Variable
from model.pix_class.pix_class import get_receptive_field_feature,pix_class_lable,pix_class_lable_one


# class object_pre(nn.Module):
#     def __init__(self,dout_base_model):
#         super(object_pre,self).__init__()
#
#         self.conv1 = nn.Conv2d(dout_base_model, dout_base_model, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv2 = nn.Conv2d(dout_base_model, 128, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv4 = nn.Conv2d(128,1, kernel_size=1, stride=1, padding=0, bias=False)
#     def forward(self, x):
#         # x = F.relu(x)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x =  self.conv4(x)
#         return x
class object_pre(nn.Module):
    def __init__(self,dout_base_model):
        super(object_pre,self).__init__()

        self.conv1 = nn.Conv2d(dout_base_model, dout_base_model, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dout_base_model, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(128,1, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        # x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        feat = F.relu(self.conv3(x))
        x =  self.conv4(feat)
        return x, feat

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean
                )  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.conv1 , 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.conv3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.conv4, 0, 0.01, cfg.TRAIN.TRUNCATED)



# class class_pre(nn.Module):
#     def __init__(self,dout_base_model,n_class):
#         super(class_pre,self).__init__()
#         self.dout = dout_base_model
#         self.conv1 = nn.Conv2d(dout_base_model, dout_base_model, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv2 = nn.Conv2d(dout_base_model, 256, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv4 = nn.Conv2d(128, n_class, 1, 1, 0)
#     def forward(self, x):
#         # x = F.relu(x)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x =  self.conv4(x)
#         return x
class class_pre(nn.Module):
    def __init__(self,dout_base_model,n_class):
        super(class_pre,self).__init__()
        self.dout = dout_base_model
        self.conv1 = nn.Conv2d(dout_base_model, dout_base_model, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dout_base_model, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(128, n_class, 1, 1, 0)
    def forward(self, x):
        # x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        feat = F.relu(self.conv3(x))
        x =  self.conv4(feat)
        return x,feat

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean
                )  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.conv1 , 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.conv3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.conv4, 0, 0.01, cfg.TRAIN.TRUNCATED)





class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, lc, gc):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = (
            cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        )

        self.conv_lst = nn.Conv2d(self.dout_base_model, self.n_classes - 1, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # self.pix_pre_Conv = nn.Conv2d(self.dout_base_model, 512, 3, 1, 1, bias=True)
        # self.conv_lst = nn.Conv2d(512, self.n_classes - 1, 1, 1, 0)

        # self.bn1 = nn.BatchNorm2d(self.dout_base_model, momentum=0.01)
        # self.bn2 = nn.BatchNorm2d(self.n_classes-1, momentum=0.01)
        self.object_pre = object_pre(256)
        # self.classs_pre = class_pre(512,self.n_classes-1)
        # self.class_pre_global = class_pre(512, self.n_classes - 1)
        self.classs_pre = class_pre(512, self.n_classes - 1)


    def forward(
        self, im_data, im_info, im_cls_lb, gt_boxes, num_boxes, target=False, eta=1.0
    ):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # gt_boxes = gt_boxes.view(1,gt_boxes.size(0),gt_boxes.size(1))

        # print('gt boxes shape is:',gt_boxes.size())

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)

        #**********************************************pix domain**********************
        # pix_object = self.object_pre(base_feat1)
        # base_object_feat = torch.cat((base_feat1, F.softmax(pix_object).detach()), dim=1)
        #
        # windows = get_receptive_field_feature(base_feat1, im_data, (40, 40), 4)
        # object_label = pix_class_lable_one(windows, gt_boxes, self.n_classes - 1, 0.6)
        #
        # pix_object = pix_object.permute(0, 2, 3, 1)
        # object_feat = pix_object.contiguous().view(base_feat1.size(2) * base_feat1.size(3), 1)
        # category_loss_object = nn.BCEWithLogitsLoss()(object_feat, object_label)


        # pix_object = self.object_pre(base_feat1)
        # base_object_feat = torch.cat((base_feat1, torch.relu(pix_object)), dim=1)
        pix_object,feat_object = self.object_pre(base_feat1)
        base_object_feat = torch.cat((base_feat1, feat_object), dim=1)

        # pix_object_1 = (F.sigmoid(pix_object) > 0.6).float()

        windows = get_receptive_field_feature(base_feat1,im_data,(40,40),4)
        object_label = pix_class_lable_one(windows,gt_boxes,self.n_classes-1,0.6)

        pix_object = pix_object.permute(0, 2, 3, 1)
        object_feat = pix_object.contiguous().view(base_feat1.size(2)*base_feat1.size(3),1)
        category_loss_object = nn.BCEWithLogitsLoss()(object_feat, object_label)
        # category_loss_object = nn.CrossEntropyLoss()(object_feat, object_label)
        # category_loss_cls = torch.cosine_similarity(cls_feat,pix_label,dim=0)
        # print('target is:', target)
        #
        # if target is  0:
        #     base_object_feat = torch.cat((base_feat1, pix_object_1.detach()), dim=1)
        # else:
        #     object_label = object_label.contiguous().view(pix_object_1.size(0),pix_object_1.size(2),pix_object_1.size(3),pix_object_1.size(1))
        #     object_label = object_label.permute(0,3,1,2)
        #     # print('object label shape is :',object_label.size())
        #     # print('base feat 1 shape is:', base_feat1.size())
        #     base_object_feat = torch.cat((base_feat1, object_label.detach()), dim=1)

        #*****************************************END***********************************


        if self.lc:
            d_pixel, _ = self.netD_pixel(grad_reverse(base_object_feat, lambd=eta))
            # print(d_pixel)
            # print('target is :', target)
            if True:
                _, feat_pixel = self.netD_pixel(base_object_feat.detach())
        else:
            d_pixel = self.netD_pixel(grad_reverse(base_object_feat, lambd=eta))

        base_feat_mid = self.RCNN_base_mid(base_feat1)

        #*******************mid pix domain**************************
        # cls_feat = self.classs_pre(base_feat_mid)
        # base_class_feat = torch.cat((base_feat_mid, F.sigmoid(cls_feat).detach()), dim=1)
        #
        # windows = get_receptive_field_feature(base_feat_mid,im_data,(92,92),8)
        # pix_label = pix_class_lable(windows,gt_boxes,self.n_classes-1,0.6)
        #
        # cls_feat = cls_feat.permute(0, 2, 3, 1)
        # cls_feat = cls_feat.contiguous().view(base_feat_mid.size(2)*base_feat_mid.size(3),self.n_classes-1)
        # # print('cls feat shape is:', cls_feat.size())
        # category_loss_mid_cls = nn.BCEWithLogitsLoss()(cls_feat, pix_label)
        # # category_loss_cls = torch.cosine_similarity(cls_feat,pix_label,dim=0)


        # cls_feat = self.classs_pre(base_feat_mid)
        # base_class_feat = torch.cat((base_feat_mid, torch.relu(cls_feat)), dim=1)
        cls_feat,feat_mid = self.classs_pre(base_feat_mid)
        base_class_feat = torch.cat((base_feat_mid, feat_mid), dim=1)
        # base_class_feat_1= (F.sigmoid(cls_feat)>0.6).float()

        windows = get_receptive_field_feature(base_feat_mid,im_data,(92,92),8)
        pix_label = pix_class_lable(windows,gt_boxes,self.n_classes-1,0.6)

        cls_feat = cls_feat.permute(0, 2, 3, 1)
        cls_feat = cls_feat.contiguous().view(base_feat_mid.size(2)*base_feat_mid.size(3),self.n_classes-1)

        category_loss_mid_cls = nn.BCEWithLogitsLoss()(cls_feat, pix_label)
        # category_loss_mid_cls = torch.cosine_similarity(cls_feat,pix_label,dim=0)

        # if target is 0:
        #     base_class_feat = torch.cat((base_feat_mid, base_class_feat_1.detach()), dim=1)
        # else:
        #     label_class = pix_label.contiguous().view(base_class_feat_1.size(0),base_class_feat_1.size(2),base_class_feat_1.size(3),base_class_feat_1.size(1))
        #     label_class = label_class.permute(0,3,1,2)
        #     # print('label class shape is:',label_class.size())
        #     # print('base feat shape is :', base_feat1.size())
        #     base_class_feat = torch.cat((base_feat_mid, label_class.detach()), dim=1)



        domain_pix_mid = self.netD_pixel_mid(grad_reverse(base_class_feat, lambd=eta))

        #*************************END************************

        base_feat = self.RCNN_base2(base_feat_mid)


        #***********************************pix*******************************
        # # cls_feat = self.pix_pre_Conv(base_feat)
        # # cls_feat = self.conv_lst(cls_feat)
        # cls_feat = self.class_pre_global(base_feat)
        # feat_base = torch.cat((base_feat, F.sigmoid(cls_feat).detach()), dim=1)
        # # print('feat base shape is :',feat_base.size())
        # windows = get_receptive_field_feature(base_feat, im_data, (196, 196), 16)
        # pix_label = pix_class_lable(windows, gt_boxes, self.n_classes - 1, 0.6)
        #
        # cls_feat = cls_feat.permute(0, 2, 3, 1)
        # cls_feat = cls_feat.contiguous().view(base_feat.size(2) * base_feat.size(3), self.n_classes - 1)
        # # print('cls feat shape is:', cls_feat.size())
        # category_loss_cls = nn.BCEWithLogitsLoss()(cls_feat, pix_label)


        # cls_feat = self.class_pre_global(base_feat)
        # feat_base = torch.cat((base_feat, torch.relu(cls_feat)), dim=1)
        # cls_feat,feat_class = self.class_pre_global(base_feat)
        # feat_base = torch.cat((base_feat, feat_class), dim=1)
        #
        # # feat_base_1=(F.sigmoid(cls_feat) > 0.6).float()
        #
        # windows = get_receptive_field_feature(base_feat, im_data, (196, 196), 16)
        # pix_label = pix_class_lable(windows, gt_boxes, self.n_classes - 1, 0.6)
        #
        # cls_feat = cls_feat.permute(0, 2, 3, 1)
        # cls_feat = cls_feat.contiguous().view(base_feat.size(2) * base_feat.size(3), self.n_classes - 1)
        #
        # category_loss_cls = nn.BCEWithLogitsLoss()(cls_feat, pix_label)
        # category_loss_cls = torch.cosine_similarity(cls_feat, pix_label, dim=0)

        # if target is 0:
        #     feat_base = torch.cat((base_feat, feat_base_1.detach()), dim=1)
        # else:
        #     label_class = pix_label.contiguous().view(feat_base_1.size(0),feat_base_1.size(2),feat_base_1.size(3),feat_base_1.size(1))
        #     label_class = label_class.permute(0,3,1,2)
        #     feat_base = torch.cat((base_feat, label_class.detach()), dim=1)

        #*********************************************************************

        if self.gc:
            domain_p, _ = self.netD(grad_reverse(base_feat, lambd=eta))
            if False:
                return d_pixel, domain_p  # , diff
            _, feat = self.netD(base_feat.detach())
        else:
            domain_p = self.netD(grad_reverse(base_feat, lambd=eta))
            if False:
                return d_pixel, domain_p  # ,diff
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes
        )
        # supervise base feature map with category level label
        cls_feat = self.avg_pool(base_feat)
        cls_feat = self.conv_lst(cls_feat).squeeze(-1).squeeze(-1)
        # cls_feat = self.conv_lst(self.bn1(self.avg_pool(base_feat))).squeeze(-1).squeeze(-1)
        category_loss_cls = nn.BCEWithLogitsLoss()(cls_feat, im_cls_lb)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2))
            )
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "align":
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # feat_pixel = torch.zeros(feat_pixel.size()).cuda()
        if self.lc:
            feat_pixel = feat_pixel.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat_pixel, pooled_feat), 1)
        if self.gc:
            feat = feat.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat, pooled_feat), 1)
            # compute bbox offset

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4
            )
            bbox_pred_select = torch.gather(
                bbox_pred_view,
                1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4
                ),
            )
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws
            )

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return (
            rois,
            cls_prob,
            bbox_pred,
            category_loss_object,
            category_loss_mid_cls,
            category_loss_cls,
            rpn_loss_cls,
            rpn_loss_bbox,
            RCNN_loss_cls,
            RCNN_loss_bbox,
            rois_label,
            d_pixel,
            domain_pix_mid,
            domain_p,
        )  # ,diff

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean
                )  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
