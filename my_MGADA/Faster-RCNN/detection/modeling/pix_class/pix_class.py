# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import numpy as np
import torch

def gt_classes2cls_lb_onehot(bachsize,targets,num_class):
    cls_lb = torch.zeros(bachsize, num_class - 1)
    for j in range(bachsize):
        array = targets[j]['labels']
        for i in array:
              cls_lb[j][i - 1] = 1
    return cls_lb

def clip_boxes(boxes, im_shape, batch_size):

    boxes[:, 0::4].clamp_(0, im_shape[1] - 1)  # limit the value in this interval
    boxes[:, 1::4].clamp_(0, im_shape[0] - 1)
    boxes[:, 2::4].clamp_(0, im_shape[1] - 1)
    boxes[:, 3::4].clamp_(0, im_shape[0] - 1)
    return boxes

def get_receptive_field_feature(feature_data,im_data,window_size,feat_stride):

    """
    feature_data: (B,C,H1,W1) ndarray of float
    im_data: (B,C,H,W) ndarray of float
    all_anchors: (N=H1*W1, 4) ndarray 
    """
    # feature_data =feature.data
    # image_data = im_data.data
    _,_,feat_height,feat_width = feature_data.size()
    _,_,img_h,img_w = im_data.size()

    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = torch.from_numpy(
        np.vstack(
            (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
        ).transpose()
    )
    shifts = shifts.contiguous().type_as(feature_data).float()
    # print('shifts is :',shifts)
    wind_w,wind_h = window_size
    anchors = torch.Tensor([-(wind_w-1)/2.0,-(wind_h-1)/2.0,(wind_w-1)/2.0,(wind_h-1)/2.0])
    anchors = anchors.type_as(feature_data).float()

    K=shifts.size(0)
    all_anchors = anchors.view(1, 1, 4) + shifts.view(K, 1, 4)
    all_anchors = all_anchors.view(K, 4)
    # print('all anchor is :',all_anchors)
    all_anchors = clip_boxes(all_anchors,(img_h,img_w),1)
    return all_anchors


def pix_class_lable(anchors, gt_boxes, num_class, apl_a):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 5) ndarray of float
    class_vector: (N, num_class) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)
    apla_a = apl_a

    class_vector = gt_boxes.new(N, num_class).zero_()

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0]+1) * (gt_boxes[:, 3] - gt_boxes[:, 1]+1)
    ).view(1, K).expand(N,K).contiguous().view(N*K,1)

    # print('gt box area is:',gt_boxes_area)
    anchors_area = (
        (anchors[:, 2] - anchors[:, 0]+1) * (anchors[:, 3] - anchors[:, 1]+1)
    ).view(N, 1).expand(N,K).contiguous().view(N*K,1)

    # print('anchor box area is:', anchors_area)
    area_choose = torch.min(gt_boxes_area[:,:],anchors_area[:,:]).contiguous().view(N,K)
    # print('area choose is :',area_choose)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 5).expand(N, K, 5)

    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        +1

    )
    iw[iw <= 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        +1

    )
    ih[ih <= 0] = 0
    # print('iw*ih is :',iw*ih)
    overlaps = iw*ih / area_choose
    # print('keep index is pre:', overlaps)
    keep_index = (overlaps[:,:] >= apla_a)
    # print('keep index is post:',keep_index)
    gt_class = query_boxes[:,:,4]
    # print('gt_class shape is1 :', gt_class)
    gt_class = gt_class*keep_index.float()

    # print('gt_class shape is2 :',gt_class)
    for i in range(num_class):
        class_save = (gt_class[:,:] == i+1)
        class_save = torch.sum(class_save.float(),dim=1)
        class_save = (class_save[:] >=1).float()
        class_vector[:,i] = class_save
    return class_vector



def pix_class_lable_one(anchors, gt_boxes, num_class, apl_a):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 5) ndarray of float
    class_vector: (N, num_class) ndarray of overlap between boxes and query_boxes
    """
    label_one_vector = anchors.new(anchors.size(0)).zero_()
    class_vector = pix_class_lable(anchors, gt_boxes, num_class, apl_a)
    label_one_vector = torch.max(class_vector,dim=1)[0]
    return label_one_vector.view(label_one_vector.size(0),1)



#
# anchor=torch.Tensor([[0,0,2,2],[2,0,4,2],[0,2,2,4],[2,2,4,4]])
# image_feature = torch.rand(1,3,5,5)
#
# base_feature = torch.rand(1,1,3,3)
# window = (3,3)
# strite = 2
# anchor = get_receptive_field_feature(base_feature,image_feature,window,strite)
# print('anchor is :',anchor)
#
# gt = torch.Tensor([[0,0,1,1,2],[0,1,1,4,3],[1,1,3,3,4],[2,3,4,4,1]])
# class_num = 4
# apla=0.5
# result= pix_class_lable(anchor,gt,class_num,apla)
# print('result is :',result)