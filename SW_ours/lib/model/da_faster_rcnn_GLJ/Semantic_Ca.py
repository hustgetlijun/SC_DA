import torch
import numpy as np
import copy

class Semantic_patch_Ca():
    def __init__(self, classes , save_gamma = 0.05 , deata = 1.1):
        super(Semantic_patch_Ca, self).__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.save_gamma = save_gamma
        self.deata= deata
    def Semantic_Ca(self,image_data,feature,gt_bbox,num_gt_bbox):
        bach_size_f,_,f_h,f_w=feature.size()
        bach_size_img,_,img_h,img_w=image_data.size()

        assert bach_size_img == bach_size_f
        w_stride = img_w/float(f_w)
        h_stride = img_h/float(f_h)

        Semantic_vecoter=torch.zeros((bach_size_f,self.num_classes,f_w,f_h))

        # print('Semantic_vecoter type is:', Semantic_vecoter.size())
        Semantic_vecoter=Semantic_vecoter.contiguous().view(bach_size_f,-1,self.num_classes)
        # print('Semantic_vecoter type1 is:',Semantic_vecoter.size())
        for bach_size_index in np.arange(0,bach_size_img):
            gt=gt_bbox[bach_size_index]
            gt_num = num_gt_bbox[bach_size_index]
            shift_x = np.arange(0, f_w).astype(np.float32) * w_stride
            # print('shift_x is :',shift_x)
            shift_y = np.arange(0, f_h).astype(np.float32) * h_stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel())).transpose())
            shifts = shifts.contiguous()

            # print('shifts size is :',shifts.data)
            for location_index in np.arange(0,len(shifts)):
                x,y = shifts[location_index]
                stride = torch.rand(2)
                stride[0] = w_stride
                stride[1] = h_stride
                stride.type_as(x).float()
                Semantic_vecoter[bach_size_index][location_index] =self.calculate_vector(x,y,stride,gt,gt_num,self.num_classes)
        Semantic_vecoter = Semantic_vecoter.transpose(2,1)
        Semantic_vecoter=Semantic_vecoter.view(bach_size_f,-1,f_w,f_h)
        Semantic_vecoter = Semantic_vecoter.transpose(3,2)
        # print('Semantic vector 0 is :',Semantic_vecoter)
        return Semantic_vecoter

    def calculate_vector(self,x,y,strides,gt,gt_num, num_class):
        # class_names = np.arange(0,num_class)
        semantic_vector_patch = torch.zeros(num_class)
        block = np.arange(0,4).astype(np.float32)
        block[0] = x
        block[1] = y
        block[2] = strides[0]
        block[3] = strides[1]
        image_block = torch.from_numpy(block)
        window= self._whctrs_img(image_block)
        gt_transform=self._whctrs_(gt,gt_num)
        window=window.type_as(gt_transform[0]).float()
        # print('---------------window is :', window.data)
        no_zero_gt = []
        window_area = window[2] * window[3]
        one_flag = False
        for gt_num_index in list(range(gt_num)):
            class_id = gt_transform[gt_num_index][4]
            class_id = int(class_id)
            ins_area = self._box_instersection_(window,gt_transform[gt_num_index])
            # print('********************ins_area is ', ins_area)
            if ins_area:
                no_zero_gt.append(gt_transform[gt_num_index][:4])
                # pra = ins_area / float(gt_transform[gt_num_index][2].data * gt_transform[gt_num_index][3])
                pra = ins_area / float(window_area)
            else:
                pra = torch.zeros(1)
            # print('------------------------------------------------------pra is :', pra)
            # print('semantic_vector_patch[class_id] :', semantic_vector_patch[class_id].data)
            if pra.cuda() > semantic_vector_patch[class_id].cuda():
                semantic_vector_patch[class_id] = pra
                if pra > 1-self.save_gamma:
                    # print('semantic_vector_patch 0 is :',semantic_vector_patch)
                    one_flag = True
                    # return semantic_vector_patch
        if one_flag :
            # one_flag = False
            # print('semantic vector data 0 is :', semantic_vector_patch.data)
            return semantic_vector_patch

        if len(no_zero_gt) == 0:
            semantic_vector_patch[0] = 1.0
            # print('semantic vector data 1 is :', semantic_vector_patch.data)
            return semantic_vector_patch

        save_gt_insert=[]
        area_all = 0

        for np_zero_index in np.arange(0,len(no_zero_gt)):
            gt_box = no_zero_gt[np_zero_index]
            insert_cor, area_box = self.coordinate_insert(window,gt_box)
            if len(insert_cor) ==1:
                if insert_cor == 1:
                    semantic_vector_patch[0] = self.save_gamma
                    # print('semantic vector data is2 :', semantic_vector_patch.data)
                    return semantic_vector_patch
                    # break
            area_all = area_all + area_box
            save_gt_insert.append(insert_cor)

        if area_all > window_area:
            semantic_vector_patch[0] = self.save_gamma
            # print('semantic vector data is 3:', semantic_vector_patch.data)
            return semantic_vector_patch
        semantic_vector_patch[0] = (window_area - area_all)/float(window_area*self.deata)
        # print('semantic vector data 4 is :',semantic_vector_patch.data)
        return semantic_vector_patch

    def coordinate_insert(self,window,box):
        x, y, w, h = window[:4]
        x1,y1,w1,h1 = box[:4]
        w2, (xmin,xmax) = self._overlap_(x,w,x1,w1)
        h2, (ymin, ymax) = self._overlap_(y, h, y1, h1)
        if w2 < 0 or h2 <0 or w2 == 0 or h2 ==0:
            return False,False
        if w2*h2 / (w*h) < self.save_gamma: #0.05
            return [0], 0
        if w2*h2/(w*h) > (1-self.save_gamma):
            return [1], 1
        return self._whctrs_img([xmin,ymin,w2,h2]), w2*h2


    def _whctrs_(self,gt,gt_num):
        """
        Return width, height, x center, and y center for an anchor (window).
        """
        # print('gt_num is :',0)
        gt_transform = copy.deepcopy(gt[:gt_num])
        # print('gt_transform  is :',gt_transform)
        for gt_index in list(range(gt_num)):
            w = gt[gt_index][2].float() - gt[gt_index][0].float()
            h = gt[gt_index][3].float() - gt[gt_index][1].float()
            x_ctr = gt[gt_index][0].float() + 0.5 * w
            y_ctr = gt[gt_index][1].float() + 0.5 * h
            # print('gt transform is :',gt_transform[gt_index][:4])
            # print('box is :',[x_ctr,y_ctr, w, h])
            gt_transform[gt_index][0] = x_ctr
            gt_transform[gt_index][1] = y_ctr
            gt_transform[gt_index][2] = w
            gt_transform[gt_index][3] = h

        # print('gt_transform1  is :', gt_transform)
        return gt_transform


    def _whctrs_img(self,anchor1):
        """
        Return width, height, x center, and y center for an anchor (window).
        """
        anchor = torch.rand(4)
        w = anchor1[2].float()
        h = anchor1[3].float()
        x_ctr = anchor1[0].float() + w/2.0
        y_ctr = anchor1[1].float() + h/2.0
        anchor[0] = x_ctr
        anchor[1] = y_ctr
        anchor[2] = w
        anchor[3] = h
        return anchor

    def _overlap_(self,x1,w1,x2,w2):

        l1=x1-w1*float(0.5)
        l2=x2-w2*float(0.5)
        if l1 > l2 :
            left = l1
        else:
            left = l2
        r1=x1+w1*0.5
        r2=x2+w2*0.5
        if r1 > r2 :
            right = r2
        else:
            right =r1
        return right-left , (left,right)

    def _box_instersection_(self,box1,box2):
        w, _ = self._overlap_(box1[0],box1[2],box2[0],box2[2])
        h, _ = self._overlap_(box1[1],box1[3],box2[1],box2[3])
        if w <0 or h <0 or w == 0 or h == 0:
            return False
        area = w*h
        # print('area is :',area)
        return area
    def _box_area_(self,box):
        return box[2]*box[3]