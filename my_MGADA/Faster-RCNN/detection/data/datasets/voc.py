from .dataset import VOCDataset


class CustomVocDataset(VOCDataset):
    def __init__(self, train, **kwargs):
        super().__init__(keep_difficult=not train, **kwargs)



class CityScape_VOC(VOCDataset):
    CLASSES = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

    def __init__(self, train, **kwargs):
        super().__init__(keep_difficult=not train, **kwargs)
        if train:
            self.is_sample = True
            img_ids = []
            origin_size = len(self.ids)
            for img_id in self.ids:
                ann = self.get_annotations_by_image_id(img_id)
                if ann == False :
                    continue
                if ann['boxes'].shape[0] > 0:
                    img_ids.append(img_id)
            self.ids = img_ids
            print('({})Only images containing gts are kept, from {} to {}'.format(self.dataset_name, origin_size, len(self.ids)))


class BDD100K_VOC(VOCDataset):
    CLASSES = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

    def __init__(self, train, **kwargs):
        super().__init__(keep_difficult=not train, **kwargs)
        if train:
            self.is_sample = True
            img_ids = []
            origin_size = len(self.ids)
            for img_id in self.ids:
                ann = self.get_annotations_by_image_id(img_id)
                if ann['boxes'].shape[0] > 0:
                    img_ids.append(img_id)
            self.ids = img_ids
            print('({})Only images containing gts are kept, from {} to {}'.format(self.dataset_name, origin_size, len(self.ids)))



class WatercolorDataset(VOCDataset):
    #CLASSES = ('__background__', 'bicycle', 'bird', 'car', 'cat', 'dog', 'person')
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, train, **kwargs):
        super().__init__(keep_difficult=not train, **kwargs)
        if train:
            self.is_sample = True
            img_ids = []
            origin_size = len(self.ids)
            for img_id in self.ids:
                ann = self.get_annotations_by_image_id(img_id)
                if ann is False:
                    continue
                if ann['boxes'].shape[0] > 0:
                    img_ids.append(img_id)
            self.ids = img_ids
            print('({})Only images containing gts are kept, from {} to {}'.format(self.dataset_name, origin_size, len(self.ids)))


class ClipartDataset(VOCDataset):
    #CLASSES = ('__background__', 'bicycle', 'bird', 'car', 'cat', 'dog', 'person')
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, train, **kwargs):
        super().__init__(keep_difficult=not train, **kwargs)
        if train:
            self.is_sample = True
            img_ids = []
            origin_size = len(self.ids)
            for img_id in self.ids:
                ann = self.get_annotations_by_image_id(img_id)
                if ann is False:
                    continue
                if ann['boxes'].shape[0] > 0:
                    img_ids.append(img_id)
            self.ids = img_ids
            print('({})Only images containing gts are kept, from {} to {}'.format(self.dataset_name, origin_size, len(self.ids)))


class Sim10kDataset(VOCDataset):
    CLASSES = ('__background__', 'car')

    def __init__(self, train, **kwargs):
        super().__init__(keep_difficult=not train, **kwargs)
        if train:
            self.is_sample = train
            img_ids = []
            origin_size = len(self.ids)
            for img_id in self.ids:
                ann = self.get_annotations_by_image_id(img_id)
                if ann['boxes'].shape[0] > 0:
                    img_ids.append(img_id)
            self.ids = img_ids
            print('({})Only images containing gts are kept, from {} to {}'.format(self.dataset_name, origin_size, len(self.ids)))


class KITTIDataset(VOCDataset):
    CLASSES = ('__background__', 'car')

    def __init__(self, train, **kwargs):
        super().__init__(keep_difficult=not train, img_ext='.png', **kwargs)
        if train:
            self.is_sample = train
            img_ids = []
            origin_size = len(self.ids)
            for img_id in self.ids:
                ann = self.get_annotations_by_image_id(img_id)
                if ann['boxes'].shape[0] > 0:
                    img_ids.append(img_id)
            self.ids = img_ids
            print('({})Only images containing gts are kept, from {} to {}'.format(self.dataset_name, origin_size, len(self.ids)))
