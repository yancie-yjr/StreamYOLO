import cv2
import numpy as np
# import json
# import time
# from PIL import Image
from pycocotools.coco import COCO
# from collections import defaultdict

# import io
import os

# from yolox.data.dataloading import get_yolox_datadir
from yolox.data.datasets.datasets_wrapper import Dataset

# from loguru import logger

class ONE_ARGOVERSEDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, data_dir='./data/Datasets/', json_file='train.json',
                 name='train', img_size=(416,416), preproc=None, cache=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            debug (bool): if True, only one data id is selected from the dataset
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        # print("data_dir", data_dir)
        self.json_file = json_file
        self.coco = COCO(self.data_dir+'/Argoverse-HD/annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        self.seq_dirs = self.coco.dataset['seq_dirs']
        self.class_ids = sorted(self.coco.getCatIds())
        # {0: {'id': 0, 'name': 'person'}, 1: {'id': 1, 'name': 'bicycle'}, 2: {'id': 2, 'name': 'car'},
        # 3: {'id': 3, 'name': 'motorcycle'}, 4: {'id': 4, 'name': 'bus'}, 5: {'id': 5, 'name': 'truck'},
        # 6: {'id': 6, 'name': 'traffic_light'}, 7: {'id': 7, 'name': 'stop_sign'}}
        self._classes = self.coco.cats
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        self.imgs = None

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        if self.imgs:
            del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        im_name = im_ann['name']
        im_sid = im_ann['sid']

        seq_len = len(self.ids)


        #################future  annotation#############

        # 该个文件夹的第一张
        if self.coco.dataset['images'][int(id_)]['fid'] == 0:
            im_ann_support = im_ann
            im_ann_support2 = im_ann
            im_ann_support3 = im_ann
        
        # 整个数据集的最后一张
        elif int(id_) == seq_len-1:
            im_ann_support = im_ann
            im_ann_support2 = im_ann
            im_ann_support3 = im_ann

        # 下一张是新文件夹的第一张(即当前是该文件夹的最后一张)
        elif self.coco.dataset['images'][int(id_ + 1)]['fid'] == 0:
            im_ann_support = im_ann
            im_ann_support2 = im_ann
            im_ann_support3 = im_ann

        # 该文件夹中的第二张
        elif self.coco.dataset['images'][int(id_)]['fid'] == 1:
            im_ann_support = self.coco.loadImgs(id_ - 1)[0]     # load 第 ft-1 张图像
            im_ann_support2 = self.coco.loadImgs(id_ - 1)[0]     # load 第 ft-1 张图像
            im_ann_support3 = self.coco.loadImgs(id_ - 1)[0]     # load 第 ft-1 张图像
        
        elif self.coco.dataset['images'][int(id_)]['fid'] == 2:
            im_ann_support = self.coco.loadImgs(id_ - 1)[0]     # load 第 ft-1 张图像
            im_ann_support2 = self.coco.loadImgs(id_ - 2)[0]     # load 第 ft-2 张图像
            im_ann_support3 = self.coco.loadImgs(id_ - 2)[0]     # load 第 ft-2 张图像

        else:
            im_ann_support = self.coco.loadImgs(id_ - 1)[0]     # load 第 ft-1 张图像
            im_ann_support2 = self.coco.loadImgs(id_ - 2)[0]     # load 第 ft-2 张图像
            im_ann_support3 = self.coco.loadImgs(id_ - 3)[0]     # load 第 ft-3 张图像

        im_name_support = im_ann_support['name']
        im_sid_support = im_ann_support['sid']
        im_name_support2 = im_ann_support2['name']
        im_sid_support2 = im_ann_support2['sid']
        im_name_support3 = im_ann_support3['name']
        im_sid_support3 = im_ann_support3['sid']



        ## back seq fid
        if id_ in [seq_len-1, seq_len-2, seq_len-3, seq_len-4]:   # 整个数据集的倒数四张
            anno_ids = self.coco.getAnnIds(imgIds=[int(seq_len)], iscrowd=False)
        ## back fid
        else:
            if self.coco.dataset['images'][int(id_)]['fid'] in [0, 1]:   # 文件夹的前两张
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)

        ## front fid
            elif self.coco.dataset['images'][int(id_ + 1)]['fid'] == 0:   # 下一张是新文件夹的第一张
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)

            else:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_ + 1)], iscrowd=False)

        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)


        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid], im_name)
        support_file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support], im_name_support)
        support_file_name2 = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support2], im_name_support2)
        support_file_name3 = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support3], im_name_support3)



        #################support  annotation#############
        support_anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        support_annotations = self.coco.loadAnns(support_anno_ids)

        support_objs = []
        for obj1 in support_annotations:
            x1 = np.max((0, obj1["bbox"][0]))
            y1 = np.max((0, obj1["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj1["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj1["bbox"][3]))))
            if obj1["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj1["clean_bbox"] = [x1, y1, x2, y2]
                support_objs.append(obj1)

        support_num_objs = len(support_objs)


        support_res = np.zeros((support_num_objs, 5))

        for ix, obj1 in enumerate(support_objs):
            support_cls = self.class_ids.index(obj1["category_id"])
            support_res[ix, 0:4] = obj1["clean_bbox"]
            support_res[ix, 4] = support_cls

        support_r = min(self.img_size[0] / height, self.img_size[1] / width)
        support_res[:, :4] *= support_r

        #################support2  annotation#############
        support2_anno_ids = self.coco.getAnnIds(imgIds=[int(id_ - 1)], iscrowd=False)
        support2_annotations = self.coco.loadAnns(support2_anno_ids)

        support2_objs = []
        for obj1 in support2_annotations:
            x1 = np.max((0, obj1["bbox"][0]))
            y1 = np.max((0, obj1["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj1["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj1["bbox"][3]))))
            if obj1["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj1["clean_bbox"] = [x1, y1, x2, y2]
                support2_objs.append(obj1)

        support2_num_objs = len(support2_objs)


        support2_res = np.zeros((support2_num_objs, 5))

        for ix, obj1 in enumerate(support2_objs):
            support2_cls = self.class_ids.index(obj1["category_id"])
            support2_res[ix, 0:4] = obj1["clean_bbox"]
            support2_res[ix, 4] = support2_cls

        support2_r = min(self.img_size[0] / height, self.img_size[1] / width)
        support2_res[:, :4] *= support2_r

        #################support3  annotation#############
        support3_anno_ids = self.coco.getAnnIds(imgIds=[int(id_ - 2)], iscrowd=False)
        support3_annotations = self.coco.loadAnns(support3_anno_ids)

        support3_objs = []
        for obj1 in support3_annotations:
            x1 = np.max((0, obj1["bbox"][0]))
            y1 = np.max((0, obj1["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj1["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj1["bbox"][3]))))
            if obj1["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj1["clean_bbox"] = [x1, y1, x2, y2]
                support3_objs.append(obj1)

        support3_num_objs = len(support3_objs)


        support3_res = np.zeros((support3_num_objs, 5))

        for ix, obj1 in enumerate(support3_objs):
            support3_cls = self.class_ids.index(obj1["category_id"])
            support3_res[ix, 0:4] = obj1["clean_bbox"]
            support3_res[ix, 4] = support3_cls

        support3_r = min(self.img_size[0] / height, self.img_size[1] / width)
        support3_res[:, :4] *= support3_r

        support_file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support], im_name_support)
        support_file_name2 = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support2], im_name_support2)
        support_file_name3 = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support3], im_name_support3)

        return (res, support_res, support2_res, support3_res, img_info, resized_info, file_name, support_file_name, support_file_name2, support_file_name3)




    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img


    def load_image(self, index):
        file_name = self.annotations[index][6]

        img_file = file_name
        # print("img_file: ", img_file)
        img = cv2.imread(img_file)
        assert img is not None

        return img


    def load_support_resized_img(self, index):
        img = self.load_support_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_support_image(self, index):
        file_name = self.annotations[index][7]

        img_file = file_name

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def load_support_resized_img2(self, index):
        img = self.load_support_image2(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_support_image2(self, index):
        file_name = self.annotations[index][8]

        img_file = file_name

        img = cv2.imread(img_file)
        assert img is not None

        return img


    def load_support_resized_img3(self, index):
        img = self.load_support_image3(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_support_image3(self, index):
        file_name = self.annotations[index][9]

        img_file = file_name

        img = cv2.imread(img_file)
        assert img is not None

        return img


    def pull_item(self, index):
        id_ = self.ids[index]
        # print("annotations: ", self.annotations[index])
        res, support_res, support_res2, support_res3, img_info, resized_info, _, _, _, _ = self.annotations[index]

        img = self.load_resized_img(index)
        support_img = self.load_support_resized_img(index)
        support_img2 = self.load_support_resized_img2(index)
        support_img3 = self.load_support_resized_img3(index)

        return img, support_img, support_img2, support_img3, res.copy(), support_res.copy(), support_res2.copy(), support_res3.copy(), img_info, np.array([id_])


    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        img, support_img, support_img2, support_img3, target, support_target, support2_target, support3_target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:

            img, support_img, support_img2, support_img3, target, support_target, support2_target, support3_target = self.preproc((img, support_img, support_img2, support_img3), (target, support_target, support2_target, support3_target), self.input_dim)

        return np.concatenate((img, support_img, support_img2, support_img3), axis=0), (target, support_target, support2_target, support3_target), img_info, img_id
