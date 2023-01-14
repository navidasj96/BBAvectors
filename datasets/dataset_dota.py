from .base import BaseDataset
import os
import cv2
import numpy as np
from datasets.dotadevkit.dotadevkit.ops.ResultMerge import mergebypoly
import pickle
class DOTA(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(DOTA, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['plane',
                         'baseball-diamond',
                         'bridge',
                         'ground-track-field',
                         'small-vehicle',
                         'large-vehicle',
                         'ship',
                         'tennis-court',
                         'basketball-court',
                         'storage-tank',
                         'soccer-ball-field',
                         'roundabout',
                         'harbor',
                         'swimming-pool',
                         'helicopter'
                         ]
        self.color_pans = [(204,78,210),
                           (0,192,255),
                           (0,131,0),
                           (240,176,0),
                           (254,100,38),
                           (0,0,255),
                           (182,117,46),
                           (185,60,129),
                           (204,153,255),
                           (80,208,146),
                           (0,0,204),
                           (17,90,197),
                           (0,255,255),
                           (102,255,102),
                           (255,255,0)]
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        if self.phase == 'train':
            image_set_index_file = os.path.join(self.data_dir, 'trainval.txt')
        else:
            image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.png')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.pkl')

    def load_annotation(self, index):

        pkl_file = open(self.load_annoFolder(self.img_ids[index]), 'rb')
        annotation = pickle.load(pkl_file)
        pkl_file.close()
            
        return annotation


    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)
'''
def load_image(self, index):
        img_id = self.img_ids[index]
        if os.path.exists(os.path.join(self.image_path, img_id+'.jpg')):
          imgFile = os.path.join(self.image_path, img_id+'.jpg')
        elif os.path.exists(os.path.join(self.image_path, img_id+'.png')):
          imgFile = os.path.join(self.image_path, img_id+'.png')
        else:
          assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

        '''
