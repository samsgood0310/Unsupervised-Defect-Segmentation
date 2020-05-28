"""Data set tool of MVTEC

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import os
import re
import cv2
import torch
import numpy as np
import torch.utils.data as data
from collections import OrderedDict
from .augment import *
from .eval_func import *


class Preproc(object):
    """Pre-procession of input image includes resize, crop & data augmentation

    Arguments:
        resize: tup(int width, int height): resize shape
        crop: tup(int width, int height): crop shape
    """
    def __init__(self, resize):
        self.resize = resize

    def __call__(self, image):
        if self.resize is not None:
            image = cv2.resize(image, self.resize)
        # random transformation
        # p = random.uniform(0, 1)
        # if (p > 0.33) and (p <= 0.66):
        #     image = mirror(image)
        # else:
        #     image = flip(image)
        # # light adjustment
        # p = random.uniform(0, 1)
        # if p > 0.5:
        #     image = lighting_adjust(image, k=(0.95, 1.05), b=(-10, 10))

        # image normal
        image = image.astype(np.float32) / 255.
        # normalize_(tile, self.mean, self.std)
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)


class MVTEC(data.Dataset):
    """A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection

    Arguments:
        root (string): root directory to mvtec folder.
        set (string): image set to use ('train', or 'test')
        preproc(callable, optional): pre-procession on the input image
    """

    def __init__(self, root, set, preproc=None):
        self.root = root
        self.preproc = preproc
        self.set = set

        if set == 'train':
            self.ids = list()
            for _item in os.listdir(root):
                item_path = os.path.join(root, _item)
                if os.path.isfile(item_path):
                    continue
                img_dir = os.path.join(item_path, set, 'good')
                for img in os.listdir(img_dir):
                    self.ids.append(os.path.join(img_dir, img))
        elif set == 'test':
            self.test_len = 0
            self.test_dict = OrderedDict()
            for _item in os.listdir(root):
                item_path = os.path.join(root, _item)
                if os.path.isfile(item_path):
                    continue
                self.test_dict[_item] = OrderedDict()
                type_dir = os.path.join(item_path, set)
                for type in os.listdir(type_dir):
                    img_dir = os.path.join(item_path, set, type)
                    ids = list()
                    for img in os.listdir(img_dir):
                        if re.search('.png', img) is None:
                            continue
                        ids.append(os.path.join(img_dir, img))
                        self.test_len += 1
                    self.test_dict[_item][type] = ids
        else:
            raise Exception("Invalid set name")

    def __getitem__(self, index):
        """Returns training image
        """
        img_path = self.ids[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_h, img_w = img.shape[0: 2]
        img = np.reshape(img, [img_h, img_w, 1])
        if self.preproc is not None:
            img = self.preproc(img)

        return img

    def __len__(self):
        if self.set == 'train':
            return len(self.ids)
        else:
            return self.test_len