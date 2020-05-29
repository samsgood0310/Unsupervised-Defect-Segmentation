import cv2
import numpy as np
import torch
from .mvtec import MVTEC
from .mvtec import Preproc as MVTEC_pre


def training_collate(batch):
    """Custom collate fn for dealing with batches of images.

    Arguments:
        batch: (tuple) A tuple of tensor images

    Return:
        (tensor) batch of images stacked on their 0 dim
    """
    # imgs = list()
    # for img in batch:
    #     _c, _h, _w = img.shape
    #     imgs.append(img.view(1, _c, _h, _w))

    return torch.stack(batch, 0)


class Test_Transform(object):
    def __init__(self):
        pass

    def __call__(self, image, IsTexture):
        _h, _w = image.shape[0: 2]
        image = image.astype(np.float32) / 255.
        if IsTexture is True:
            crop_h = int(_h / 2)
            crop_w = int(_w / 2)
            crop_list = []
            crop_list.append(torch.from_numpy(image[0:crop_h, 0: crop_w]))
            crop_list.append(torch.from_numpy(image[crop_h:_h, 0: crop_w]))
            crop_list.append(torch.from_numpy(image[0:crop_h, crop_w: _w]))
            crop_list.append(torch.from_numpy(image[crop_h:_h, crop_w: _w]))
            img_tensor = torch.stack(crop_list, 0)
            img_tensor = img_tensor.unsqueeze(1)
        else:
            img_tensor = torch.from_numpy(image)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.unsqueeze(1)

        return img_tensor

