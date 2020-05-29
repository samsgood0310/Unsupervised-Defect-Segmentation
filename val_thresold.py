import os
import cv2
import json
import torch
import argparse
import numpy as np
from db import Test_Transform, MVTEC
from factory import load_test_model_from_factory
from model import ssim_seg


def parse_args():
    parser = argparse.ArgumentParser(description='Estimate the threshold from  randomly selected validation images.')
    parser.add_argument('--cfg', help="Path of config file", type=str, required=True)
    parser.add_argument('--model_path', help="Path of model", type=str, required=True)
    parser.add_argument('--gpu_id', help="ID of GPU", type=int, default=0)

    return parser.parse_args()


def load_params(net, path):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    w_dict = torch.load(path)
    for k, v in w_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)


if __name__ == '__main__':
    args = parse_args()

    # load config file
    cfg_file = os.path.join('./config', args.cfg + '.json')
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    # load data set
    val_set = MVTEC(root=configs['db']['data_dir'], set=configs['db']['val_split'], preproc=None)
    transform = Test_Transform()
    print('Data set: {} has been loaded'.format(configs['db']['name']))

    # load model
    net = load_test_model_from_factory(configs)
    load_params(net, args.model_path)
    net = net.eval().cuda(args.gpu_id)
    print('Model: {} has been loaded'.format(configs['model']['name']))

    # start validation
    for item in val_set.val_dict:
        img_list = val_set.val_dict[item]
        for img_info in img_list:
            path = img_info[0]
            IsTexture = img_info[1]
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_id = path.split('.')[0][-3:]

            input_tensor = transform(image, IsTexture)
            with torch.no_grad():
                input_tensor = input_tensor.cuda(args.gpu_id)
                re_img = net(input_tensor)

            # fetech from GPU
            re_img = torch.squeeze(re_img)
            re_img = re_img.cpu().numpy()

            # projected to Grayscale image
            re_img = re_img * 255
            re_img = re_img.astype(np.uint8)

            # segmentation
            if IsTexture is True:
                ori_img = list()
                ori_img[0] = image[0:128, 0:128]
                ori_img[1] = image[128:256, 0:128]
                ori_img[2] = image[0:128, 128:256]
                ori_img[3] = image[128:256, 128:256]
            else:
                s_mask = ssim_seg(image, re_img, win_size=11, threshold=0.5, gaussian_weights=True)
                cv2.imwrite('./tmp/{}_re_{}.png'.format(item, img_id), re_img)
                cv2.imwrite('./tmp/{}_ori_{}.png'.format(item, img_id), image)
                cv2.imwrite('./tmp/{}_mask_{}.png'.format(item, img_id), s_mask)
                pass