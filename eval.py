import os
import cv2
import json
import torch
import argparse
import numpy as np
from db import Transform, MVTEC
from factory import load_test_model_from_factory
from tools import *


def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised defect segmentaion base on auto-encoder.')
    parser.add_argument('--cfg', help="Path of config file", type=str, required=True)
    parser.add_argument('--model_path', help="Path of model", type=str, required=True)
    parser.add_argument('--gpu_id', help="ID of GPU", type=int, default=0)
    parser.add_argument('--res_dir', help="Directory path of result", type=str, default='./eval_result')
    parser.add_argument('--retest', default=False, type=bool)

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

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)

    # load data set
    test_set = MVTEC(root=configs['db']['data_dir'], set=configs['db']['test_split'], preproc=None)
    print('Data set: {} has been loaded'.format(configs['db']['name']))

    # load model
    transform = Transform(tuple(configs['db']['resize']))
    net = load_test_model_from_factory(configs)
    load_params(net, args.model_path)
    net = net.eval().cuda(args.gpu_id)
    print('Model: {} has been loaded'.format(configs['model']['name']))

    print('Start Testing... ')
    _t = Timer()
    cost_time = list()
    for item in test_set.test_dict:
        item_dict = test_set.test_dict[item]
        if not os.path.exists(os.path.join(args.res_dir, item)):
            os.mkdir(os.path.join(args.res_dir, item))
        for type in item_dict:
            if not os.path.exists(os.path.join(args.res_dir, item, type)):
                os.mkdir(os.path.join(args.res_dir, item, type))

            _time = list()
            img_list = item_dict[type]
            for path in img_list:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                _t.tic()
                ori_img, input_tensor = transform(image)
                with torch.no_grad():
                    input_tensor = input_tensor.cuda(args.gpu_id)
                    re_img = net(input_tensor)

                # project to BGR image
                re_img = re_img * 255
                re_img = re_img.cpu().numpy()[0]
                re_img = re_img.astype(np.uint8)
                re_img = re_img.transpose((1, 2, 0))
                inference_time = _t.toc()
                _time.append(inference_time)
                img_id = path.split('.')[0][-3:]
                cv2.imwrite(os.path.join(args.res_dir, item, type, 'ori_{}.png'.format(img_id)), ori_img)
                cv2.imwrite(os.path.join(args.res_dir, item, type, 're_{}.png'.format(img_id)), re_img)
            cost_time += _time
            mean_time = np.array(_time).mean()
            print('Evaluate: Item:{}; Type:{}; Mean time:{:.1f}ms'.format(item, type, mean_time * 1000))
            _t.clear()
    # calculate mean time
    cost_time = np.array(cost_time)
    cost_time = np.sort(cost_time)
    num = cost_time.shape[0]
    num90 = int(num * 0.9)
    cost_time = cost_time[0:num90]
    mean_time = np.mean(cost_time)
    print('Mean_time: {:.1f}ms'.format(mean_time * 1000))