from __future__ import print_function
import numpy as np
import torch
import cv2
import yaml
import os
from torch.autograd import Variable
from models.networks import get_generator
import torchvision
import time
import argparse

def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--weights_path', required=True, help='Weights path')
	return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    with open('config/config_Stripformer_gopro.yaml') as cfg:
        config = yaml.safe_load(cfg)
    blur_path = './datasets/HIDE/blur/'
    out_path = './out/Stripformer_HIDE_results'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model = get_generator(config['model'])
    model.load_state_dict(torch.load(args.weights_path))
    model = model.cuda()
    test_time = 0
    iteration = 0
    total_image_number = 2025

    # warm up
    warm_up = 0
    print('Hardware warm-up')
    for img_name in os.listdir(blur_path):
        warm_up += 1
        img = cv2.imread(blur_path + '/' + img_name)
        img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
        with torch.no_grad():
            img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()
            result_image = model(img_tensor)
        if warm_up == 20:
            break
        break

    for img_name in os.listdir(blur_path):
        img = cv2.imread(blur_path + '/' + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
        with torch.no_grad():
            iteration += 1
            img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()

            start = time.time()
            result_image = model(img_tensor)
            stop = time.time()

            print('Image:{}/{}, CNN Runtime:{:.4f}'.format(iteration, total_image_number, (stop - start)))
            test_time += stop - start
            print('Average Runtime:{:.4f}'.format(test_time / float(iteration)))
            result_image = result_image + 0.5
            out_file_name = out_path + '/' + img_name
            torchvision.utils.save_image(result_image, out_file_name)