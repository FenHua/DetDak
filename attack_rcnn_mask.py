# coding=utf-8
import os
import sys
sys.path.append('./mmdetection/')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector
from attack_utils.attackloss_rcnn import L2_attack, ada_attack
from attack_utils.GetMask import InitialMask,SFinalMask
from util_copy.utils import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config, checkpoint, device='cuda:0')      # 构建 faster rcnn


# 循环攻击目录中的每张图片
clean_path = 'select1000_new/'  # 干净图片目录
dirty_path = 'select1000_new_p/'  # 对抗图片存放位置
imgs_list = os.listdir(clean_path)

for i in range(len(imgs_list)):
    image_name = os.path.basename(imgs_list[i]).split('.')[0]  # 测试图片名称
    print('It is attacking on the {}-th image, the image name is {}'.format(i, image_name))
    image_path = os.path.join(clean_path, imgs_list[i])
    img = cv2.imread(image_path)
    mask = np.zeros([500,500,3])
    flag, mask = InitialMask(mask)
    # 开始执行刷选mask
    #mask = np.load('InitialMask/{}.npy'.format(image_name))
    _, noise = L2_attack(model, img, max_iter=100, epsilon=100, mask=mask) # L2 20->1
    mask = SFinalMask(mask, noise)
    kk = noise*mask
    #mask = np.load('attack_utils/Mask/{}.npy'.format(image_name))
    #finalimg, noise = ada_attack(model, img, max_iter=200, epsilon=4, mask=mask)
    #image_pert_path = os.path.join(dirty_path, imgs_list[i])
    #cv2.imwrite(image_pert_path, finalimg)
    np.save('Mask/NOISE/rcnn/{}.npy'.format(image_name),kk)
print('done...')