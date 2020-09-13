# coding=utf-8
import os
import sys
import numpy as np
from util_copy.utils import *
from torchvision import transforms
from tool.darknet2pytorch import *
from attack_utils.attackloss_yolo import gen_attack
from attack_utils.GetMask import SFinalMask, InitialMask
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cfgfile = "models/yolov4.cfg"
weightfile = "models/yolov4.weights"
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
darknet_model = darknet_model.eval().cuda()

resize_big = transforms.Compose([
            transforms.Resize((608, 608))])
# 循环攻击目录中的每张图片
clean_path = 'select1000_new/'  # 干净图片目录
imgs_list = os.listdir(clean_path)
for i in range(len(imgs_list)):
    image_name = os.path.basename(imgs_list[i]).split('.')[0]  # 测试图片名称
    print('It is attacking on the {}-th image, the image name is {}'.format(i, image_name))
    image_path = os.path.join(clean_path, imgs_list[i])
    img0 = Image.open(image_path).convert('RGB')
    img = np.array(img0, dtype=np.int)
    mask = np.zeros([500,500,3])  # 格式为 1*3*M*N（大小与输入图片一样）
    flag, mask = InitialMask(mask)
    # 内部迭代5次，选出效果最好的mask
    #mask = np.load('InitialMask/{}.npy'.format(image_name))
    _, noise = gen_attack(darknet_model, img, conf_thresh=0.4, norm_ord='L2', max_iter=100, epsilon=5, mask=mask)
    mask = SFinalMask(mask, noise)
    kk = noise*mask
    np.save('Mask/NOISE/yolo/{}.npy'.format(image_name), kk)
print('done...')