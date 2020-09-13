# coding=utf-8
import os
import sys
sys.path.append('./mmdetection/')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from mmdet import __version__
import cv2
import numpy as np
from torchvision import transforms
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
from util_copy.utils import *
from tool.darknet2pytorch import *
import matplotlib.pyplot as plt
from attack_utils.attackloss import L2_attack,ada_attack
from attack_utils.GetMask import InitialMask, FinalMask
from mmdet.apis import init_detector, inference_detector

# yolo
cfgfile = "models/yolov4.cfg"
weightfile = "models/yolov4.weights"
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
darknet_model = darknet_model.eval().cuda()

#faster rcnn
config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
rcnn_model = init_detector(config, checkpoint, device='cuda:1')      # 构建 faster rcnn

resize_big = transforms.Compose([
            transforms.Resize((608, 608))])
resize_small = transforms.Compose([
            transforms.Resize((500, 500))])

img_path1 = 'select1000_new//14.png'
img = cv2.imread(img_path1)
mask = np.load('Mask/14.npy')
print('Starting attacking.....')
'''
mask = np.zeros([500,500,3])
flag, mask = InitialMask(mask)
_, noise = L2_attack(darknet_model,rcnn_model, img, conf_thresh=0.4, max_iter=100, epsilon=200, mask=mask)
flag,mask =FinalMask(flag, mask, noise)
'''
finalimg0, noise = ada_attack(darknet_model,rcnn_model, img, conf_thresh=0.4, max_iter=100, epsilon=6, mask=mask)
finalimg0.save('hh.png')
img7 = Image.open('hh.png').convert('RGB')
finalimg = resize_big(img7)
boxes = do_detect(darknet_model, finalimg, 0.5, 0.4, True)
#plot_boxes(finalimg0,boxes,savename='test.png')
# 攻击完成后对当前噪声图片的检测结果进行显示
#plt.imshow(noise*5)
#plt.show()
print(len(boxes))

result_clean = inference_detector(rcnn_model, img_path1)
result_dirty = inference_detector(rcnn_model, 'hh.png')
result_above_confidence_num_clean = 0
result_above_confidence_num_dirty = 0
for ir in range(len(result_clean)):
    for j in range(len(result_clean[ir])):
        if result_clean[ir][j,4]>0.3:
            result_above_confidence_num_clean = result_above_confidence_num_clean + 1
for ir in range(len(result_dirty)):
    for j in range(len(result_dirty[ir])):
        if result_dirty[ir][j,4]>0.3:
            result_above_confidence_num_dirty = result_above_confidence_num_dirty + 1
print(1 - np.min([result_above_confidence_num_dirty,result_above_confidence_num_clean])/result_above_confidence_num_clean)