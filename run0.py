# coding=utf-8
import os
import sys
sys.path.append('./mmdetection/')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from mmdet.apis import init_detector
from util_copy.utils import *
from tool.darknet2pytorch import *
from attack_utils.attackloss import ada_attack

# yolo
cfgfile = "models/yolov4.cfg"
weightfile = "models/yolov4.weights"
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
darknet_model = darknet_model.eval().cuda()

# faster rcnn
config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
rcnn_model = init_detector(config, checkpoint, device='cuda:1')  # 构建 faster rcnn

# 循环攻击目录中的每张图片
clean_path = 'select1000_new/'  # 干净图片目录
dirty_path = 'select1000_new_p/'  # 对抗图片存放位置
imgs_list = os.listdir(clean_path)
imgs_list.sort()


# 网格稀疏函数
def get_mesh(mask, threshold=2000):
    cnt = np.sum(mask) / 3
    stride = 1
    f = lambda x: (2 * x - 1) / (x * x)
    while cnt * f(stride) >= threshold:
        stride += 1
    mesh = np.zeros((500, 500, 3))
    mesh[stride//2::stride, :, :] = 1
    mesh[:, stride//2::stride, :] = 1
    return mesh


median = int(0.5*len(imgs_list))
# median
for i in range(1):
    image_name = os.path.basename(imgs_list[i]).split('.')[0]  # 测试图片名称
    print('It is attacking on the {}-th image, the image name is {}'.format(i, image_name))
    image_path = os.path.join(clean_path, imgs_list[i])
    img = cv2.imread(image_path)
    mask = np.load('Mask/Mask/{}.npy'.format(image_name))           # 读取已经生成的mask
    # 对mask进行网格稀疏化
    # mesh = get_mesh(mask)
    # mask = mask * mesh
    # 简单的梯度符号法对两个融合后的白盒模型进行攻击
    finalimg, noise = ada_attack(darknet_model, rcnn_model, img, conf_thresh=0.3, max_iter=150, epsilon=6, mask=mask)
    image_pert_path = os.path.join(dirty_path, imgs_list[i])
    finalimg.save(image_pert_path)
print('done...')
