import os
import cv2
import torch
import numpy as np
from torch.backends import cudnn
from efficientdet.backbone import EfficientDetBackbone
from attack_utils.attackloss_efdet import ada_attack, L2_attack
from attack_utils.GetMask import InitialMask, SampleMask, SFinalMask
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
compound_coef = 0           # 模型等级，越高效果越好
force_input_size = None     # 设置为None，默认使用指定图片大小
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]            # anchor比例
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]    # anchor尺寸
threshold = 0.3             # 存在前景目标的最小置信度
iou_threshold = 0.2         # NMS 参数IOU
use_cuda = True             # 使用cuda
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
# 构建模型
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=90,
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load('models/efficientdet-d{}.pth'.format(compound_coef)))
#model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]                   # 输入大小设置
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
framed_metas = [(512,512,500,500,0,0)]
# 循环攻击目录中的每张图片
clean_path = 'select1000_new/'  # 干净图片目录
dirty_path = 'select1000_new_p/'  # 对抗图片存放位置
imgs_list = os.listdir(clean_path)
for i in range(len(imgs_list)):
    image_name = os.path.basename(imgs_list[i]).split('.')[0]  # 测试图片名称
    print('It is attacking on the {}-th image, the image name is {}'.format(i, image_name))
    image_path = os.path.join(clean_path, imgs_list[i])
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #mask = np.load('InitialMask/{}.npy'.format(image_name))
    mask = np.zeros([500,500,3])
    flag, mask = InitialMask(mask)
    finalimg, kmaps= L2_attack(model, img_rgb, max_iter=300, epsilon=200, mask=mask)  # epsilon=10->0.01, 100->0.1
    mask = SFinalMask(mask,kmaps)
    kk = kmaps*mask
    np.save('Mask/NOISE/efdet/{}.npy'.format(image_name), kk)
print('done...')