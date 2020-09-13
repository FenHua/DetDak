import os
import cv2
import sys
import torch
import numpy as np
from util_copy.utils import post_YAN
from efficientdet.utils import BBoxTransform, ClipBoxes
resize = torch.nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

def scores_loss(model,img):
    # img tensor类型，cuda
    img = img / 255
    img = img.unsqueeze(0).permute(0, 3, 1, 2)
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    for i in range(3):
        img[:, i, :, :] -= mean[i]
        img[:, i, :, :] /= std[i]
    x = resize(img)
    features, regression, classification, anchors = model(x)  # 推理结果
    regressBoxes = BBoxTransform()  # box转换器
    clipBoxes = ClipBoxes()  # box过滤函数
    # 检测结果的后期处理
    scores = post_YAN(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      0.25, 0.2)
    if len(scores)>0:
        loss = torch.sum(scores)    # 取200归一化
    else:
        loss =0.0
    return loss

def filter(NOISE):
    m = 500  # 得到 mask 的长宽
    n = 500
    P_pixels = m*n*0.02       # 总共可以扰动的像素数
    radius = int(0.5*np.sqrt(P_pixels/10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2*radius   # 块的宽（正方形）
    flag_m = int(m/DR)
    flag_n = int(n/DR)
    N = flag_n*flag_m  #块儿数
    idx = np.zeros(N)   # 用来记录每个块的重要程度，越大越不重要
    #根据像素绝对扰动量大小排序
    for i in range(N):
        x = int(i/flag_n)*DR
        y = int(i%flag_n)*DR
        idx[i] = np.mean(np.abs(NOISE[x:(x+DR),y:(y+DR),:]))
    idx_sort = np.argsort(-idx)   # 返回从小到大的下坐标
    # 重新 设置flag以及mask
    mask = np.zeros([500,500,3])
    for j in range(5):
        x = int(idx_sort[j] / flag_n) * DR
        y = int(idx_sort[j] % flag_n) * DR
        mask[x:(x + DR), y:(y + DR),:] = 1.0
    return mask

def L2_attack(model, img, max_iter, epsilon, mask):
    # 输入为RGB图像，返回攻击成功的照片，以及对应的关键块映射图
    img = np.array(img, dtype=np.float32)
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.cuda()
    img_tensor.requires_grad =True
    P = [0.58395, 0.5712, 0.57375]
    Kmaps = np.zeros([500,500,3])
    noise = np.zeros([500, 500,3])
    for iter_n in range(max_iter):
        loss = scores_loss(model, img_tensor)
        '''
        if iter_n==0:
            min_loss = loss    # 当前最小的loss
        if (loss <min_loss):
            score = ((min_loss-loss)/min_loss).item()
            min_loss = loss
            Kmaps += normalized_grad * mask * epsilon* score    # 加上前一次的噪声
       '''
        #print(loss)
        if loss>0:
            loss.backward()
            grad = img_tensor.grad
            grad_np = grad.data.cpu().numpy()
            normalized_grad = grad_np / np.sqrt(np.sum(grad_np * grad_np, axis=(0, 1), keepdims=True))  # l2
        else:
            return finalimg, Kmaps
        #mask = filter(normalized_grad * epsilon)
        noise += normalized_grad * epsilon * mask *P # 整个输入图片的单次扰动 [500,500,3]
        finalimg = np.clip((img - noise), 0, 255)  # 改为int32
        Kmaps = img - finalimg
        finalimg = np.uint8(finalimg)
        img_tensor.data = torch.from_numpy(np.array(finalimg, dtype=np.float32)).cuda()
    return finalimg, Kmaps

def ada_attack(model, img, max_iter, epsilon, mask):
    # 输入为RGB图像
    img = np.array(img, dtype=np.float32)  # 先转换为float32
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.cuda()
    img_tensor.requires_grad = True
    last_loss=[]
    P = [0.58395, 0.5712, 0.57375]
    noise = np.zeros([500,500,3])
    finalimg = img
    for iter_n in range(max_iter):
        loss = scores_loss(model, img_tensor)
        #print(loss)
        if loss>0:
            loss.backward()
            grad = img_tensor.grad
            grad_np = grad.data.cpu().numpy()
            normalized_grad = np.sign(grad_np)
        else:
            return finalimg,NOISE
        last_loss.append(loss.item())
        temp_list = last_loss[-5:]
        if temp_list[-1]>temp_list[0]:
            if epsilon > 2:
                epsilon = epsilon-1.0
            else:
                epsilon = 1.0
        noise += normalized_grad * epsilon  # 整个输入图片的单次扰动
        NOISE = noise * mask
        finalimg = np.clip((img - NOISE * P), 0, 255)  # 改为int32
        finalimg = np.uint8(finalimg)
        img_tensor.data = torch.from_numpy(np.array(finalimg, dtype=np.float32)).cuda()
    return finalimg,NOISE