import os
import sys
sys.path.append('../mmdetection/')
from mmdet import __version__
from mmdet.apis import prepare_data    # 检测器
import cv2
import mmcv
import torch
import numpy as np


def GetObject(model, img):
    # 先获取有效分值的下坐标
    with torch.no_grad():
        data = prepare_data(model, img)       # tuple类型
        result = model(return_loss=False, rescale=True,**data)   # 将rpn去掉
        # 总共80个类别
        index = []
        for i in range(len(result)):
            for j in range(len(result[i])):
                if(result[i][j,4]>0.3):
                    index.append([i,j])
    return index

def L2_loss(model,data,idx):
    # 检测模型以及输入图片的路径
    result = model(return_loss=False, rescale=True,**data)   # 将rpn去掉
    # 总共80个类别
    loss_key = 0.0
    for i in range(len(idx)):
        if (len(result[idx[i][0]])>idx[i][1]):
            if(result[idx[i][0]][idx[i][1],4]>0.3):
                loss_key += result[idx[i][0]][idx[i][1],4]
    loss_other = 0.0
    for i in range(len(result)):
        for j in range(len(result[i])):
            if(result[i][j,4]>0.3):
                loss_other += result[i][j,4]
    loss = 0.9*loss_key + 0.1*loss_other
    return loss


def total_loss(model, data, show_score_thr=0.3):
    # 检测模型以及输入图片的路径
    result = model(return_loss=False, rescale=True,**data)   # 将rpn去掉
    # 总共80个类别
    loss = 0.0
    for i in range(len(result)):
        for j in range(len(result[i])):
            if(result[i][j,4]>show_score_thr):
                loss += result[i][j,4]
    return loss


def gen_attack(model, img, norm_ord, max_iter, epsilon, mask):
    P = [0.58395, 0.5712,0.57375]  #RGB
    # 数据处理过程先转换为RGB，然后resize到800，然后正则化
    data = prepare_data(model, img)       # tuple类型
    data['img'][0].requires_grad = True    # 需要可导
    noise = np.zeros([800, 800,3])        # 处理好图片的大小
    finalimg = img
    for iter_n in range(max_iter):
        loss= total_loss(model, data)
        #print(loss)
        if loss>0:
            loss.backward()
            grad = data['img'][0].grad
            grad_np = grad.data.cpu().numpy()
            if norm_ord == 'sign':
                normalized_grad = np.sign(grad_np)
            elif norm_ord == 'L1':
                normalized_grad = grad_np / np.sum(np.abs(grad_np), axis=(2, 3), keepdims=True)  # l1
            elif norm_ord == 'L2':
                normalized_grad = grad_np / np.sqrt(np.sum(grad_np * grad_np, axis=(2, 3), keepdims=True))  # l2
            else:
                raise ValueError('This norm_ord does not support...')
        else:
            return finalimg,NOISE
        NP_P = normalized_grad * epsilon        # 整个输入图片的单次扰动
        noise += NP_P.squeeze().transpose(1,2,0)   # [800,800,3]
        NOISE = cv2.resize(noise*P,(500,500))*mask
        NOISE = NOISE[:,:,::-1]  # 从RGB对应的数据转换为BGR
        finalimg = np.clip((img - NOISE), 0, 255)  # 改为int32
        finalimg = np.uint8(finalimg)
        temp_data = prepare_data(model,finalimg)
        data['img'][0].data = temp_data['img'][0].data
    return finalimg,NOISE

def L2_attack(model, img, max_iter, epsilon, mask):
    P = [0.58395, 0.5712,0.57375]  #RGB
    # 数据处理过程先转换为RGB，然后resize到800，然后正则化
    data = prepare_data(model, img)       # tuple类型
    data['img'][0] = torch.autograd.Variable(data['img'][0], requires_grad=True)   # 需要可导
    noise = np.zeros([500, 500,3])        # 处理好图片的大小
    finalimg = img
    NOISE = np.zeros([500,500,3])
    idx = GetObject(model, img)
    for iter_n in range(max_iter):
        loss = L2_loss(model, data,idx)
        '''
        if iter_n==0:
            min_loss = loss    # 当前最小的loss
        if (loss <min_loss):
            score = ((min_loss-loss)/min_loss).item()
            min_loss = loss
            NOISE += NP_P * mask * score    # 加上前一次的噪声
       '''
        #print(loss)
        if loss>0:
            loss.backward()
            grad = data['img'][0].grad
            grad_np = grad.data.cpu().numpy()
            normalized_grad = grad_np / np.sqrt(np.sum(grad_np * grad_np, axis=(2, 3), keepdims=True))  # l2
        else:
            return finalimg,NOISE
        NP_P = normalized_grad * epsilon        # 整个输入图片的单次扰动
        NP_P = NP_P.squeeze().transpose(1,2,0)
        NP_P = cv2.resize(NP_P,(500,500))
        noise += NP_P*mask  # [800,800,3]
        finalimg = np.clip((np.array(img, dtype = np.float) - noise[:,:,::-1]), 0, 255)  # 改为int32
        NOISE = np.array(img,dtype = np.float)- finalimg
        finalimg = np.uint8(finalimg)
        temp_data = prepare_data(model,finalimg)
        data['img'][0].data = temp_data['img'][0].data
    return finalimg,NOISE

def ada_attack(model, img, max_iter, epsilon, mask):
    P = [0.58395, 0.5712,0.57375]  #RGB
    # 数据处理过程先转换为RGB，然后resize到800，然后正则化
    data = prepare_data(model, img)       # tuple类型
    data['img'][0] = torch.autograd.Variable(data['img'][0], requires_grad=True)   # 需要可导
    noise = np.zeros([800, 800,3])        # 处理好图片的大小
    finalimg = img
    last_loss=[]
    for iter_n in range(max_iter):
        loss = total_loss(model, data)
        #print(loss)
        if loss>0:
            loss.backward()
            grad = data['img'][0].grad
            grad_np = grad.data.cpu().numpy()
            normalized_grad = np.sign(grad_np)
        else:
            return finalimg,NOISE
        last_loss.append(loss.item())
        temp_list = last_loss[-5:]
        if temp_list[-1]>temp_list[0]:
            if epsilon > 1:
                epsilon = epsilon-1.0
            else:
                epsilon = 1.0
        NP_P = normalized_grad * epsilon        # 整个输入图片的单次扰动
        noise += NP_P.squeeze().transpose(1,2,0)   # [800,800,3]
        NOISE = cv2.resize(noise*P,(500,500))*mask
        NOISE = NOISE[:,:,::-1]  # 从RGB对应的数据转换为BGR
        finalimg = np.clip((img - NOISE), 0, 255)  # 改为int32
        finalimg = np.uint8(finalimg)
        temp_data = prepare_data(model,finalimg)
        data['img'][0].data = temp_data['img'][0].data
    return finalimg,NOISE
