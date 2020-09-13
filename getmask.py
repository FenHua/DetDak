import os
import sys
sys.path.append('../mmdetection/')
from mmdet import __version__
from mmdet.apis import prepare_data    # 检测器
import cv2
import mmcv
import torch
import numpy as np

# 此脚本用于解决faster噪声图过于分散问题，通过单个proposal的梯度一次性回传限定一部分选块区域（最终没使用，自动忽略）


def Norm_Sel_Mask(NOISE, num):
    # 方便多个结果的融合
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
    NOISE = np.abs(NOISE)
    #根据像素绝对扰动量大小排序
    for i in range(N):
        x = int(i/flag_n)*DR
        y = int(i%flag_n)*DR
        idx[i] = np.mean(NOISE[x:(x+DR),y:(y+DR),:])
    idx_sort = np.argsort(-idx)   # 返回从小到大的下坐标
    result =[]
    for t in range(num):
        result.append(idx_sort[t])
    return result


def singlemask(img,model):
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
    # 计算每个目标最多可以贴几个种子块（将三分之二的块设置为种子块）
    object_num = np.min([len(index),10])
    Kflag = []
    for i in range(object_num):
        data = prepare_data(model, img)       # tuple类型
        data['img'][0].requires_grad=True     # 需要可导
        noise = np.zeros([800, 800, 3])
        result = model(return_loss=False, rescale=True,**data)   # 将rpn去掉
        loss = result[index[i][0]][index[i][1],4]
        loss.backward()
        grad = data['img'][0].grad
        grad_np = grad.data.cpu().numpy()
        normalized_grad = grad_np / np.sqrt(np.sum(grad_np * grad_np, axis=(2, 3), keepdims=True))  # l2
        NP_P = normalized_grad * 600 # 整个输入图片的单次扰动
        noise += NP_P.squeeze().transpose(1,2,0)  # [512,512,3]
        NOISE = cv2.resize(noise,(500,500))
        Kflag.append(Norm_Sel_Mask(NOISE, 20))
    return Kflag


def InitialMask(img, model):
    m = 500
    n = 500
    Mask = np.zeros([500,500,3])
    P_pixels = m*n*0.02       # 总共可以扰动的像素数
    radius = int(0.5*np.sqrt(P_pixels/10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2*radius   # 块的宽（正方形）
    flag_m = int(m/DR)
    flag_n = int(n/DR)
    Kflag = singlemask(img=img, model=model)
    for i in range(len(Kflag)):
        for j in range(len(Kflag[i])):
            x = int(Kflag[i][j] / flag_n) * DR
            y = int(Kflag[i][j] % flag_n) * DR
            Mask[x:(x + DR), y:(y + DR), :] = 1.0
    return Kflag, Mask


def TopTen(NOISE,Kflag,ratio):
    # 方便多个结果的融合
    m = 500  # 得到 mask 的长宽
    n = 500
    P_pixels = m * n * 0.02  # 总共可以扰动的像素数
    radius = int(0.5 * np.sqrt(P_pixels / 10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2 * radius  # 块的宽（正方形）
    flag_m = int(m / DR)
    flag_n = int(n / DR)
    N = flag_n * flag_m  # 块儿数
    idx = np.zeros(N)  # 用来记录每个块的重要程度，越大越不重要
    # 首先将noise取绝对值后，求和，归一化
    NOISE = np.abs(NOISE)
    # 根据像素绝对扰动量大小排序
    for i in range(N):
        x = int(i / flag_n) * DR
        y = int(i % flag_n) * DR
        idx[i] = np.mean(NOISE[x:(x + DR), y:(y + DR), :])
    idx_sort = np.argsort(-idx)  # 返回从小到大的下坐标
    idx_sort = idx_sort[:10]     # 只考虑噪声大小的10个块
    finalidx = []                # 最终选择块的idx
    kn = np.max([1,int(10*ratio/len(Kflag))])
    for i in range(len(Kflag)):
        dd = np.zeros(len(Kflag[i]))  # 记录当前目标各种子块的重要程度
        for j in range(len(Kflag[i])):
            dd[j] = idx[Kflag[i][j]]
        tt_sort = np.argsort(-dd)   # 对当前目标各种子块的重要程度pl
        for t in range(kn):
            # 每个目标保留最可能的kn个种子块
            finalidx.append(Kflag[i][tt_sort[t]])
    # 根据所有块的重要程度，添加块，直到总的块的数量达到10个
    finalidx = list(set(finalidx))    # 提分！！！！！！！！！！！！！
    for j in range(10):
        if idx_sort[j] in finalidx:
            continue
        else:
            finalidx.append(idx_sort[j])
        if(len(finalidx)==10):
            break
    # 更新mask
    mask = np.zeros([500,500,3])
    for j in range(10):
        x = int(finalidx[j] / flag_n) * DR
        y = int(finalidx[j] % flag_n) * DR
        mask[x:(x + DR), y:(y + DR), :] = 1.0
    return mask