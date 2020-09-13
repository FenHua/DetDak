# coding=utf-8
import cv2
import torch
import numpy as np


# mask初始化函数
def InitialMask(Mask):
    m,n,_ = np.shape(Mask)    # 得到 mask 的长宽
    P_pixels = m*n*0.02       # 总共可以扰动的像素数（可以进行调整，从而扩大所选块的面积）
    radius = int(0.5*np.sqrt(P_pixels/10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2*radius   # 块的宽（正方形）
    flag_m = int(m/DR)
    flag_n = int(n/DR)
    flag = np.zeros((flag_m*flag_n))   # 记录这些块的状态，初始时都是被选中的
    # 根据bboxes信息确定patch的大概位置（加速计算）
    for i in range(flag_m):
        for j in range(flag_n):
            # 计算每个块的起始位置
            start_x = i*DR
            start_y = j*DR
            Mask[start_x:(start_x+DR),start_y:(start_y+DR),:] = 1.0
            flag[(i*flag_n +j)] = 1.0
    return flag, Mask


# 二分法不断缩小最重要的块的数量
def SampleMask(flag, Mask, NOISE):
    m,n,_ = np.shape(Mask)        # 得到 mask 的长宽
    P_pixels = m*n*0.02           # 总共可以扰动的像素数
    radius = int(0.5*np.sqrt(P_pixels/10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2*radius   # 块的宽（正方形）
    flag_m = int(m/DR)
    flag_n = int(n/DR)
    N = flag_n*flag_m            #块儿数
    idx = np.zeros(N)            # 用来记录每个块的重要程度，越大越不重要
    #根据像素绝对扰动量大小排序
    for i in range(N):
        if flag[i]<1.0:
            idx[i] = 0
        else:
            x = int(i/flag_n)*DR
            y = int(i%flag_n)*DR
            idx[i] = np.mean(np.abs(NOISE[x:(x+DR),y:(y+DR),:]))
    idx_sort = np.argsort(-idx)   # 返回从小到大的下坐标
    # 重新 设置flag以及mask
    f = np.zeros((flag_m * flag_n))
    mask = np.zeros_like(Mask)
    half = int(0.5 * np.sum(flag))
    if half <= 10:
        for j in range(10):
            f[idx_sort[j]] = 1.0
            x = int(idx_sort[j] / flag_n) * DR
            y = int(idx_sort[j] % flag_n) * DR
            mask[x:(x + DR), y:(y + DR), :] = 1.0
    else:
        for j in range(half):
            f[idx_sort[j]] = 1.0
            x = int(idx_sort[j] / flag_n) * DR
            y = int(idx_sort[j] % flag_n) * DR
            mask[x:(x + DR), y:(y + DR),:] = 1.0
    return f, mask


# 一步法，根据噪声幅度图，一次性选出10个重要的块
def FinalMask(flag, Mask, NOISE):
    m,n,_ = np.shape(Mask)  # 得到 mask 的长宽
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
        if flag[i]<1.0:
            idx[i] = 0
        else:
            x = int(i/flag_n)*DR
            y = int(i%flag_n)*DR
            idx[i] = np.mean(np.abs(NOISE[x:(x+DR),y:(y+DR),:]))
    idx_sort = np.argsort(-idx)   # 返回从小到大的下坐标
    # 重新 设置flag以及mask
    f = np.zeros((flag_m * flag_n))
    mask = np.zeros_like(Mask)
    for j in range(10):
        f[idx_sort[j]] = 1.0
        x = int(idx_sort[j] / flag_n) * DR
        y = int(idx_sort[j] % flag_n) * DR
        mask[x:(x + DR), y:(y + DR),:] = 1.0
    return f, mask

def SFinalMask(Mask, NOISE):
    m,n,_ = np.shape(Mask)  # 得到 mask 的长宽
    # 0.02  0.2  0.4
    P_pixels = m*n*0.1      # 总共可以扰动的像素数（扩大10倍）
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
    mask = np.zeros_like(Mask)
    for j in range(10):
        x = int(idx_sort[j] / flag_n) * DR
        y = int(idx_sort[j] % flag_n) * DR
        mask[x:(x + DR), y:(y + DR),:] = 1.0
    return mask