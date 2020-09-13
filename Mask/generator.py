import os
import numpy as np


def InitialMask(Mask):
    m,n,_ = np.shape(Mask)    # 得到 mask 的长宽
    P_pixels = m*n*0.02       # 总共可以扰动的像素数
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

def Norm_Sel_Mask(flag, Mask, noise0, noise1, noise2):
    # 方便多个结果的融合
    m,n,_ = np.shape(Mask)  # 得到 mask 的长宽
    # 0.02 0.2 0.4
    P_pixels = m*n*0.5       # 总共可以扰动的像素数
    radius = int(0.5*np.sqrt(P_pixels/10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2*radius   # 块的宽（正方形）
    flag_m = int(m/DR)
    flag_n = int(n/DR)
    N = flag_n*flag_m  #块儿数
    idx = np.zeros(N)   # 用来记录每个块的重要程度，越大越不重要
    # 首先将noise取绝对值后，求和，归一化
    NOISE0 = np.abs(noise0)
    T0 = np.sum(NOISE0)
    Norm_NOISE0 = NOISE0/T0
    NOISE1 = np.abs(noise1)
    T1 = np.sum(NOISE1)
    Norm_NOISE1 = NOISE1/T1
    NOISE2 = np.abs(noise2)
    T2 = np.sum(NOISE2)
    Norm_NOISE2 = NOISE2/T2

    Norm_NOISE = Norm_NOISE0 + Norm_NOISE1 + Norm_NOISE2
    #根据像素绝对扰动量大小排序
    for i in range(N):
        if flag[i]<1.0:
            idx[i] = 0
        else:
            x = int(i/flag_n)*DR
            y = int(i%flag_n)*DR
            idx[i] = np.mean(Norm_NOISE[x:(x+DR),y:(y+DR),:])
    idx_sort = np.argsort(-idx)   # 返回从小到大的下坐标
    # 重新 设置flag以及mask
    f = np.zeros((flag_m * flag_n))
    mask = np.zeros_like(Mask)
    # 10
    for j in range(10):
        f[idx_sort[j]] = 1.0
        x = int(idx_sort[j] / flag_n) * DR
        y = int(idx_sort[j] % flag_n) * DR
        mask[x:(x + DR), y:(y + DR),:] = 1.0
    return f, mask


noise_root ='NOISE'
yolo_noise_root = 'NOISE/yolo/'
noise_list = os.listdir(yolo_noise_root)
for i in range(len(noise_list)):
    noise_name = os.path.basename(noise_list[i]).split('.')[0]   # 名称
    print('It is generating {}-th image mask, the image name is {}'.format(i,noise_name))
    yolo_noise = os.path.join(noise_root,'yolo',noise_list[i])
    rcnn_noise = os.path.join(noise_root, 'rcnn', noise_list[i])
    efdet_noise = os.path.join(noise_root, 'efdet', noise_list[i])
    Ynoise = np.load(yolo_noise)
    Rnoise = np.load(rcnn_noise)
    Enoise = np.load(efdet_noise)
    mask = np.zeros([500,500,3])
    flag, mask = InitialMask(mask)
    _, mask = Norm_Sel_Mask(flag,mask,noise0=Ynoise,noise1=Rnoise,noise2=Enoise)    # 生成mask
    #_, mask = Yolo_Rcnn_Mask(flag,mask,noise0=Ynoise,noise1=Enoise)                # 俩白盒生成mask
    #NOISE = GetMaps(noise0=Ynoise,noise1=Rnoise,noise2=Enoise)
    np.save('Mask/{}.npy'.format(noise_name),mask)
    #np.save('Kmaps/{}.npy'.format(noise_name), NOISE)
print('done')