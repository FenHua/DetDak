import os
import cv2
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

resize_big = transforms.Compose([transforms.Resize((608, 608))])
resize_small = transforms.Compose([transforms.Resize((500, 500))])


def single_attak_loss(output, conf_thresh, num_classes,num_anchors, only_objectness=1):
    # output 是不同大小的feature map（每次输入一层feature map）
    if len(output.shape) == 3:
        output = np.expand_dims(output, axis=0)
    batch = output.shape[0]  # patch数量，默认为1
    assert (output.shape[1] == (5 + num_classes) * num_anchors)
    h = output.shape[2]   # feature map 的宽
    w = output.shape[3]   # feature map 的高 (1, 0, 2)
    output = output.reshape(batch * num_anchors, 5 + num_classes, h * w).transpose(1, 0).reshape(
        5 + num_classes,
        batch * num_anchors * h * w)   # 将 feature map 转换为（80+5，*）
    det_confs = torch.sigmoid(output[4])    # 当前feature map该点处存在目标的概率 sigmoid(output[4])
    loss = 0.0
    idx = np.where((det_confs[:]).cpu().data>conf_thresh)
    loss += torch.sum(det_confs[idx])
    return loss


def total_loss(model, img, conf_thresh, use_cuda=1):
    model.eval()
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    elif type(img) == torch.Tensor and len(img.shape) == 4:
        img = img
    else:
        print("unknow image type")
        exit(-1)
    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img, requires_grad=True)
    list_boxes = model(img)
    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # 将三个feature map的loss求和
    loss = single_attak_loss(list_boxes[0], conf_thresh, 80, len(anchor_masks[0])) + \
           single_attak_loss(list_boxes[1], conf_thresh, 80, len(anchor_masks[1])) + single_attak_loss(list_boxes[2],
                                                                                                       conf_thresh, 80,
                                                                                                       len(anchor_masks[
                                                                                                               2]))
    return img, loss


def gen_attack(model, img, conf_thresh, norm_ord, max_iter, epsilon, mask):
    noise = np.zeros([608, 608, 3])  # 设置为原图大小
    Kmaps = np.zeros([500,500,3])
    finalimg = Image.fromarray(np.uint8(img))  # 500*500*3
    new_input = resize_big(finalimg)
    for iter_n in range(max_iter):
        im, loss = total_loss(model, new_input, conf_thresh, use_cuda=1)
        if loss>0:
            loss.backward()
            grad = im.grad
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
            return finalimg,Kmaps
        NP_P = normalized_grad * epsilon  # 整个输入图片的单次扰动
        noise += NP_P.squeeze().transpose(1,2,0)  # [608,608,3]
        NOISE = cv2.resize(noise,(500,500))*mask
        temp_img = np.clip((img - NOISE), 0, 255)  # 改为int32
        Kmaps = img - temp_img
        finalimg = Image.fromarray(np.uint8(temp_img))
        new_input = resize_big(finalimg)
    return finalimg,Kmaps