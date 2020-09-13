import sys
sys.path.append('../mmdetection/')
from mmdet.apis import prepare_data  # 检测器
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
resize_big = transforms.Compose([transforms.Resize((608, 608))])


def yolo_single_loss(output, conf_thresh, num_classes, num_anchors, only_objectness=1):
    # output 是不同大小的feature map（每次输入一层feature map）
    if len(output.shape) == 3:
        output = np.expand_dims(output, axis=0)
    batch = output.shape[0]  # patch数量，默认为1
    assert (output.shape[1] == (5 + num_classes) * num_anchors)
    h = output.shape[2]  # feature map 的宽
    w = output.shape[3]  # feature map 的高 (1, 0, 2)
    output = output.reshape(batch * num_anchors, 5 + num_classes, h * w).transpose(1, 0).reshape(
        5 + num_classes,
        batch * num_anchors * h * w)  # 将 feature map 转换为（80+5，*）
    det_confs = torch.sigmoid(output[4])  # 当前feature map该点处存在目标的概率 sigmoid(output[4])
    loss = 0.0
    idx = np.where((det_confs[:]).cpu().data > conf_thresh)
    loss += torch.sum(det_confs[idx])
    return loss


def yolo_loss(model, img, conf_thresh, use_cuda=1):
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
    img.requires_grad = True
    list_boxes = model(img)
    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # 将三个feature map的loss求和
    loss = yolo_single_loss(list_boxes[0], conf_thresh, 80, len(anchor_masks[0])) + \
           yolo_single_loss(list_boxes[1], conf_thresh, 80, len(anchor_masks[1])) + yolo_single_loss(list_boxes[2],
                                                                                                     conf_thresh, 80,
                                                                                                     len(anchor_masks[
                                                                                                             2]))
    loss = loss / 3.0
    return img, loss


def rcnn_loss(model, data, show_score_thr=0.3):
    result = model(return_loss=False, rescale=True, **data)  # 将rpn去掉
    # 总共80个类别
    loss = 0.0
    for i in range(len(result)):
        for j in range(len(result[i])):
            if (result[i][j, 4] > show_score_thr):
                loss += result[i][j, 4]
    return loss


def GetObject(model, img):
    # 先获取faster_rcnn有效分值的下坐标
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


def rcnn_L2_loss(model,data,idx):
    # 检测模型以及输入图片的路径
    result = model(return_loss=False, rescale=True,**data)   # 此处非rpn
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


def L2_attack(yolo_model, rcnn_model, img, conf_thresh,max_iter, epsilon,mask):
    P = [0.58395, 0.5712,0.57375]  #RGB
    # yolo 部分
    noise_yolo = np.zeros([608, 608, 3])  # 设置为原图大小
    # 将输入图片转换到PIL格式
    finalimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 500*500*3  RGB
    new_input = resize_big(finalimg)
    # faster rcnn部分
    idx = GetObject(rcnn_model, img)
    data = prepare_data(rcnn_model, img)  # tuple类型
    data['img'][0].requires_grad = True    # 需要可导
    noise_rcnn = np.zeros([800, 800, 3])  # 处理好图片的大小
    for iter_n in range(max_iter):
        rloss = rcnn_L2_loss(rcnn_model,data,idx)
        im, yloss = yolo_loss(yolo_model, new_input, conf_thresh, use_cuda=1)
        tloss = 0.5*yloss+0.5*rloss
        print(tloss.item())
        if tloss>0:
            # yolo
            tloss.backward()
            if(yloss>0):
                grad_yolo = im.grad
                grad_np_yolo = grad_yolo.data.cpu().numpy()
                normalized_grad_yolo = grad_np_yolo / np.sqrt(np.sum(grad_np_yolo * grad_np_yolo, axis=(2, 3), keepdims=True))  # l2
            else:
                normalized_grad_yolo = np.zeros([1,3,608,608])
            # faster rcnn
            if(rloss>0):
                grad_rcnn = data['img'][0].grad
                grad_np_rcnn = grad_rcnn.data.cpu().numpy()
                normalized_grad_rcnn = grad_np_rcnn / np.sqrt(np.sum(grad_np_rcnn * grad_np_rcnn, axis=(2, 3), keepdims=True))  # l2
            else:
                normalized_grad_rcnn = np.zeros([1,3,800,800])
        else:
            return finalimg,NOISE
        NP_P_yolo = normalized_grad_yolo * epsilon # 整个输入图片的单次扰动
        noise_yolo += NP_P_yolo.squeeze().transpose(1,2,0)    # [608,608,3]
        NP_P_rcnn = normalized_grad_rcnn * epsilon        # faster rcnn 量级不一样
        noise_rcnn += NP_P_rcnn.squeeze().transpose(1,2,0)
        NOISE = cv2.resize(noise_yolo,(500,500))*mask + cv2.resize(noise_rcnn*P,(500,500))*mask
        temp_img = np.clip((img - NOISE[:,:,::-1]), 0, 255)  # 改为int32
        X = np.uint8(temp_img)
        finalimg = Image.fromarray(cv2.cvtColor(X, cv2.COLOR_BGR2RGB))  # 500*500*3
        new_input = resize_big(finalimg)
        temp_data = prepare_data(rcnn_model, X)
        data['img'][0].data = temp_data['img'][0].data
    return finalimg,NOISE


def ada_attack(yolo_model, rcnn_model, img, conf_thresh, max_iter, epsilon, mask):
    # yolo 部分
    noise_yolo = np.zeros([608, 608, 3])  # 设置为原图大小
    # 将输入图片转换到PIL格式
    finalimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 500*500*3  RGB
    new_input = resize_big(finalimg)
    # noise_yolo = 0.2*np.array(new_input,dtype=np.float)
    # faster rcnn部分
    data = prepare_data(rcnn_model, img)  # tuple类型
    data['img'][0].requires_grad = True  # 需要可导
    noise_rcnn = np.zeros([800, 800, 3])  # 处理好图片的大小
    last_loss = []
    NOISE = None
    for iter_n in range(max_iter):
        rloss = rcnn_loss(rcnn_model, data)
        im, yloss = yolo_loss(yolo_model, new_input, conf_thresh, use_cuda=1)
        tloss = (yloss + rloss) / 2   # faster_rcnn和yolo损失的融合
        # print(tloss.item())
        if tloss > 0:
            # yolo
            tloss.backward()
            if yloss > 0:
                grad_yolo = im.grad
                grad_np_yolo = grad_yolo.data.cpu().numpy()
                normalized_grad_yolo = np.sign(grad_np_yolo)
            else:
                normalized_grad_yolo = np.zeros([1, 3, 608, 608])
            # faster rcnn
            if rloss > 0:
                grad_rcnn = data['img'][0].grad
                grad_np_rcnn = grad_rcnn.data.cpu().numpy()
                normalized_grad_rcnn = np.sign(grad_np_rcnn)
            else:
                normalized_grad_rcnn = np.zeros([1, 3, 800, 800])
        else:
            return finalimg, NOISE
        last_loss.append(tloss.item())
        temp_list = last_loss[-5:]
        if temp_list[-1] > temp_list[0]:
            if epsilon > 1:
                epsilon = epsilon * 0.93
            else:
                epsilon = 1.0
        NP_P_yolo = normalized_grad_yolo * epsilon  # 整个输入图片的单次扰动
        noise_yolo += NP_P_yolo.squeeze().transpose(1, 2, 0)  # [608,608,3]
        NP_P_rcnn = normalized_grad_rcnn * epsilon
        noise_rcnn += NP_P_rcnn.squeeze().transpose(1, 2, 0)
        NOISE = cv2.resize(noise_yolo, (500, 500)) * mask + cv2.resize(noise_rcnn, (500, 500)) * mask
        ind_one = np.logical_and(NOISE >= 0, NOISE <= 1.1)
        ind_minus_one = np.logical_and(NOISE >= -1.1, NOISE < 0)
        NOISE[ind_one] = 1.1
        NOISE[ind_minus_one] = -1.1
        NOISE = NOISE * mask
        if iter_n == 0:
            NOISE += np.ones_like(NOISE)
        temp_img = np.clip((img - NOISE[:, :, ::-1]), 0, 255)  # 改为int32
        X = np.uint8(temp_img)
        finalimg = Image.fromarray(cv2.cvtColor(X, cv2.COLOR_BGR2RGB))  # 500*500*3
        new_input = resize_big(finalimg)
        temp_data = prepare_data(rcnn_model, X)
        data['img'][0].data = temp_data['img'][0].data
    return finalimg, NOISE
