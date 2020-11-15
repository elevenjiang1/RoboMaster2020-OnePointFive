"""
这个用于一些常用工具的生成
"""
import cv2
import numpy as np
import torch.nn as nn
import torch


#####################用于图片预处理的函数
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    #进行xy点的变换
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)#基于第二个点,得到反方向上的一个点

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    """
    通过生成3组对应的点,得到仿射变换的位置
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):#如果scale不是nparray或者不是list
        scale = np.array([scale, scale], dtype=np.float32)#将scale变成一个list

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]#输出尺寸
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180#把rot变成变成rad
    src_dir = get_dir([0, src_w * -0.5], rot_rad)#可能是生成变换矩阵
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)#给定3组一一对应的点,可以得到仿射变换中的解
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift#基于尺寸进行中心移动
    src[1, :] = center + src_dir + scale_tmp * shift#基于
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])#分别送入第一个第二个点,得到第三个点
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:#是正变换还是反变换
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))#基于对应的尺寸进行图片的变换
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))#默认为正变换

    return trans



##########用于decode的函数###########################
def _nms(heat, kernel=3):
    """
    送入heatmap,进行了max_pool2d的操作之后,获取3x3中的最大值(相当于是做了一个最大值滤波)
    @param heat:
    @param kernel:
    @return:
    """
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=100):
    """
    这一快在网络构建完成之后就知道到底是怎么样操作的了,现在只是知道他们一起获得了dets
    @param scores:
    @param K:
    @return:
    """
    batch, cat, height, width = scores.size()#得到heatmap的参数

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)#把height,width打平成一维向量

    topk_inds = topk_inds % (height * width)#取余得到每张heatmap的索引
    topk_ys   = (topk_inds / width).int().float()#除法得到ys
    topk_xs   = (topk_inds % width).int().float()#取余得到xs

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)#再对种类进行操作,得到每个种类的分数

    topk_clses = (topk_ind / K).int()#进行拆分?

    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs



def _gather_feat(feat, ind):
    dim  = feat.size(2)#索引到第三个参数
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)#把维度1中的参数变为ind的参数

    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()#把cat放到最后面
    feat = feat.view(feat.size(0), -1, feat.size(3))#第一个还是batch,最后就是cat,中间是网络结构
    feat = _gather_feat(feat, ind)#把对应的ind和feat送入
    return feat




##########用于网络创建的函数###########################

from build_network import get_network#这个是一个用于创建网络的函数


#
# _model_factory = {
#     'res': get_pose_net, # default Resnet with deconv
#     'dlav0': get_dlav0, # default DLAup
#     'dla': get_dla_dcn,
#     'resdcn': get_pose_net_dcn,
#     'hourglass': get_large_hourglass_net,
# }


def create_model(arch,heads,head_conv):
    """
    用于基于opt的arch创建网络模型
    @param arch:
    @param heads:
    @param head_conv:
    @return:
    """

    """
    opt.arch内部的参数为:
    'res_18 | res_101 | resdcn_18 | resdcn_101 |dlav0_34 | dla_34 | hourglass'
    """
    arch='res_18'
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0#获得需要创建的网络层
    arch = arch[:arch.find('_')] if '_' in arch else arch#用于查看创建的网络种类
    # get_model = _model_factory[arch]#这一句其实就是获得了函数
    # model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    model = get_network(num_layers=num_layers, heads=heads, head_conv=head_conv)


    return model




########################网络后处理########################33
def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        #对dets进行处理
        top_preds = {}

        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))#x方向变换?

        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))#y方向变换?

        classes = dets[i, :, -1]

        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([dets[i, inds, :4].astype(np.float32),dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()

        ret.append(top_preds)
    return ret

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)#生成坐标系大小

    trans = get_affine_transform(center, scale, 0, output_size, inv=1)#进行反着的仿射变换,变回去目标的形状

    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]




















