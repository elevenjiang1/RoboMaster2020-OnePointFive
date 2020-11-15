"""
这个用于生成神经网络的网络部分
基于网络识别进行输出
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import time

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat



class MyDetector():
    def __init__(self,opt):
        self.opt=opt
        self.model=None#之后需要补充网络模型

    def process(self,images,return_time=False):
        """
        这里面把Tensor送入网络,然后进行decode,得到最后的识别结果
        """
        with torch.no_grad():
            output=self.model(images)[-1]
            heatmap=output["heatmap"]
            whmap=output["whmap"]#用于输出一个目标的width和height
            regmap=output["regmap"]#用于输出细分的尺寸


        torch.cuda.synchronize()
        forward_time=time.time()
        dets=self.decode(heatmap,whmap,regmap)

        if return_time:
            return output,dets,forward_time
        else:
            return output,dets





    def decode(self,heatmap,whmap,regmap,cat_spec_wh=False,K=100):
        """
        用于进行解码操作
        """
        batch,cat,height,width=heatmap.size()#基于heatmap获取对应的尺寸
        heatmap=self._nms(heatmap)#对heatmap进行nms操作

        scores,inds,clses,ys,xs=self._topk(heatmap,K=K)#对heatmap索引最大的100个上下左右都比他小的值



        #基于heatmap得到了inds,然后对regmap和whmap进行处理,得到他们的对应参数
        if regmap is not None:
            regmap = _transpose_and_gather_feat(regmap, inds)
            regmap = regmap.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + regmap[:, :, 0:1]#对xs进行微调
            ys = ys.view(batch, K, 1) + regmap[:, :, 1:2]#对ys进行微调
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5

        whmap  = _transpose_and_gather_feat(whmap, inds)#得到了whmap的参数

    
        if cat_spec_wh:
            whmap = whmap.view(batch, K, cat, 2)
            clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
            whmap = whmap.gather(2, clses_ind).view(batch, K, 2)
        else:
            whmap = whmap.view(batch, K, 2)#改变一下尺寸

        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - whmap[..., 0:1] / 2,
                            ys - whmap[..., 1:2] / 2,
                            xs + whmap[..., 0:1] / 2,
                            ys + whmap[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)#最后得到了detections的结果
    
        return detections




            
    def _nms(heatmap, kernel=3):
        """
        采用torch.max_pool2d,获取3x3中的最大值进行保留
        """
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heatmap).float()#这里应该是保留的值?
        return heatmap * keep#keep是map_pool之后的最大值,进行保存

    def _topk(self,scores, K=100):
        batch, cat, height, width = scores.size()#就是heatmap的size,种类,长高的参数

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)#对heatmap进行了view(打平,打平的最后一个维度为128x128,得到了对应的heatmap值和索引的位置

        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds / width).int().float()#得到了每个点的y值
        topk_xs   = (topk_inds % width).int().float()#得到了每个点的x值

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)#得到了单个目标的topk_score和topk_ind
        topk_clses = (topk_ind / K).int()

        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


    def _gather_feat(self,feat, ind, mask=None):
        """
        ind即topk产生的index
        """
        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat#得到了对应的输出?