"""
这里面负责处理图片前向传播的三个操作
1.预处理(图片->Tensor)
2.神经网络(构建res18和decode)
3.神经网络解析
"""
import cv2 as cv
import numpy as np
import torch
from my_opt import opts
import my_tools


mean = np.array([0.40789654, 0.44719302, 0.47026115],
                dtype=np.float32).reshape(1, 1, 3)
std  = np.array([0.28863828, 0.27408164, 0.27809835],
                dtype=np.float32).reshape(1, 1, 3)


class My_Network:
    def __init__(self,opt):
        self.opt=opt#太多默认的参数,因此暂时先使用这里面的看看效果
        self.model=my_tools.create_model(opt.arch,opt.heads,opt.head_conv)

    def pre_process(self,image,scale):
        """
        送入图片,最后预处理完成后的Tensor
        根据不同的情况,返回的尺寸不同,这里面pad的操作我还是有一点没看懂,但是coco的512维度的图片的处理看懂了
        之后可以尝试不同尺度的训练等的操作
        还有图像的仿射变换这一部分,虽然没看懂,但是也可以掉包了,之后再进行处理把
        @param image:
        @param scale:应该是多尺度预处理,这里面默认为1,不过这里面我觉得也可以尝试一下多尺度
        @return:pre_processed_image
        """
        height,width=image.shape[0:2]#得到图片长宽
        new_height=int(height*scale)#scale本身默认为1,不尽兴图片尺寸的更改
        new_width=int(width*scale)


        fix_res=True#是否进行固定尺寸操作
        if fix_res:
            #input_h,input_w=dataset.default_resolution
            #于coco中是:  default_resolution = [512, 512]
            #voc中是  default_resolution = [384, 384]
            input_h,input_w=[512,512]
            inp_height,inp_width=input_h,input_w#送入网络的尺寸
            c=np.array([new_width/2.,new_height/2.],dtype=np.float32)#图片本来的中心
            s=max(height,width)*1.0#最大的尺度

        else:
            #pad=127 if 'hourglass' in opt.arch else 31
            #这里的pad有可能是为了保证最后面是整数,同时可以同时除以很多次2
            pad=31
            inp_height=(new_height|pad)+1#不是很懂这里面按位与的操作是干什么的,31和127的确都是去全1的
            inp_width=(new_width|pad)+1

            c=np.array([new_width/2,new_height/2],dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)#不固定像素(应该是像素比),直接进行pad操作


        trans_input=my_tools.get_affine_transform(center=c,scale=s,rot=0,output_size=[inp_width,inp_height])#基于中心点和旁边几个点,获得图片变换矩阵

        resized_image=cv.resize(image,(new_width,new_height))#图片resize(其实就是原图)

        #生成了inp_width,inp_height尺寸的图片
        inp_image=cv.warpAffine(src=resized_image,M=trans_input,dsize=(inp_width,inp_height),flags=cv.INTER_LINEAR)#进行仿射变换

        mean = np.array([0.40789654, 0.44719302, 0.47026115],
                        dtype=np.float32).reshape(1, 1, 3)
        std  = np.array([0.28863828, 0.27408164, 0.27809835],
                        dtype=np.float32).reshape(1, 1, 3)

        inp_image=((inp_image/255.-mean)/std).astype(np.float32)#用于归一化到0,1的图片,每一个通道/255,然后减去均值,得到目标

        # inp_image=((inp_image/255.-self.mean)/self.std).astype(np.float32)#用于归一化到0,1的图片

        images=inp_image.transpose(2,0,1).reshape(1,3,inp_height,inp_width)#先变成3,512,512,之后变成1,3,512,512
        #
        # if self.opt.flip_test:
        #     images=np.concatenate((images,images[:,:,:,::-1]),axis=0)

        images=torch.from_numpy(images)

        meta = {'c': c, 's': s,
                'out_height': inp_height // 4,#神经网络的降采样比例
                'out_width': inp_width // 4}
                # 'out_height': inp_height // self.opt.down_ratio,#神经网络的降采样比例
                # 'out_width': inp_width // self.opt.down_ratio}


        return images, meta

    def process(self,pre_processed_image):
        """
        送入预处理完成的图片,返回网络识别结果
        @param pre_processed_image:
        @return: network_output
        """
        with torch.no_grad():
            output=self.model(pre_processed_image)[-1]
            heatmap=output["heatmap"].sigmoid_()
            whmap=output["whmap"]
            regmap=output["regmap"]
            detect=self.decode(heatmap=heatmap,whmap=whmap,regmap=regmap,cat_spec_wh=False,K=100)

        return output,detect

    def decode(self,heatmap,whmap,regmap=None,cat_spec_wh=False,K=100):
        """
        送入神经网络输出的各种map,然后把map转换成了目标的输出结果
        @param heatmap: 目标检测图
        @param whmap: wh的回归图
        @param regmap: 微调整图
        @param cat_spec_wh: 这个看不懂
        @param K: 每一张图最多的检测目标(之后这个可以降下去)
        @return: 检测的目标
        """
        batch,cat,height,width=heatmap.size()#获取heatmap信息

        heatmap=my_tools._nms(heatmap)#对heatmap进行了最大值的滤波

        scores,inds,clses,ys,xs=my_tools._topk(heatmap,K=K)#采用topk对heatmap处理,获取识别结果,最后打平成了batch,K的尺寸

        if regmap is not None:
            regmap=my_tools._transpose_and_gather_feat(regmap,inds)
            regmap=regmap.view(batch,K,2)
            xs=xs.view(batch,K,1)+regmap[:,:,0:1]
            ys=ys.view(batch,K,1)+regmap[:,:,1:2]

        else:
            xs=xs.view(batch,K,1)+0.5
            ys=ys.view(batch,K,1)+0.5


        whmap=my_tools._transpose_and_gather_feat(whmap,inds)

        if cat_spec_wh:#默认这个为False
            whmap=whmap.view(batch,K,cat,2)
            clses_ind=clses.view(batch,K,1,1).expand(batch,K,1,2).long()#这个expand的操作有一点看不懂,还有view的操作也是
            whmap=whmap.gather(2,clses_ind).view(batch,K,2)#gather的操作也看不懂
        else:
            whmap=whmap.view(batch,K,2)


        clses=clses.view(batch,K,1).float()
        scores=scores.view(batch,K,1)
        bboxes = torch.cat([xs - whmap[..., 0:1] / 2,
                            ys - whmap[..., 1:2] / 2,
                            xs + whmap[..., 0:1] / 2,
                            ys + whmap[..., 1:2] / 2], dim=2)


        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections

    def post_process(self,network_output,meta,scale=1):
        """
        送入网络的输出,得到识别结果
        @param network_output: 网络输出
        @return: detections(识别结果)
        """
        network_output=network_output.detach().cpu().numpy()#detach相当于是取消了这个tensor的梯度操作
        network_output=network_output.reshape(1,-1,network_output.shape[2])

        network_output=my_tools.ctdet_post_process(dets=network_output,c=[meta['c']],s=[meta['s']],h=meta['output_height'],w=meta['output_width'],num_classes=1)
        for j in range(1,self.num_classes+1):
            network_output[0][j]=np.array([network_output[0][j]],dtype=np.float32).reshape(-1,5)
            network_output[0][j][:,:4]=network_output[0][j][:,:4]/scale#scale=1

        return network_output[0]





    def network_forward(self,image_path):
        image=cv.imread(image_path)
        # image=cv.resize(image,(1000,1000))

        # for scale in self.scales:
        scale=1
        images,meta=self.pre_process(image,scale)
        print(images.dtype)
        print(images.shape)

        image=images.numpy()
        print(image.shape)
        image=image.transpose(0,2,3,1)
        print(image.shape)
        image=image[0]
        cv.imshow("image",image)
        cv.waitKey(0)





if __name__ == '__main__':
    image_path="1.jpg"
    my_network=My_Network(opts)
    my_network.network_forward(image_path)






