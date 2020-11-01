"""
封装YOLO的识别
"""
import torch
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import time
from torch.autograd import Variable

from YOLO import yolo_models
from YOLO.yolo_utils.datasets import pad_to_square,resize
from YOLO.yolo_utils.utils import non_max_suppression


class Yolo_Detect():
    def __init__(self,cfgfile_path,weight_path):
        """
        进行YOLO_Detect类的初始化
        两个文件都可以放到总的路径下
        :param cfgfile_path: cfg文件路径
        :param weight_path: 参数路径
        """
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=yolo_models.Darknet(cfgfile_path).to(device)#导入cfg文件
        self.model.load_state_dict(torch.load(weight_path))#导入参数
        self.model.eval()

        self.image=None#我觉得没必要存着一个东西
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detections=None

    def pre_process_img(self,image):
        """
        送入预处理图片,让它变为Tensor进行识别
        (其中还包含了预处理后图片的显示操作)
        :param image: 需要处理的图片
        :return:处理好的图片
        """
        self.image=image#用于结果展示保存的图片
        #3.1:变RGB色域,同时改变数据类型
        RGBimage=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        img=transforms.ToTensor()(RGBimage)

        #3.2:图像变换到正方形
        img,_=pad_to_square(img,0)
        img=resize(img,416)

        #3.3:加载到GPU上进行处理
        img=img.unsqueeze(0)#增加一个维度
        img=img.to(self.device)#加载到GPU上

        #这一块是进行图片将Tensor图片变换为opencv可以处理的图片
        # processed_img=img.numpy()#尺寸变为了3,416,416
        # print("处理过后的图像的尺寸为:",processed_img.shape)
        #
        #之后变换回416,416,3,进行图像显示
        # show_img=np.transpose(processed_img,(1,2,0))
        # cv.imshow("处理完图像",show_img)
        # cv.waitKey(0)

        return img

    def detect(self,image,confidence=0.6,nms_thres=0.4,is_preprocessed=False):
        """
        送入图片,进行预处理,最后进行识别.主要用于单帧识别
        :param image: 送入的图片
        :param is_preprocessed: 图片是否经过预处理,没有的话就进行一次预处理
        :param confidence: 这里的confidence是这个地方是否有物体的判断,而非识别这个目标的正确率的判断
        :return:
        """
        if not is_preprocessed:#进行图片预处理
            image=self.pre_process_img(image)

        self.detections=None
        detections=self.model(image)


        detections=non_max_suppression(detections,conf_thres=confidence,nms_thres=nms_thres)#这里可以设置正确率的大小
        self.detections=detections


        return detections

    def detect_parrel_better(self,car_list,parrel_number):
        """
        传入一个list,其中的每一个dict中包含了image这个项,最终在每一个dict中返回
        同时,这里还需要包含预处理的工作
        :param car_list: 包含图片的list
        :param parrel_number: 一次识别的图片数目
        :return:
        """
        with torch.no_grad():
            #1:整数Tensor识别
            # begein=time.time()
            int_number=int(len(car_list)/parrel_number)#计算可以整轮重复的个数
            if int_number>0:
                for i in range(int_number):#这里会向下取整
                    #存入一次处理的图片
                    image_temp=[]
                    for j in range(parrel_number):
                        image_temp.append(self.pre_process_img(car_list[i*parrel_number+j]['image']))

                    #进行参数拼接
                    conbine=torch.cat(image_temp)
                    conbine=Variable(conbine)

                    # end1=time.time()

                    detections=self.detect(conbine,is_preprocessed=True)

                    # end2=time.time()

                    #最后保存result参数
                    for j,detection in enumerate(detections):
                        car_list[i*parrel_number+j]['result']=detection



            # endfirst=time.time()
            # print("第一轮消耗总时间为:{}".format(endfirst-begein))
            # print("最后一次预处理耗时:{}".format(end1-begein))
            # print("最后一次识别耗时:{}".format(end2-end1))


            #2:补充缺漏识别
            if len(car_list)>int_number*parrel_number:
                image_temp=[]
                begin_number=int_number*parrel_number
                for j in range(begin_number,len(car_list)):
                    image_temp.append(self.pre_process_img(car_list[j]['image']))

                conbine=torch.cat(image_temp)
                conbine=Variable(conbine)

                # end1=time.time()
                detections=self.detect(conbine,is_preprocessed=True,confidence=0.8)

                # end2=time.time()

                for j,detection in enumerate(detections):
                    car_list[begin_number+j]['result']=detection



            # endsencond=time.time()
            #
            # print("第二轮消耗总时间为:{}".format(endsencond-endfirst))
            # print("最后一次预处理耗时:{}".format(end1-endfirst))
            # print("最后一次识别耗时:{}".format(end2-end1))


        return car_list

    def change_origin(self,x1,y1,x2,y2,origin_image=None):
        """
        由于yolov3是经过了resize的操作的,因此这里我们也需要进行一个把所有的识别目标resize回去的操作
        这里的目标都是resize到416
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """
        if origin_image is not None:
            h,w,channel=origin_image.shape
        elif self.image is not None:
            h,w,channel=self.image.shape
        else:
            h,w,channel=1080,1920,3

        if w>h:#即更宽,y方向上要处理
            x1=x1*w/416
            x2=x2*w/416
            y1=y1*w/416
            y2=y2*w/416

            remove_pad=(w-h)/2#变掉pad操作
            y1=y1-remove_pad
            y2=y2-remove_pad

        else:
            x1=x1*h/416
            x2=x2*h/416
            y1=y1*h/416
            y2=y2*h/416

            #因为pad操作,因此需要重新移动一下目标
            remove_pad=(h-w)/2#变掉pad操作
            x1=x1-remove_pad
            x2=x2-remove_pad

        return int(x1),int(y1),int(x2),int(y2)

    def show_result(self,puttext=False):
        """
        这里主要是展示一下如何进行识别效果展示
        :return:
        """
        show_image=self.image.copy()
        for detection in self.detections:
            if detection is not None:#self.detections返回的是[None]而非直接None,因此需要直接进行索引
                for every_object in detection:
                    x1, y1, x2, y2, conf, cls_conf, cls_pred=every_object#返回的是(x1, y1, x2, y2, object_conf, class_score, class_pred)
                    x1, y1, x2, y2=yolo_detect.change_origin(x1, y1, x2, y2,show_image)
                    cv.rectangle(show_image,(x1,y1),(x2,y2),(0,255,0),1)
                    if puttext:
                        puttext_size=cv.getTextSize("conf:{},class:{}".format(cls_conf,cls_pred),cv.FONT_HERSHEY_SIMPLEX,0.3,1)
                        show_image[y1-puttext_size[0][1]:y1+puttext_size[1],x1:x1+puttext_size[0][0]]=(0,255,0)#区域上色
                        cv.putText(show_image,"conf:{},class:{}".format(cls_conf,cls_pred),(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1)

                cv.namedWindow("result",cv.WINDOW_NORMAL)
                cv.imshow("result",show_image)
                cv.waitKey(0)

        else:
            print("没有发现目标")


if __name__ == '__main__':
    #进行图片的识别,查看效果
    yolo_detect=Yolo_Detect(cfgfile_path="../../cfg_weight/yolov3-armor.cfg",weight_path="../../cfg_weight/yolov3_armor_36.pth")
    image_path="./3.jpg"
    image=cv.imread(image_path)
    detections=yolo_detect.detect(image,confidence=0.8)#类里面也进行了detections的查看
    yolo_detect.show_result()
