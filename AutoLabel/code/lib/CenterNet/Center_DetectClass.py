"""
封装CenterNet的识别
"""
import os
import sys
import cv2 as cv
import torch


CENTERNET_PATH =os.path.dirname(__file__)+"/lib_oneclass"#导入对应的lib文件

sys.path.insert(0, CENTERNET_PATH)


if os.path.exists(CENTERNET_PATH):
    print("CenterNet Path is existed")
else:
    print("[Error] The Detect Path is not Define")

# from CenterNet.lib.opts import opts
# from CenterNet.lib.detectors.detector_factory import detector_factory

from detectors.detector_factory import detector_factory
from opts import opts

class CenterDetect():
    def __init__(self,model_path,Task):
        #初始化模型
        self.MODEL_PATH =model_path
        self.TASK = Task # Task为'ctdet'(目前仅支持这一种)
        self.opt = opts().init('{} --load_model {}'.format(self.TASK, self.MODEL_PATH).split(' '))#类似于yolo中的初始化cfg文件
        self.Center_Detector=detector_factory[self.opt.task]
        self.detector=self.Center_Detector(self.opt)

        #参数初始化
        self.detections=None
        self.image=None

    def detect(self,image):
        """
        用于进行centernet的图像识别
        :param image: 送入图像
        :return:识别到的detections
        (注意):
        detectios是一个字典,字典的索引是每一个类别,detections[类别]是每一个类别识别到的所有目标,每个目标包含了5个值,即x1,y1,x2,y2,socre
        """
        with torch.no_grad():
            detections=self.detector.run(image)['results']#返回的是一个dict,dict的索引对应的是类别id

            # print(detections)

        self.detections=detections
        self.image=image
        self.show_image=image.copy()
        return detections

    def show_result(self,confidence=0.5,show_windows="result",waitKey=0,show_confidence=True):
        if self.detections is not None:
            for category in self.detections:#用于获取detections中的id号
                for result in self.detections[category]:#用于获取detections中的每个类
                    if result[4]>confidence:
                        x1,y1,x2,y2,score=result
                        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                        cv.rectangle(self.show_image,(x1,y1),(x2,y2),(0,255,0),2)

                        if show_confidence:
                            show_text_info="car {:.3f}".format(score)
                            text_box_size=cv.getTextSize(show_text_info,cv.FONT_HERSHEY_SIMPLEX,0.5,2)
                            self.show_image[y1-text_box_size[0][1]:y1,x1:x1+text_box_size[0][0]]=(0,255,0)
                            cv.putText(self.show_image,show_text_info,(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)


            cv.namedWindow(show_windows,cv.WINDOW_NORMAL)
            cv.imshow(show_windows,self.show_image)
            cv.waitKey(waitKey)


    def fileter_detections(self,confidence=0.5,category=1,detections=None):
        """
        用于过滤识别结果
        :param confidence: 置信度
        :param category: 识别种类
        :return: 好的结果
        """
        return_result=[]
        if detections is None:
            detections=self.detections

        for result in detections[category]:
            if result[4]>confidence:
                x1,y1,x2,y2,score=result
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                return_result.append([x1,y1,x2,y2,score,category])#最后加上一个种类

        return return_result


    def detect_morethanone(self,image,confidence=0.5,return_bboxes=1,category=1):
        """
        高于confidence有目标,则有多少个返回多少个,如果没有高于confidence的,则返回return_bboxes的数量
        :param confidence:
        :return:
        """
        with torch.no_grad():
            detections=self.detector.run(image)['results']#返回的是一个dict,dict的索引对应的是类别id

        filtered_results=self.fileter_detections(confidence=confidence,category=category,detections=detections)
        if len(filtered_results)>0:
            return filtered_results
        else:
            #保证至少有一个目标
            list_detections=detections[category].tolist()
            list_detections.sort(key=lambda x:x[4],reverse=True)#按照置信度高低排列
            return_list=[]
            for i,result in enumerate(list_detections):
                if i>=return_bboxes:#当i大于等于return_bboxes时,则不进行返回
                    break
                x1,y1,x2,y2,score=result
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                return_list.append((x1,y1,x2,y2,score,category))#最后加上一个种类

            return return_list


if __name__ == '__main__':
    center_detect=CenterDetect(model_path="/home/elevenjiang/Documents/Project/RM/Code/CenterNet/CenterNet-OneCategory/models/model_last.pth",Task='ctdet')
    image=cv.imread("test_image.jpg")
    print(image.shape)
    detections=center_detect.detect(image)
    center_detect.show_result()





