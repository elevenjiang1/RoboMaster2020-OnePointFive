import os
import sys
import cv2 as cv
import threading
import time
import tkinter as tk
import tools
import show_results
import copy
from CenterNet.Center_DetectClass import CenterDetect
CENTERNET_PATH="CenterNet/lib_oneclass"
sys.path.insert(0,CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts
from YOLO.YOLO_DetectClass import Yolo_Detect
import numpy as np

class BaseDetector:
    def __init__(self):
        """
        基本的检测类,所有的自动结果检测都需要集成这个类
        """
        self.flag_trackcar=False
        self.flag_detectcar=False
        self.flag_detectarmor=False
        self.flag_show_annotation=False

    def tracker_init(self,image):
        """
        用于初始化目标跟踪器
        这里面需要更新self.track_category_id和self.tracker两个内置参数,并且选择对应的bbox
        @param image: 送入目标跟踪器
        @return: bboxes 手动标注的跟踪bboxes
        """
        raise NotImplementedError

    def tracker_detect(self,image):
        """
        跟踪识别接口,送入图片,返回跟踪目标的bbox
        返回的格式为:
        track_flag,[[x1,y1,x2,y2,None,category_id],[x1,y1,x2,y2,None,category_id],...]
        #track_ok表示是否跟踪成功
        #[x1,y1,x2,y2,None,category_id]为跟踪目标.None指代分数,单目标没有分数
        @param image: 送入图片
        @return: track_ok,[[x1,y1,x2,y2,None,category_id],[x1,y1,x2,y2,None,category_id],...]
        """
        raise NotImplementedError

    def network_detect(self,image):
        """
        神经网络识别接口,送入图片,返回检测目标的bboxes
        返回格式:
        [[x1,y1,x2,y2,score,category_id],[x1,y1,x2,y2,score,category_id],...]
        @param image: 识别的图片
        @return: 识别结果
        """
        raise NotImplementedError

    def armor_detect(self,image):
        """
        装甲板检测接口,等待填写
        @param image: 识别装甲板
        @return:
        """
        raise NotImplementedError

    def all_detect_onimage(self,image,image_show):
        """
        融合目标跟踪和目标检测
        @return:trackbbox,networkbboxes
        """
        trackbboxes=[]
        networkbboxes=[]
        if self.flag_trackcar:
            track_ok,trackbboxes=self.tracker_detect(image)
            if track_ok:
                show_results.show_track_result(trackbboxes,image_show)
            else:
                print("跟踪失败")
                trackbboxes=[]

        if self.flag_detectcar:
            networkbboxes=self.network_detect(image)
            show_results.show_network_result(networkbboxes,image_show)

        return trackbboxes,networkbboxes

class EasyDetector(BaseDetector):
    def __init__(self,Dataset_Name):
        """
        EasyDetector用于不导入任何模型,只让使用者用最基本的内容
        @param Dataset_Name: 数据集名称,用于使用不同的网络训练参数
        """
        super(EasyDetector,self).__init__()

    def tracker_init(self,image):
        pass

    def tracker_detect(self,image):
        return False,None

    def network_detect(self,image):
        return None

    def armor_detect(self,image):
        return None

class MyDetector(BaseDetector):
    def __init__(self,Dataset_Name):
        """
        MyDetector作为我自己使用的检测器,主要完成tracker_detect,network_detect和armor_detect的接口实现
        @param Dataset_Name: 数据集名称,用于使用不同的网络训练参数
        """
        super(MyDetector,self).__init__()
        #1:初始化跟踪
        self.tracker=cv.MultiTracker_create()#使用多目标跟踪方法
        self.track_method="KCF"#默认使用KCF的方法,太多方法其实没太大必要
        self.tracker_category_id=None

        #2:初始化车辆识别
        if Dataset_Name=="SmallMap":
            model_path="../../CenterNet_File/CenterNet_CK/exp/ctdet/coco_dla/model_last.pth"
            model_path=os.path.abspath(model_path)
            self.centerDetect=CenterDetect(model_path=model_path,Task='ctdet')

        elif Dataset_Name=="HitRotate":
            model_path="../../CenterNet_File/Center_HitRotate/exp/ctdet/coco_dla/model_last.pth"
            model_path=os.path.abspath(model_path)
            self.centerDetect=CenterDetect(model_path=model_path,Task='ctdet')

        else:
            model_path="../../CenterNet_File/CenterNet_CK/exp/ctdet/coco_dla/model_last.pth"
            model_path=os.path.abspath(model_path)
            print("一个全新的数据集,默认使用全场识别数据")
            self.centerDetect=CenterDetect(model_path=model_path,Task='ctdet')

        #2.1:识别中使用前景检测
        self.use_foreground=False
        self.first_frame_gray=None

        #3:初始化装甲板检测
        cfgfile_path="/home/elevenjiang/Documents/Project/RM/Code/SmallMap/DetectAfterDetect/Code/cfg_weight/yolov3-armor.cfg"
        weight_path="/home/elevenjiang/Documents/Project/RM/Code/SmallMap/DetectAfterDetect/Code/cfg_weight/yolov3_armor_36.pth"
        self.yolo_detect=Yolo_Detect(cfgfile_path=cfgfile_path,weight_path=weight_path)

    def tracker_init(self,image):
        """
        用于进行目标跟踪初始化
        都采用KCF的方法进行跟踪,其他的方法并不好用
        :param image: 初始化的图片
        :param track_method: 跟踪算法(默认为KCF)
        :return:
        """
        #每一次初始化,都把以前的跟踪目标给重置,要不然越add越多
        self.tracker=cv.MultiTracker_create()
        category_num=10#默认最多不超过10个类
        #1:进行bboxes的选择
        cv.namedWindow("ROI",cv.WINDOW_FULLSCREEN)
        select_bboxes=cv.selectROIs("ROI",image)
        bboxes=[]

        #1.1bboxes中的所有bbox转成tuple类型
        for bbox in select_bboxes:
            x,y,w,h=bbox
            bboxes.append((x,y,w,h))

        #2:输入跟踪类别id
        print("请在图片中输入种类id")
        while True:
            info_input = cv.waitKey()
            info_input = int(info_input) - 48
            if -1<info_input<category_num:#在所有种类中则完成输入
                self.tracker_category_id = info_input
                break

            else:#避免输入其他错误参数
                print("请输入一个介于0到{}之间的数字".format(category_num))

        #3:进行目标添加
        for bbox in bboxes:
            self.tracker.add(cv.TrackerKCF_create(),image,bbox)
        cv.destroyWindow("ROI")
        return bboxes

    def tracker_detect(self,image):
        return_bboxes=[]
        ok,bboxes=self.tracker.update(image)
        for bbox in bboxes:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            x1,y1,x2,y2=x,y,x+w,y+h
            return_bboxes.append((x1,y1,x2,y2,None,self.tracker_category_id))
        return ok,return_bboxes

    def merge_detectforeground(self,results,mask_blur):
        for i,result in enumerate(results):
            x1,y1,x2,y2,score,category_id=result
            roi=mask_blur[y1:y2,x1:x2]#直接统计bbox里面的白色的值
            sum=np.sum(roi)
            rate=sum/((x2-x1)*(y2-y1)*255)#除以每一个都是255的值,得到应该加的分数
            results[i][4]=results[i][4]+rate#对score进行分数的增加
        return results

    def getMaskblur(self,image):
        """
        送入图片,与整体的背景做差分,得到mask_blur
        @param image: 某一帧图片
        @return: mask_blur
        """
        gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        foreground=cv.absdiff(gray_image,self.first_frame_gray)
        mask=cv.inRange(foreground,30,255)
        mask=cv.morphologyEx(mask,cv.MORPH_OPEN,tools.generate_kernel(5,5))
        mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,tools.generate_kernel(10,10))
        mask=cv.morphologyEx(mask,cv.MORPH_DILATE,tools.generate_kernel(20,20))
        mask_blur=cv.blur(mask,(15,15))
        return mask_blur

    def network_detect(self,image):
        if not self.use_foreground:
            filtered_results=self.centerDetect.detect_morethanone(image)

        else:#使用前景检测
            #生成置信度图
            mask_blur=self.getMaskblur(image)
            cv.namedWindow("mask_blur",cv.WINDOW_NORMAL)
            cv.imshow("mask_blur",mask_blur)
            detections=self.centerDetect.detect(image)
            results=self.centerDetect.fileter_detections(confidence=0.2,category=1,detections=detections)
            merge_results=self.merge_detectforeground(results,mask_blur)
            filtered_results=tools.filter_results(merge_results,confidence=0.6)

        return filtered_results

    def armor_detect(self,image):
        #这里先不写东西,全部嵌入到yolo里面进行处理了
        pass

class Annotation(EasyDetector):
    def __init__(self,Dataset_Name):
        super(Annotation,self).__init__(Dataset_Name=Dataset_Name)

    def str_to_bool(self,flag):
        """
        用于把string改成True或者False
        @param flag:
        @return:
        """
        if flag=="True":
            return True
        else:
            return False

    def Annotation_Mission(self,annotation_info):
        """
        主要用于处理所有的文件路径
        @param annotation_info:
        @return:
        """
        #1:导入annotation_info的信息
        dataset_path=annotation_info["dataset_path"]#读入数据集路径
        images_path=os.path.join(dataset_path,"images")
        annotations_path=os.path.join(dataset_path,"image_annotations")
        tools.check_path(images_path,annotations_path)#确保文件存在

        #2:确定使用的识别方法等
        self.flag_trackcar=self.str_to_bool(annotation_info["flag_trackcar"])
        self.flag_detectcar=self.str_to_bool(annotation_info["flag_detectcar"])
        self.flag_detectarmor=self.str_to_bool(annotation_info["flag_detectarmor"])
        self.flag_show_annotation=self.str_to_bool(annotation_info["flag_show_annotation"])

        #3:进行标注
        images_list=os.listdir(images_path)
        try:
            images_list.sort(key=lambda x:int(x[:-4]))
        except:
            print("排序的文件中有非数字,因此直接sort,而不是采用数字进行sort")
            images_list.sort()#如果有不是数字的,则进行第二种排序


        #3.1:如果有进行一开始就进行跟踪,则进行一下初始化
        if self.flag_trackcar:
            self.tracker=cv.MultiTracker_create()
            track_firstframe_path=os.path.join(images_path,images_list[0])
            track_firstframe=cv.imread(track_firstframe_path)
            self.tracker_init(track_firstframe)#进行跟踪器跟踪

        #进行标注
        index_id=0
        while index_id<len(images_list):
            filename=images_list[index_id][:-4]
            print("*****************{}.jpg的处理**************************".format(filename))
            #4.1:获取image_path和txt_path
            image_path=os.path.join(images_path,images_list[index_id])
            txt_path=os.path.join(annotations_path,filename+".txt")


            return_state=self.Annotation_File(image_path,txt_path)
            if return_state=='b':
                index_id=index_id-2
            if return_state=='j':
                jump_data=input("请输入要跳转的图片名称")
                name=str(jump_data)+".jpg"
                index_id=images_list.index(name)
                print("即将跳转的id为:",index_id)

            if return_state=='q':
                cv.waitKey(1)
                cv.destroyAllWindows()
                break


            index_id=index_id+1
            cv.waitKey(1)

        cv.destroyAllWindows()

    def Annotation_File(self,image_path,txt_path):
        """
        #用于基于图片路径和txt路径,完成标注任务
        @param image_path:
        @param txt_path:
        @return:
        """
        image=cv.imread(image_path)
        image_h,image_w,image_c=image.shape
        image_show=image.copy()

        #处理所有识别结果,并进行绘制
        trackbboxes,networkbboxes=self.all_detect_onimage(image,image_show)


        #进行结果的处理
        image_save_show=image.copy()
        save_infos=[]

        if self.flag_show_annotation:
            label_image=image.copy()
            tools.drawAnnotation(txt_path,label_image)
            cv.namedWindow("labeled_image",cv.WINDOW_NORMAL)
            cv.imshow("labeled_image",label_image)

        while True:
            double_image=cv.hconcat([image_show,image_save_show])#把两张图片拼接从而查看
            cv.namedWindow("image_detect                                          save_image",cv.WINDOW_NORMAL)
            cv.imshow("image_detect                                          save_image",double_image)


            info_input=cv.waitKey()
            #回车键看下一张图
            if info_input==13:
                cv.waitKey(1)
                break

            #送入数字进行车子保存
            input_number=int(info_input)-48
            if -1<input_number<10:#默认一次最多保存10个
                for i,networkbbox in enumerate(networkbboxes):
                    if i>input_number:#一张图只进行置信度最高的9个显示
                        break
                    x1,y1,x2,y2,score,category_id=networkbbox
                    cv.rectangle(image_save_show,(x1,y1),(x2,y2),(0,255,0),3)
                    show_text_info="save:{}   network:{}".format(networkbbox[5],i)#
                    tools.show_text(show_text_info,(x1,y1),image_save_show)
                    if networkbbox not in save_infos:#避免保存多次
                        save_infos.append(networkbbox)

            #特定数字保存
            if info_input==ord('e'):
                while True:
                    C_input_number=cv.waitKey()
                    input_number=int(C_input_number)-48
                    if not -1<input_number<10:
                        print("请输入数字进行保存")
                    else:
                        networkbbox=networkbboxes[input_number]
                        x1,y1,x2,y2,score,category_id=networkbbox
                        cv.rectangle(image_save_show,(x1,y1),(x2,y2),(0,255,0),3)
                        show_text_info="save:{}  ".format(networkbbox[5])#
                        tools.show_text(show_text_info,(x1,y1),image_save_show)
                        save_infos.append(networkbbox)
                        print("完成数据保存")
                        break

            #保存跟踪结果
            if info_input==ord('t'):
                if trackbboxes is not None:
                    #进行跟踪目标的保存
                    for bbox in trackbboxes:
                        x1,y1,x2,y2,score,category_id=bbox
                        # x1,y1,x2,y2,score,category_id=trackbbox
                        cv.rectangle(image_save_show,(x1,y1),(x2,y2),(0,255,0),3)
                        show_text_info="save:{} track".format(self.tracker_category_id)
                        tools.show_text(show_text_info,(x1,y1),image_save_show)
                        if (x1,y1,x2,y2,score,category_id) not in save_infos:
                            save_infos.append((x1,y1,x2,y2,score,category_id))
                else:
                    print("没有开启目标跟踪,可以通过r进行目标跟踪的更新")

            #进行保存
            if info_input==ord('s'):#s键进行保存
                for save_info in save_infos:
                    x1,y1,x2,y2,score,category_id=save_info
                    annotation = '{},{},{},{},{},\n'.format(str(category_id), str(int(x1)), str(int(y1)), str(int(x2)),str(int(y2)))  # 此处输入应该全都是整数

                    #进行保存前bbox的检测
                    w=x2-x1
                    h=y2-y1
                    all_large_0=x1>0 and y1>0 and w>0 and h>0
                    smaller_than_picture=(x1+w)<image_w and (y1+h)<image_h
                    if all_large_0 and smaller_than_picture:
                        file=open(txt_path,'a')
                        print("成功写入: "+annotation)
                        file.write(annotation)
                        file.close()

                    else:
                        print("!!!BBOX存在问题,不进行保存!!!")
                        print("无法保存的信息为:",annotation[:-2])
                        if not all_large_0:
                            print("  这四个值存在一个小于0")
                            if x1<0:
                                x1=1
                            if y1<0:
                                y1=1

                        if not smaller_than_picture:
                            print("  存在大于图片尺寸的情况")
                            print("请输入s或n进行超范围保存选择")
                            if (x1+w)>image_w:
                                print("     横向超出了范围,x+w为{},而图片横向为:{}".format(x1+w,image_w))
                                x2=image_w-2

                            if (y1+h)>image_h:
                                print("     纵向超出了范围,y+h为{},而图片纵向为:{}".format(y1+h,image_h))
                                y2=image_h-2


                            #超出范围的也进行保存,保存的尺寸按照图片中的尺寸保存
                            while True:
                                delete_select=cv.waitKey()
                                print("请输入s或n进行超范围保存选择")
                                if delete_select==ord('s'):
                                    file=open(txt_path,'a')
                                    annotation = '{},{},{},{},{},\n'.format(str(category_id), str(int(x1)), str(int(y1)), str(int(x2)),str(int(y2)))
                                    print("成功写入: "+annotation)
                                    file.write(annotation)
                                    file.close()
                                    break

                                elif delete_select==ord('n'):
                                    print("!!!BBOX存在问题,不进行保存!!!\n")
                                    break

                                else:
                                    print("请输入y或n进行超尺寸保存问题")

                break

            #清空保存
            if info_input==ord('c'):#进行存储删除,图片情况
                save_infos=[]
                image_save_show=image.copy()

            #多目标写入
            if info_input==ord('w'):#进行目标写入,绘制多个bbox,然后输入对应id
                print("进行bboxes的选择,按ESC退出bboexes的选择")
                cv.namedWindow("ROIs",cv.WINDOW_FULLSCREEN)
                bboxes=cv.selectROIs("ROIs",image)
                print("请在图片中输入种类id")#这里为了将目标进行增多,因此要求按一次回车才认为id输入完成,不过最多允许两个值


                category_list=[]#一个list用于保存每一次的输入
                while True:
                    ui_input=cv.waitKey()
                    category_id_input=int(ui_input)-48
                    if -1<category_id_input<10:#在所有种类中则完成输入默认的是不超过10个
                        print("输入的数字为:{}".format(category_id_input))
                        category_list.append(category_id_input)

                    elif ui_input==13:#回车键才进行break
                        break

                    else:#避免输入其他错误参数
                        print("请输入一个介于0到{}之间的数字".format(10))

                for index in range(len(category_list)):
                    if index==0:
                        category_id=category_list[-1]
                    else:
                        category_id=category_id+10^index*category_list[-(index+1)]

                print("多个bbox的选择为:[")
                for bbox in bboxes:
                    x,y,w,h=bbox
                    x,y,w,h=int(x),int(y),int(w),int(h)
                    x1,y1,x2,y2=x,y,x+w,y+h#进行了保存
                    cv.rectangle(image_save_show,(x1,y1),(x2,y2),(0,255,0),2)
                    show_text_info="save:{} hand make".format(category_id)
                    tools.show_text(show_text_info,(x1,y1),image_save_show)

                    if (x1,y1,x2,y2,None,category_id) not in save_infos:
                        save_infos.append((x1,y1,x2,y2,None,category_id))
                        print("      [",x1,y1,x2,y2,None,category_id,"]")

                print("\n")
                cv.destroyWindow("ROIs")
                cv.waitKey(1)

            #更新跟踪
            if info_input==ord('r'):#用于目标跟踪的重新升级
                self.flag_trackcar=True
                bboxes=self.tracker_init(image)#新一张图的效果
                for bbox in bboxes:
                    x,y,w,h=bbox
                    x1,y1,x2,y2=int(x),int(y),int(x+w),int(y+h)
                    save_infos.append((x1,y1,x2,y2,None,self.tracker_category_id))
                    cv.rectangle(image_save_show,(x1,y1),(x2,y2),(0,255,0),3)
                    show_text_info="new track:{}".format(self.tracker_category_id)
                    tools.show_text(show_text_info,(x1,y1),image_save_show)

            #进行目标删除
            if info_input==ord('d'):#用于删除标注数据
                delete_image=image.copy()
                file=open(txt_path,'r')
                txt_data=file.readlines()
                file.close()

                for i,line in enumerate(txt_data):
                    ann=line.split(',')[:-1]#用于去除掉最后的\n
                    ann_int=map(int,ann)
                    number_ann=list(ann_int)
                    category_id, x1, y1, x2, y2 = number_ann
                    cv.rectangle(delete_image,(x1,y1),(x2,y2),(0,0,255),2)#在保存图上面绘制所有标注结果
                    show_text_info="already saved {}".format(i)
                    tools.show_text(show_text_info,(x1,y1),delete_image,color=(0,0,255))

                cv.imshow("deletewindow",delete_image)
                print("请输入希望删除的数字")
                while True:
                    delete_number = cv.waitKey()
                    delete_number =int(delete_number) - 48
                    if -1<delete_number<10:
                        if delete_number<len(txt_data):
                            output_data=txt_data.pop(delete_number)
                            print("删除的的为:",output_data)

                            file=open(txt_path,'w')
                            for line in txt_data:
                                file.write(line)
                            file.close()

                        break
                    else:
                        print("请输入一个介于0到{}之间的数字".format(10))
                cv.destroyWindow("deletewindow")

            #快速标注
            if info_input==ord('f'):
                networkbbox=networkbboxes(0)#只保存第一个目标,其他的都不保存
                x1,y1,x2,y2,score,category_id=networkbbox
                annotation = '{},{},{},{},{},\n'.format(str(category_id), str(int(x1)), str(int(y1)), str(int(x2)),str(int(y2)))  # 此处输入应该全都是整数


                #进行保存前bbox的检测
                w=x2-x1
                h=y2-y1
                all_large_0=x1>0 and y1>0 and w>0 and h>0
                smaller_than_picture=(x1+w)<image_w and (y1+h)<image_h
                if all_large_0 and smaller_than_picture:
                    file=open(txt_path,'a')
                    print("成功写入: "+annotation)
                    file.write(annotation)
                    file.close()

                else:
                    print("无法保存的信息为:",annotation[:-2])
                    if not all_large_0:
                        print("  这四个值存在一个小于0")
                        if x1<0:
                            print("x1的值为:{}".format(x1))
                            x1=1
                        if y1<0:
                            print("y1的值为:{}".format(y1))
                            y1=1

                    if not smaller_than_picture:
                        print("  存在大于图片尺寸的情况")
                        if (x1+w)>image_w:
                            print("     横向超出了范围,x+w为{},而图片横向为:{}".format(x1+w,image_w))
                            x2=image_w-2

                        if (y1+h)>image_h:
                            print("     纵向超出了范围,y+h为{},而图片纵向为:{}".format(y1+h,image_h))
                            y2=image_h-2


                    #直接进行保存,不再询问,反正保存的bbox没问题的
                    file=open(txt_path,'a')
                    annotation = '{},{},{},{},{},\n'.format(str(category_id), str(int(x1)), str(int(y1)), str(int(x2)),str(int(y2)))
                    print("强行更改尺寸之后写入的参数为: "+annotation)
                    file.write(annotation)
                    file.close()

            #装甲板保存
            if info_input==ord('a'):
                """
                这里进行装甲板的保存
                2   3   4   5   6   7   8   9   10  11   12     13
                红1,红2,红3,红4,红5,蓝1,蓝2, 蓝3, 蓝4, 蓝5  看不清  死亡
                """
                print("*************开始装甲板识别任务********************")

                origin_save_infos=copy.deepcopy(save_infos) #为了不进行装甲板保存信息的处理

                for save_info in origin_save_infos:
                    #每一个车子进行处理
                    car_x1,car_y1,car_x2,car_y2,score,category_id=save_info#获取目标参数
                    car_image=image[car_y1:car_y2,car_x1:car_x2]
                    armor_detections=self.yolo_detect.detect(car_image,confidence=0.7,nms_thres=0.2)[0]#默认返回第一个,多个图片的时候可能会出问题

                    absolute_onecar_armors=[]#用于保存在大图中的位置,这里面的
                    onecar_armors=[]#用于保存在小图的位置
                    car_image_show=car_image.copy()
                    if armor_detections is  None:
                        armor_detections=[]


                    for all_i,armor_result in enumerate(armor_detections):
                        #先进行这个车子的结果保存
                        armor_x1,armor_y1,armor_x2,armor_y2,conf, cls_conf, cls_pred=armor_result
                        cls_pred=int(cls_pred)#从tensor变为int
                        cls_conf=float(cls_conf)
                        armor_x1,armor_y1,armor_x2,armor_y2=self.yolo_detect.change_origin(armor_x1,armor_y1,armor_x2,armor_y2,origin_image=car_image)#图片后处理,返回到对应尺寸

                        #获取在大图中的绝对位置
                        absolute_x1,absolute_y1,absolute_x2,absolute_y2=armor_x1+car_x1,armor_y1+car_y1,armor_x2+car_x1,armor_y2+car_y1

                        #处理一下实际保存效果
                        ###################################之后肯定需要重新更改yolo的标签###################################
                        #之后肯定是需要重新做数据,重新改一下对应的id,这里先简单处理一下适配上一个网络的id
                        if cls_pred==0:
                            cls_pred=13
                            armor_id=7#死亡展示的是7

                        else:
                            cls_pred=cls_pred+1
                            if cls_pred<=6:
                                armor_id=cls_pred-1
                            elif 6<cls_pred<13:
                                armor_id=cls_pred-6#看不清展示的是6
                            else:
                                armor_id=6

                        ###################################之后肯定需要重新更改yolo的标签###################################

                        #保存装甲板数据
                        absolute_onecar_armors.append((absolute_x1,absolute_y1,absolute_x2,absolute_y2,conf,cls_conf,cls_pred))
                        onecar_armors.append((armor_x1,armor_y1,armor_x2,armor_y2,conf,cls_conf,cls_pred))

                        #展示的是装甲板id和颜色,但是保存的是绝对的id
                        #到时候看情况决定是否添加小车上面的标注信息
                        if int(cls_pred)<=6:
                            color=(0,0,255)

                        elif 6<cls_pred<=11:
                            color=(255,0,0)

                        else:
                            color=(0,255,0)

                        show_text_info="{},{}".format(armor_id,all_i)
                        cv.rectangle(car_image_show,(armor_x1,armor_y1),(armor_x2,armor_y2),color)
                        tools.show_text(show_text_info,(armor_x1,armor_y1),car_image_show,0.5,color)


                    #对这个车子识别结果进行判断
                    cv.namedWindow("car_image",cv.WINDOW_NORMAL)
                    cv.imshow("car_image",car_image_show)

                    while True:
                        print("请输入s保存装甲板信息,如果不正确,按n进行下一个car的识别")
                        info_input=cv.waitKey()

                        input_number_armor=int(info_input)-48
                        if -1<input_number_armor<10:
                            #删除保存目标

                            absolute_onecar_armors.pop(input_number_armor)
                            onecar_armors.pop(input_number_armor)

                            #更新显示的界面
                            cv.destroyWindow("car_image")

                            car_image_show=car_image.copy()
                            for temp_i,armor in enumerate(onecar_armors):
                                #这里的armor的cls_pred已经正确了
                                armor_x1,armor_y1,armor_x2,armor_y2,conf,cls_conf, cls_pred=armor

                                #获取展示颜色
                                if cls_pred<=6:
                                    color=(0,0,255)

                                elif 6<cls_pred<=11:
                                    color=(255,0,0)

                                else:
                                    color=(0,255,0)

                                #获得装甲板数据
                                if cls_pred==13:
                                    armor_id=7#死亡展示的是7

                                elif cls_pred<=6:
                                    armor_id=cls_pred-1
                                elif 6<cls_pred<13:
                                    armor_id=cls_pred-6#看不清展示的是6
                                else:
                                    armor_id=6

                                show_text_info="{},{}".format(armor_id,temp_i)
                                cv.rectangle(car_image_show,(armor_x1,armor_y1),(armor_x2,armor_y2),color)
                                tools.show_text(show_text_info,(armor_x1,armor_y1),car_image_show,0.5,color)
                                print("完成了图片的绘制,绘制内容为")
                            cv.namedWindow("car_image",cv.WINDOW_NORMAL)
                            cv.imshow("car_image",car_image_show)


                        if info_input==ord('s'):
                            for armor in absolute_onecar_armors:
                                absolute_x1,absolute_y1,absolute_x2,absolute_y2,conf,cls_conf,cls_pred=armor

                                #绘制东西
                                show_text_info="{}".format(cls_pred)
                                if cls_pred<=6:
                                    #大图绘制
                                    cv.rectangle(image_save_show,(absolute_x1,absolute_y1),(absolute_x2,absolute_y2),(0,0,255),3)
                                    tools.show_text(show_text_info,(absolute_x1,absolute_y1),image_save_show,color=(0,0,255))


                                elif 6<cls_pred<=11:
                                    #大图绘制
                                    cv.rectangle(image_save_show,(absolute_x1,absolute_y1),(absolute_x2,absolute_y2),(255,0,0),3)
                                    tools.show_text(show_text_info,(absolute_x1,absolute_y1),image_save_show,color=(255,0,0))

                                else:
                                    #绿色绘制其他种类
                                    cv.rectangle(image_save_show,(absolute_x1,absolute_y1),(absolute_x2,absolute_y2),(0,255,0),3)
                                    tools.show_text(show_text_info,(absolute_x1,absolute_y1),image_save_show,color=(0,255,0))

                                save_infos.append((absolute_x1,absolute_y1,absolute_x2,absolute_y2,cls_conf,cls_pred))#保存装甲板
                                print("     成功写入装甲板保存信息:",absolute_x1,absolute_y1,absolute_x2,absolute_y2,cls_conf,cls_pred)

                            break

                        elif info_input==ord('n'):
                            print("     不保存这张图的结果")
                            break

                        elif info_input==ord('w'):
                            print("进行bboxes的选择,按ESC退出bboexes的选择")
                            cv.namedWindow("ROIs",cv.WINDOW_FULLSCREEN)
                            bboxes=cv.selectROIs("ROIs",car_image)
                            print("请在图片中输入种类id")
                            print("装甲板的对应id为:"
                                  "2   3   4   5   6   7   8   9   10  11   12     13"
                                  "红1,红2,红3,红4,红5,蓝1,蓝2, 蓝3, 蓝4, 蓝5  看不清  死亡"
                                  "")
                            while True:
                                category_id_input=cv.waitKey()
                                category_id_input=int(category_id_input)-48
                                if -1<category_id_input<10:#在所有种类中则完成输入默认的是不超过10个
                                    break

                                else:#避免输入其他错误参数
                                    print("请输入一个介于0到{}之间的数字".format(10))

                            print("多个bbox的选择为:[")
                            for bbox in bboxes:
                                x,y,w,h=bbox
                                x,y,w,h=int(x),int(y),int(w),int(h)
                                x1,y1,x2,y2=x,y,x+w,y+h#进行了保存
                                cv.rectangle(image_save_show,(x1,y1),(x2,y2),(0,255,0),2)
                                show_text_info="save:{} hand make".format(category_id_input)
                                tools.show_text(show_text_info,(x1,y1),image_save_show)

                                if (x1,y1,x2,y2,None,category_id_input) not in save_infos:
                                    save_infos.append((x1,y1,x2,y2,None,category_id_input))
                                    print("      [",x1,y1,x2,y2,None,category_id_input,"]")

                            print("\n")
                            cv.destroyWindow("ROIs")
                            cv.waitKey(1)


                        else:
                            print("请输入s保存装甲板信息,如果不正确,按n进行下一个car的识别")


                    print("\n")
                    print("*************装甲板识别保存完成********************")

            #更改前景图片(merge)
            if info_input==ord('m'):
                self.use_foreground=not self.use_foreground
                if self.use_foreground:
                    print("开始前景检测,并更新前景图为当前灰度图")
                    self.first_frame_gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)#把这张图片作为前景

                else:
                    print("前景检测关闭")

            #跳至上一张
            if info_input==ord('b'):
                return 'b'

            #退出标注
            if info_input==ord('q'):
                return 'q'

            #跳转操作
            if info_input==ord('j'):#用于进行index的跳转
                return 'j'

            #用于修改参数
            if info_input==ord('p'):
                #用于动态修改动态阈值识别阈值,但是不想加进去了,太累了
                #这个的整体实现到时候就类似于D435的rs-sensor-control的方法,之后实现了再来完成这个任务
                pass

    def See_Annotation_Mission(self,dataset_path):
        """
        用于进行标注信息的查看
        :param dataset_path:
        :return:
        """
        images_path=os.path.join(dataset_path,"images")
        annotations_path=os.path.join(dataset_path,"image_annotations")
        images_list=os.listdir(images_path)
        images_list.sort(key=lambda x: int(x[:-4]))  # 进行图片名称的排序

        index_id=0
        while index_id<len(images_list):
            filename=images_list[index_id][:-4]
            image_path=os.path.join(images_path,filename+".jpg")
            txt_path=os.path.join(annotations_path,filename+".txt")

            image=cv.imread(image_path)
            exist=tools.drawAnnotation(txt_path,image)
            if exist:
                print("{}.jpg文件存在标注文件".format(filename))
            else:
                print("***********{}.jpg文件不存在标注文件************".format(filename))
            cv.imshow("image",image)
            quit_flag=False
            while True:
                info_input=cv.waitKey()
                if info_input==13:
                    break

                if info_input==ord('b'):
                    index_id=index_id-2
                    break

                if info_input==ord('q'):
                    quit_flag=True
                    break

                if info_input==ord('j'):#用于进行index的跳转
                    jump_data=input("请输入要跳转的图片名称")
                    name=str(jump_data)+".jpg"
                    index_id=images_list.index(name)
                    print("即将跳转的id为:",index_id)
                    break

                #d是直接删除一个图片
                if info_input==ord('d'):
                    os.remove(image_path)
                    print("你删除了{}".format(image_path))

                    #这里另外更新一个image_list,避免参数出错
                    images_list=os.listdir(images_path)
                    images_list.sort(key=lambda x: int(x[:-4]))  # 进行图片名称的排序
                    break


            if quit_flag:
                break
            index_id=index_id+1

            cv.waitKey(1)
        cv.destroyAllWindows()

