"""
数据增强类,基于送入的图片进行数据增强,增强的数据相比于原来的数据前缀增加00x_xxx的文件
这里面的类直接基于生成的文件进行尝试

数据增强类,目前主要的数据增强是颜色变化和bbox移位两个操作
"""
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import albumentations as A
import random
import sys
import tools
from tqdm import tqdm
import shutil

class DataAugment:
    def __init__(self,dataset_path=None,aug_path=None):
        #确保路径全部存在
        assert dataset_path is not None,"数据增强类请初始化数据路径"
        assert aug_path is not None,"数据增强类请初始化生成文件夹路径"
        
        self.dataset_path=dataset_path
        self.images_path=self.dataset_path+"/images"
        self.annotations_path=self.dataset_path+"/image_annotations"

        self.aug_path=aug_path
        self.aug_images_path=self.aug_path+"/images"
        self.aug_annotations_path=self.aug_path+"/image_annotations"

        assert os.path.exists(self.images_path),"图片保存路径不存在"
        assert os.path.exists(self.annotations_path),"保存路径不存在"

    def move_origin_to_aug(self):
        print("进行图片移动操作")
        #直接把两个文件夹复制过去,不需要重命名
        if os.path.exists(self.aug_images_path):
            print("已存在数据增强数据,进行删除")
            shutil.rmtree(self.aug_images_path)#删除图片路径
            shutil.copytree(self.images_path,self.aug_images_path)#直接弄过去
        else:
            shutil.copytree(self.images_path,self.aug_images_path)

        if os.path.exists(self.aug_annotations_path):
            print("已存在数据增强数据,进行删除")
            shutil.rmtree(self.aug_annotations_path)
            shutil.copytree(self.annotations_path,self.aug_annotations_path)
        else:
            shutil.copytree(self.annotations_path,self.aug_annotations_path)
        print("完成数据复制任务")
        
    def saveAug(self,save_number,image,save_txt_info):
        """
        用于保存到aug文件下
        :param save_number:
        :param image:
        :param save_txt_info:
        :return:
        """
        new_filename_name=str(save_number)
        new_image_name=new_filename_name+".jpg"
        new_txt_name=new_filename_name+".txt"

        save_image_path=os.path.join(self.aug_images_path,new_image_name)
        save_txt_path=os.path.join(self.aug_annotations_path,new_txt_name)

        cv.imwrite(save_image_path,image)#进行颜色保存
        new_txt_file=open(save_txt_path,'w')
        for line in save_txt_info:
            category_id,x1,y1,x2,y2=line
            save_info="{},{},{},{},{},\n".format(category_id,int(x1),int(y1),int(x2),int(y2))
            new_txt_file.write(save_info)

        new_txt_file.close()

    def getAug(self,aug, min_area=0., min_visibility=0.65):
        """
        生成一个Compose
        :param aug:给一个增强方法的list
        :param min_area: 允许bbox的最小区域
        :param min_visibility: 允许bbox相比于原来bbox缩小的范围
        :return:
        """
        return A.Compose(aug, bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area,
                                                   min_visibility=min_visibility, label_fields=['category_id']))#用于生成一个增强类,Compose的操作

    def aug_color_motion(self):
        """
        进行亮度变换和运动模糊增强
        :return:
        """
        
        #1:读取图片
        image_list=os.listdir(self.images_path)
        image_list.sort(key=lambda x:int(x[:-4]))

        aug_list=os.listdir(self.aug_images_path)
        aug_list.sort(key=lambda x:int(x[:-4]))
        save_number=int(aug_list[-1][:-4])+1

        #2:每张图片进行数据增强
        print("进行颜色运动模糊数据增强")
        for all_i,image_name in tqdm(enumerate(image_list)):
            #2.1:处理图片路径
            filename=image_name[:-4]
            image_path=os.path.join(self.images_path,filename+".jpg")
            txt_path=os.path.join(self.annotations_path,filename+".txt")
            image=cv.imread(image_path)
            txt_file=open(txt_path)
            txt_info=txt_file.readlines()
            txt_file.close()

            #2.2:进行数据增强
            bboxes=[]#用于保存bboxes信息,用于绘图
            category_ids=[]#用于保存种类信息

            for line in txt_info:
                ann=line.split(',')[:-1]
                ann_int=map(int,ann)
                number_ann=list(ann_int)#变换成了数字的list
                bboxes.append(number_ann[1:])
                category_ids.append(number_ann[0])

            #进行数据增强选项添加
            aug=self.getAug([
                A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.3),
                A.MotionBlur(blur_limit=7,p=0.3),#进行运动模糊
                A.MotionBlur(blur_limit=25,p=0.05)#进行比较大的运动模糊
            ])

            #得到增强结果
            augmented=aug(image=image,bboxes=bboxes,category_id=category_ids)#augmented是一个dict,包含了image,bboxes,category_id
            aug_img=augmented["image"]
            aug_bboxes=augmented["bboxes"]
            aug_category=augmented["category_id"]

            #3:进行增强结果的保存
            save_txt_info=[]
            for i,bbox in enumerate(aug_bboxes):
                x1,y1,x2,y2=bbox
                category_id=aug_category[i]
                save_txt_info.append((category_id,x1,y1,x2,y2))


            self.saveAug(save_number,aug_img,save_txt_info)
            save_number=save_number+1
            sys.stdout.write("\r当前进行了{}张图片的生成,占总图片的{:.1f}%".format(all_i,all_i/len(image_list)*100))

    def aug_more_bboxes(self,iou_score=0.2,try_times=20,small_bbox_size=625):
        """
        #PMJ的iou新增避免了遮挡问题,但是小目标增多的情况没有加入,大小目标情况都一样,这里新增一个数据增强,把小目标的情况同时考虑
        增加更多的bboxes,同时提升小目标的增加概率
        @return:
        """
        #1:读取图片
        image_list=os.listdir(self.images_path)
        image_list.sort(key=lambda x:int(x[:-4]))

        aug_list=os.listdir(self.aug_images_path)
        aug_list.sort(key=lambda x:int(x[:-4]))
        save_number=int(aug_list[-1][:-4])+1#基于最后一张数据开始移动


        #2:开始进行数据增强
        print("开始进行数据增强")
        for all_i,image_name in tqdm(enumerate(image_list)):
            #2.1:读取txt和image
            filename=image_name[:-4]
            image_path=os.path.join(self.images_path,image_name)
            txt_path=os.path.join(self.annotations_path,filename+".txt")
            image=cv.imread(image_path)
            txt_file=open(txt_path)
            txt_info=txt_file.readlines()
            txt_file.close()


            #2.2:获取原来的bbox
            ROIs=[]#用于存储所有的bbox
            aug_image=image.copy()
            save_txt_info=[]#txt保存的信息
            for line in txt_info:
                #解析原来的标注信息
                ann=line.split(',')[:-1]
                ann_int=map(int,ann)#从string变为int
                number_ann=list(ann_int)
                category_id,x1,y1,x2,y2=number_ann

                #保存标注信息
                roi=image[y1:y2,x1:x2]
                bbox=(x1,y1,x2,y2)
                bbox_size=(x2-x1)*(y2-y1)
                if bbox_size<small_bbox_size:
                    target_change_time=3
                else:
                    target_change_time=1
                save_txt_info.append((category_id,x1,y1,x2,y2))#保存原有的bbox
                ROIs.append({"roi_img":roi,"bbox":bbox,"bbox_size":bbox_size,"category_id":category_id,"target_change_time":target_change_time})


            #2.3:进行每个roi的增强,确保每一个不进行遮挡
            change_flag=False
            for roi in ROIs:
                iou_correct_flag=True#先认为每个新增roi都是可以添加的
                target_change_time=roi["target_change_time"]
                change_time=0

                #2.3.1:为了更有可能地添加新目标,可以提升尝试次数
                for try_time in range(try_times):
                    #原来的roi
                    category_id=roi['category_id']
                    roi_h,roi_w,roi_c=roi['roi_img'].shape
                    image_h,image_w,image_c=image.shape

                    #获取新生成的bbox
                    x_n=random.random()
                    y_n=random.random()
                    new_x1=int(x_n*image_w)
                    new_y1=int(y_n*image_h)
                    new_x2=new_x1+roi_w
                    new_y2=new_y1+roi_h

                    #当新的bbox超出图片范围,则继续生成新的roi
                    if new_x2>image_w or new_y2>image_h:
                        continue

                    #与已经存在的roi做匹配,确保新的bbox和原来的bbox的iou值不超过iou_score
                    for every_roi in save_txt_info:
                        every_category_id,every_x1,every_y1,every_x2,every_y2=every_roi
                        iou_w=min(every_x1,every_x2,new_x1,new_x2)+roi_w+(new_x2-new_x1)-max(every_x1,every_x2,new_x1,new_x2)#IOU_W=min(x1,x2,x3,x4)+w1+w2-max(x1,x2,x3,x4)
                        iou_h=min(every_y1,every_y2,new_y1,new_y2)+roi_h+(new_y2-new_y1)-max(every_y1,every_y2,new_y1,new_y2)#IOU_H=min(y1,y2,y3,y4)+h1+h2-max(y1,y2,y3,y4)

                        every_size=(every_x2-every_x1)*(every_y2-every_y1)
                        new_size=(new_x2-new_x1)*(new_y2-new_y1)
                        iou_size=iou_w*iou_h
                        try:#避免除以0
                            iou=iou_size/(new_size+every_size-iou_size)#两个iou的size相同
                        except:
                            continue

                        if iou>iou_score:
                            iou_correct_flag=False
                            break#不用再判断了,生成的不成功,try下一次

                    #如果筛选之后,iou_correct_flag仍然为正确,说明生成成功
                    if iou_correct_flag:
                        change_time=change_time+1
                        new_x1,new_y1,new_x2,new_y2=int(new_x1),int(new_y1),int(new_x2),int(new_y2)
                        aug_image[new_y1:new_y2,new_x1:new_x2]=roi['roi_img']
                        save_txt_info.append((category_id,new_x1,new_y1,new_x2,new_y2))
                        change_flag=True


                    #满足增强次数之后,则可以退出,否则继续增强
                    if change_time>=target_change_time:
                        break


            #当发生了改动,则生成新图片
            if change_flag:
                self.saveAug(save_number,aug_image,save_txt_info)
                save_number=save_number+1

            sys.stdout.write("\r当前进行了{}张图片的生成,占总图片的{:.1f}%".format(all_i,all_i/len(image_list)*100))

    def seeResult(self):
        image_list=os.listdir(self.images_path)
        image_list.sort(key=lambda x:int(x[:-4]))

        for all_i,name in enumerate(image_list):
            try:
                print("当前查看的图片为:",name)

                filename=name[:-4]
                image_path=os.path.join(self.images_path,name)
                txt_path=os.path.join(self.annotations_path,filename+".txt")
                image=cv.imread(image_path)
                txt_file=open(txt_path)
                txt_info=txt_file.readlines()
                txt_file.close()
                ann_infos=[]

                for line in txt_info:
                    ann=line.split(',')[:-1]
                    ann_int=map(int,ann)
                    number_ann=list(ann_int)

                    category_id,x1,y1,x2,y2=number_ann
                    print("当前的bbox的size为:",(x2-x1)*(y2-y1))
                    print("*************")
                    ann_infos.append((category_id,x1,y1,x2,y2))

                tools.visualize(ann_infos,image)
                cv.waitKey(0)
            except:
                print("发生错误,错误位置在名称{}".format(name))
                cv.waitKey(1)

        cv.destroyAllWindows()

if __name__ == '__main__':
    dataaugment=DataAugment("/home/elevenjiang/Documents/Project/RM/Code/SmallMap/MakeData/Code/MakeMore")#指定dataset的主路径
    # dataaugment.aug_color_motion()
    # dataaugment.aug_more_bboxes()
    dataaugment.seeResult()#生成
