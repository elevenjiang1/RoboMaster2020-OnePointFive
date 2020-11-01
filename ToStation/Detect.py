"""
这里面重新写Detect的代码,重构一下,要不然真的是写的太乱了,很多地方都不是很规范的东西
这里面主要用于进行矩形框的检测任务
"""

import cv2 as cv
import numpy as np
import math
import time
import tools
from tqdm import tqdm
import pyrealsense2 as rs


#用于进行不同的debug模式
global DEBUG_FLAG
global SHOW_IMAGE_FLAG
global USE_SERIAL_FLAG

DEBUG_FLAG=False
SHOW_IMAGE_FLAG=True
USE_SERIAL_FLAG=False

class RS:
    def __init__(self,open_depth=True,open_color=True,frame=30,resolution='1280x720'):
        """
        初始化相机
        :param open_depth: 是否开启深度
        :param open_color: 是否开启颜色
        :param frame: 帧率设置
        :param resolution: 像素值大小
        """
        #1:确定相机内参(从rs-sensor-control中获取),同时导入矩阵用于进行检测计算xyz图
        all_matrix=np.load("all_matrix.npz")#.npz文件在类中的generate_xy_matrix()的函数中生成
        if resolution=='640x480':
            self.image_width=640
            self.image_height=480
            self.fx=383.436
            self.fy=383.436
            self.cx=318.613
            self.cy=238.601
            self.x_matrix=all_matrix['x_matrix640']
            self.y_maxtrix=all_matrix['y_matrix640']
        elif resolution=='1280x720':
            self.image_width=1280
            self.image_height=720
            self.fx=639.059
            self.fy=639.059
            self.cx=637.688
            self.cy=357.688
            self.x_matrix=all_matrix['x_matrix1280']
            self.y_maxtrix=all_matrix['y_matrix1280']
        elif resolution=='848x480':
            self.image_width=848
            self.image_height=480
            self.fx=423.377
            self.fy=423.377
            self.cx=422.468
            self.cy=238.455
            self.x_matrix=all_matrix['x_matrix848']
            self.y_maxtrix=all_matrix['y_matrix848']
        else:
            assert False,"请输入正确的resolution值"

        #2:初始化一系列参数
        self.open_depth=open_depth
        self.open_color=open_color
        self.pipeline = rs.pipeline()#开启通信接口
        config_rs = rs.config()

        #2.1:使能深度和颜色图
        if open_depth:
            config_rs.enable_stream(rs.stream.depth, self.image_width, self.image_height, rs.format.z16, frame)
            self.depth_image=None
            self.color_map=None

        if open_color:
            config_rs.enable_stream(rs.stream.color,self.image_width, self.image_height, rs.format.bgr8, frame)
            self.color_image=None

        #2.2:开始通信流
        self.profile=self.pipeline.start(config_rs)

        #2.3:当RGB和深度同时开启时,将颜色图向深度图对齐
        if open_depth and open_color:
            align_to=rs.stream.depth
            self.align=rs.align(align_to)
        else:
            self.align=None


        #3:定义滤波器
        self.dec_filter=rs.decimation_filter(4)#降采样
        # self.temp_filter=rs.temporal_filter(3)#上下帧之间利用时间信息避免跳动,参数看官方文档
        self.hole_filter=rs.hole_filling_filter(2)#hole填充

    def get_data(self):
        """
        用于获取color_image和depth_image
        如果设定中没有就返回None
        :return:
        """
        #对齐帧并获取颜色和深度图帧
        frames=self.pipeline.wait_for_frames()
        if self.align is not None:
            frames=self.align.process(frames)#与深度图对齐

        #获取深度图
        if self.open_depth:
            depth_frame=frames.get_depth_frame()
            #使用滤波器处理
            hole_filtered=self.hole_filter.process(depth_frame)
            dec_filtered=self.dec_filter.process(hole_filtered)
            depth_image=np.asanyarray(dec_filtered.get_data())
            depth_image=cv.resize(depth_image,(self.image_width,self.image_height))

        else:
            depth_image=None



        #获取颜色图
        if self.open_color:
            color_frame=frames.get_color_frame()
            color_image=np.asanyarray(color_frame.get_data())
        else:
            color_image=None


        #生成为类中的图
        self.depth_image=depth_image
        self.color_image=color_image

        return color_image,depth_image

    def get_color_map(self,depth_image=None,range=None):
        """
        送入深度图,返回对应的颜色图
        :param depth_image:需要生成的颜色图,如果为None,则选取自带的深度图
        :param range: 是否需要滤除掉一定距离之后的值
        :return:
        """
        #没有深度图则直接采用类中原本的深度图
        if depth_image is None:
            depth_image=self.depth_image

        #有range要求则进行阈值操作
        range_image=depth_image.copy()
        if range is not None:
            depth_mask=cv.inRange(depth_image,0,range)
            if SHOW_IMAGE_FLAG:
                cv.imshow("depth_mask",depth_mask)
            range_image=depth_image*depth_mask/255

        #开始转深度图
        color_map=range_image.copy()
        cv.normalize(color_map,color_map,255,0,cv.NORM_MINMAX)
        color_map=color_map.astype(np.uint8)
        color_map=cv.applyColorMap(color_map,cv.COLORMAP_JET)
        self.color_map=color_map

        return color_map

    def get_xyz_image(self):
        """
        基于深度图,获取一张xyz_image的图,3通道,分别存放了该像素点的xyz值
        :return:xyz_image
        """
        xyz_image=np.array([self.x_matrix*self.depth_image,self.y_maxtrix*self.depth_image,self.depth_image])
        xyz_image=xyz_image.transpose((1,2,0))
        return xyz_image

    def get_xyz(self,point,range_area=2):
        """
        获取point点的xyz值
        当索引到边上时,会直接所以该点的Z值
        :param point:需要获取xyz的像素点
        :param range_area:取周围邻域的中间值
        :return:np.array((X,Y,Z))
        """
        u,v=point
        u=int(u)
        v=int(v)
        center_Z=[]
        #1:对center_Z进行排序,得到中值作为深度
        try:
            for x in range(-range_area,range_area+1):
                for y in range(-range_area,range_area+1):
                    center_Z.append(self.depth_image[v-y,u-x])#采用行列索引
            center_Z.sort()
            Z=center_Z[int(len(center_Z)/2)]
        except:
            try:
                Z=self.depth_image[v,u]
            except:
                Z=0

        #2:使用外参进行反解
        X=(u-self.cx)*Z/self.fx
        Y=(v-self.cy)*Z/self.fy
        return np.array((X,Y,Z))

    ##############################功能性函数####################################
    def generate_xy_matrix(self):
        """
        用于生成xyz_image的矩阵
        测距点本质上只与z有关,向平面的xy是固定的,由z进行比例放大缩小
        会在这个的目录下生成all_matrix.npz文件,其中包含了对应需要的xy比例的矩阵

        #使用方法:
        # data=np.load("all_matrix.npz")
        # x_640=data['x_matrix640']
        # x_1280=data['x_matrix1280']
        # print(x_640.shape)
        # print(x_1280.shape)
        :return:
        """

        #1:生成1280的矩阵
        x_1280_matrix=np.zeros((720,1280))
        y_1280_matrix=np.zeros((720,1280))
        fx=639.059
        fy=639.059
        cx=637.688
        cy=357.688
        for i in tqdm(range(1280)):
            for j in range(720):
                # print(temp_1280[j,i])#默认的索引是行列索引
                x_1280_matrix[j,i]=(i-cx)/fx
                y_1280_matrix[j,i]=(j-cy)/fy



        #2:生成640的矩阵
        x_640_matrix=np.zeros((480,640))
        y_640_matrix=np.zeros((480,640))
        fx=383.436
        fy=383.436
        cx=318.613
        cy=238.601
        for i in tqdm(range(640)):
            for j in range(480):
                x_640_matrix[j,i]=(i-cx)/fx
                y_640_matrix[j,i]=(j-cy)/fy



        #3:生成848x480的内参
        x_848_matrix=np.zeros((480,848))
        y_848_matrix=np.zeros((480,848))
        fx=423.377
        fy=423.377
        cx=422.468
        cy=238.455

        for i in tqdm(range(848)):
            for j in range(480):
                x_848_matrix[j,i]=(i-cx)/fx
                y_848_matrix[j,i]=(j-cy)/fy

        #保存对应的矩阵
        np.savez('all_matrix.npz',x_matrix640=x_640_matrix,y_matrix640=y_640_matrix,x_matrix1280=x_1280_matrix,y_matrix1280=y_1280_matrix,x_matrix848=x_848_matrix,y_matrix848=y_848_matrix)

    def check_distance(self,roi_size=15):
        """
        用于进行相机深度值确定
        @param roi_size: 定义roi的长宽,从而知道多少范围的roi合适
        @return:
        """
        while True:
            color_image,depth_image=self.get_data()

            #查看测距是否准确,随机取几个点,然后进行测距,看看效果
            color_map=self.get_color_map(depth_image,10000)

            #获取图像中心点
            h,w=depth_image.shape
            h=int(h/2)
            w=int(w/2)

            #生成一个区域进行测距
            xyz_image=self.get_xyz_image()
            roi_w=roi_size
            roi_h=roi_size
            middle_roi=xyz_image[h-roi_h:h+roi_h,w-roi_w:w+roi_w]#得到中心区域的xyz值
            middle_roi=middle_roi.reshape(-1,3)

            #对选取区域求平均之后去除掉方差以外的值
            mean=np.mean(middle_roi[:,2])
            std=np.std(middle_roi[:,2])
            origin_number=len(middle_roi)
            correct_middle_roi=abs(middle_roi[:,2]-mean)<0.8*std#在其内部的roi值
            middle_roi=middle_roi[correct_middle_roi]
            new_number=len(middle_roi)
            print("选取剩余0.8个方差之后的值有:{},剩余值占原来的{:.2f}%".format(new_number,new_number/origin_number*100))

            #得到最终的测试距离
            mean_distance=np.mean(middle_roi[:,2])#获取正确的xyz值

            #最后输出测距距离
            print("中心的距离为:",mean_distance)
            color_map[h-roi_h:h+roi_h,w-roi_w:w+roi_w]=(0,0,255)

            cv.imshow("depth_image",depth_image)
            cv.imshow("color_map",color_map)
            cv.waitKey(0)

class DetectStation:
    def __init__(self):
        #相机参数
        self.color_image=None
        self.depth_image=None
        self.color_map=None
        self.camera=RS(open_depth=True,open_color=True,frame=30)

        #识别参数
        self.error=50#允许最大误差
        self.target_legnth=900#目标尺寸

        #上一帧识别结果
        self.last_point=None

    def get_pointrect_mask(self,point=None,rect_w=50,rect_h=30,show_image=False):
        """
        送入深度图,基于point的点生成矩形,对矩形的所处平面进行mask的分割
        :param point: 矩形中心点,如果不指定则为全图中心
        :param rect_w: 矩形w
        :param rect_h: 矩形h
        :param show_image:展示图片
        :return: 识别的mask
        """
        #1:获取xyz矩阵
        xyz_image=self.camera.get_xyz_image()

        #2:生成搜索的ROI
        point_x,point_y=point
        roi=xyz_image[point_y-rect_h:point_y+rect_h,point_x-rect_w:point_x+rect_w]
        roi=roi.reshape(-1,3)#用于统一尺寸

        #3:开始基于mask找到目标
        #计算深度值,滤除超过2*方差的点
        mean=np.mean(roi[:,2])
        std=np.std(roi[:,2])
        correct_index=abs(roi[:,2]-mean)<2*std#进行一次滤波,避免最小二成效果不好
        filtered_roi=roi[correct_index]#索引误差不超过2*std的

        #4:求取平面矩阵
        if np.linalg.det(filtered_roi.T@filtered_roi)==0:
            return np.zeros(xyz_image.shape,dtype=np.uint8)#如果矩阵出现逆解则直接返回全黑的Mask

        Y=-np.ones(filtered_roi.shape[0])
        plane_param=np.linalg.inv(filtered_roi.T@filtered_roi)@filtered_roi.T@Y#生成平面参数
        all=math.sqrt(plane_param[0]*plane_param[0]+plane_param[1]*plane_param[1]+plane_param[2]*plane_param[2])#计算平面的x^2+y^2+z^2

        distance_array=abs(xyz_image.dot(plane_param)+1)/all#获取与生成平面距离的值
        mask=cv.inRange(distance_array,0,self.error)#mask允许的误差

        #5:进行形态学操作
        processed_mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,tools.generate_kernel(20,20))#形态学操作提升结果
        processed_mask=cv.morphologyEx(processed_mask,cv.MORPH_OPEN,tools.generate_kernel(100,30))

        if show_image:
            self.color_map=self.camera.get_color_map()
            self.color_map[point_y-rect_h:point_y+rect_h,point_x-rect_w,point_x+rect_w]=(0,0,255)
            if SHOW_IMAGE_FLAG:
                cv.imshow("{},{}mask".format(point_x,point_y),self.color_map)

        return processed_mask

    def check_x_correct(self,middle_xyzs):
        """
        送入中心矩形框的xyz值,查看是否为目标
        :param middle_xyzs:中心矩形的xyzs
        :return:
        """
        distance1=tools.get_distance(middle_xyzs[0],middle_xyzs[3])
        distance2=tools.get_distance(middle_xyzs[1],middle_xyzs[2])

        X_temp=(distance1+distance2)/2
        X_correct=abs(X_temp-self.target_legnth/2)<self.error

        if DEBUG_FLAG:
            print("正确的X_temp为:",X_temp,"期望的距离为:",self.target_legnth/2)
            print("检测到的rect的X_temp为:",X_temp,"期望的距离为:",self.target_legnth/2)

        return X_correct

    def check_rect(self,point,rect):
        """
        基于找到的矩形进行多重确定,确保是目标值
        :param point:寻找的中心点
        :param rect: 矩形
        :return:
        """
        #1:确保矩形是包含点的
        in_rect_flag=tools.is_in_rect(point,rect)
        if not in_rect_flag:
            return None,[],[]

        #2:确定矩形的xyz符合要求
        fourpoints=cv.boxPoints(rect)
        correct_points=tools.sort_four_points(fourpoints,rect[0])

        middle_points=[]
        middle_xyzs=[]
        for point in correct_points:#获取每个点的值
            if point is None:#为什么要加这个限制?这个不懂
                if DEBUG_FLAG:
                    print("有时候会进入到这个的状态,这里面的输出为:",correct_points)
                return False,middle_points,middle_xyzs
            middle_points.append(tools.get_middle(rect[0],point))
            middle_xyzs.append(self.camera.get_xyz(middle_points[-1]))


        #2.2:确保x方向符合要求
        return self.check_x_correct(middle_xyzs),middle_points,middle_xyzs

    def get_best_rect(self,rects_list,rects_xyz_list,rects_center):
        """
        对于找到的一堆矩形选取最为正确的
        :param rects_list: 矩形4个点的list
        :param rects_xyz_list: 矩形xyz的4个点
        :return:
        """
        if len(rects_list)==1:
            return rects_list[0],rects_xyz_list[0],rects_center[0]

        elif len(rects_list)==0:
            return None,None,None

        else:#多于1个的情况
            if DEBUG_FLAG:
                print("出现了多个矩形的情况,他们的矩形情况为:")
                print(rects_list)
                print(rects_xyz_list)
            correct=self.error
            correct_i=0
            for i,middle_xyzs in enumerate(rects_xyz_list):
                distance1=tools.get_distance(middle_xyzs[0],middle_xyzs[3])
                distance2=tools.get_distance(middle_xyzs[1],middle_xyzs[2])

                temp=abs((distance1+distance2)/2-self.target_legnth/2)-self.error
                if temp<correct:
                    correct_i=i

            return rects_list[correct_i],rects_xyz_list[correct_i],rects_center[correct_i]

    def get_target_frommask(self,mask,center_point):
        """
        :param mask:找到的mask寻找矩形框
        :param center_point: 基于中心点寻找的mask
        送入mask,找到对应的尺寸
        :return:
        """
        rects_list=[]
        rects_xyz_list=[]
        rects_center=[]
        try:
            contours,hierarchy=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        except:
            return False,None,None


        for contour in contours:
            #2.1:筛选掉小的矩形
            if cv.contourArea(contour)<30000:
                continue#区域太小直接跳过

            #2.2:确保找到的mask都在矩形中
            rect=cv.minAreaRect(contour)
            find_station_flag,middle_points,middle_xyzs=self.check_rect(center_point,rect)
            if find_station_flag:
                rects_list.append(middle_points)
                rects_xyz_list.append(middle_xyzs)
                rects_center.append((int(rect[0][0]),int(rect[0][1])))

        target_rect,target_rect_xyz,target_center=self.get_best_rect(rects_list,rects_xyz_list,rects_center)
        if target_rect is not None:
            #找到了资源岛,更新last_point
            self.last_point=target_center
            return True,target_rect,target_rect_xyz
        else:
            return False,None,None

    def get_station(self,depth_image):
        """
        送入深度图,目标获得资源岛的xyz三个值
        如果没有找到资源岛,则后两个返回值为None
        :param depth_image:
        :return:Find_Flag(是否找到资源岛),target_rect(目标矩形),target_rect_xyz(目标矩形的xyz)
        """
        #1:首先现在上一帧的地方找资源岛,找到则返回True,target_rect,target_rect_xyz
        if self.last_point is not None:
            last_point=self.last_point
            last_mask=self.get_pointrect_mask(last_point)
            find_flag,target_rect,target_rect_xyz=self.get_target_frommask(last_mask,last_point)
            if find_flag:
                return find_flag,target_rect,target_rect_xyz#找到就返回目标

        #2:若上一帧没找到点,则从中心区域开始找,找到就直接返回值
        h,w=depth_image.shape
        center_point=(int(w/2),int(h/2)+60)
        center_mask=self.get_pointrect_mask(center_point)
        find_flag,target_rect,target_rect_xyz=self.get_target_frommask(center_mask,center_point)
        if find_flag:
            return find_flag,target_rect,target_rect_xyz#找到就返回目标

        #5:中心区域仍未找到,则采用一个For循环进行寻找,不断更新中心点
        for i in range(10):
            all_point=(int(w/2+60*(i-5)),int(h/2)+60)
            right_mask=self.get_pointrect_mask(all_point)
            find_flag,target_rect,target_rect_xyz=self.get_target_frommask(right_mask,all_point)
            if find_flag:
                return find_flag,target_rect,target_rect_xyz#找到就返回目标


        #5:如果都没有找到,则返回False
        return False,None,None

##############################样例代码####################################
def get_camera_data():
    """
    用于显示相机图像
    @return:
    """
    camera=RS()
    while True:
        color_image,depth_image=camera.get_data()
        color_map=camera.get_color_map()
        cv.imshow("color_image",color_image)
        cv.imshow("color_map",color_map)
        cv.waitKey(1)

def detect_station():
    """
    用于寻找资源岛样例代码
    @return:
    """
    detectStation=DetectStation()#识别类
    while True:
        #1:获取颜色图,深度图,和深度图对应的color_map
        color_image,depth_image=detectStation.camera.get_data()
        color_map=detectStation.camera.get_color_map()

        #2:寻找资源岛
        find_station_flag,station_rect,station_xyz=detectStation.get_station(depth_image)
        if find_station_flag:
            for i in range(4):
                cv.line(color_map,tuple(station_rect[i]),tuple(station_rect[(i+1)%4]),(255,255,255),2)

        #3:进行结果展示
        show_image=cv.hconcat([color_image,color_map])
        cv.namedWindow("result",cv.WINDOW_NORMAL)
        cv.imshow("result",show_image)
        cv.waitKey(1)


if __name__ == '__main__':
    # get_camera_data()#用于测试相机是否能正常开启
    detect_station()#用于测试是否能够检测出资源岛

