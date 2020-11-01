"""
这里面用于存放一些常用的工具
"""
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
from Camera import RS
from tqdm import tqdm
import math

global DEBUG_FLAG
global SHOW_IMAGE_FLAG
global USE_SERIAL_FLAG

DEBUG_FLAG=False
SHOW_IMAGE_FLAG=False
USE_SERIAL_FLAG=False



##################################################功能性函数##################################################
def check_distance():
    """
    用于进行距离的测试操作,标定摄像头用
    :return:
    """
    camera=RS(open_color=True,open_depth=True,frame=30,resolution='640x480')
    while True:
        color_image,depth_image=camera.get_data()

        #查看测距是否准确,随机取几个点,然后进行测距,看看效果
        color_map=camera.get_color_map(depth_image)

        #进行点的XYZ显示,选择3个点
        h,w=depth_image.shape
        h=int(h/2)
        w=int(w/2)
        left_point=(w-50,h)
        right_point=(w+50,h)
        down_point=(w,h)
        cv.circle(color_map,left_point,3,(0,0,255),2)
        cv.circle(color_map,right_point,3,(0,0,255),2)
        cv.circle(color_map,down_point,3,(0,0,255),2)

        cv.circle(color_image,left_point,3,(0,0,255),2)
        cv.circle(color_image,right_point,3,(0,0,255),2)
        cv.circle(color_image,down_point,3,(0,0,255),2)


        left_xyz=camera.get_xyz(left_point[0],left_point[1])
        right_xyz=camera.get_xyz(right_point[0],right_point[1])

        cv.putText(color_map,"{:.2f},{:.2f},{:.2f}".format(left_xyz[0],left_xyz[1],left_xyz[2]),(left_point[0],left_point[1]+30),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv.putText(color_map,"{:.2f},{:.2f},{:.2f}".format(right_xyz[0],right_xyz[1],right_xyz[2]),(right_point[0],right_point[1]-30),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        distance=left_xyz[0]-right_xyz[0]
        cv.putText(color_map,"{:.3f}".format(distance),(200,400),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
        cv.putText(color_image,"{:.3f}".format(distance),(200,400),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        cv.imshow("image",color_image)
        cv.imshow("depth_image",depth_image)

        cv.imshow("color_map",color_map)
        cv.waitKey(0)

def check_distance_better():
    camera=RS(open_color=False,open_depth=True,frame=30,resolution='1280x720')
    while True:

        color_image,depth_image=camera.get_data()

        #查看测距是否准确,随机取几个点,然后进行测距,看看效果
        color_map=camera.get_color_map(depth_image,5000)

        #进行点的XYZ显示,选择3个点
        h,w=depth_image.shape
        h=int(h/2)
        w=int(w/2)


        #生成一个区域进行测距
        xyz_image=camera.get_xyz_image()
        roi_w=15
        roi_h=15

        middle_roi=xyz_image[h-roi_h:h+roi_h,w-roi_w:w+roi_w]#得到中心区域的xyz值
        middle_roi=middle_roi.reshape(-1,3)


        mean=np.mean(middle_roi[:,2])
        std=np.std(middle_roi[:,2])
        origin_number=len(middle_roi)

        correct_middle_roi=abs(middle_roi[:,2]-mean)<0.8*std#在其内部的roi值
        middle_roi=middle_roi[correct_middle_roi]

        new_number=len(middle_roi)
        print("选取剩余0.8个方差之后的值有:{},剩余值占原来的{:.2f}%".format(new_number,new_number/origin_number*100))


        mean_distance=np.mean(middle_roi[:,2])#获取正确的xyz值



        print("中心的距离为:",mean_distance)
        color_map[h-roi_h:h+roi_h,w-roi_w:w+roi_w]=(0,0,255)

        cv.imshow("depth_image",depth_image)
        cv.imshow("color_map",color_map)
        cv.waitKey(0)

def make_matrix():
    """
    用于生成xyz_image的矩阵
    测距点本质上至于z有关,向平面的xy是固定的,有z进行放大缩小
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

##################################################常用函数##################################################
def generate_kernel(x,y):
    return np.ones([x,y],dtype=np.uint8)

def get_color_map(depth_image):
    """
    送入深度图,返回对应的颜色图
    RS类中有更好的深度图生成函数
    :param depth_image: 深度图
    :return:
    """
    color_map=depth_image.copy()
    cv.normalize(color_map,color_map,255,0,cv.NORM_MINMAX)
    color_map=color_map.astype(np.uint8)
    color_map=cv.applyColorMap(color_map,cv.COLORMAP_JET)

    return color_map

def get_middle(point1,point2):
    """
    由两个点获取他们的中心点
    :param point1:
    :param point2:
    :return:
    """
    middle_point=(int((point1[0]+point2[0])/2),int((point1[1]+point2[1])/2))
    return middle_point

def get_distance(xyz1,xyz2):
    """
    获取两个xyz的之间距离
    :param xyz1:
    :param xyz2:
    :return:
    """
    distance=math.sqrt(pow((xyz1[0]-xyz2[0]),2)+pow((xyz1[1]-xyz2[1]),2)+pow((xyz1[2]-xyz2[2]),2))
    return distance

def get_center_fromfourpoints(fourpoints):
    """
    送入四个点,获取这四个点的中心点,用于排列四个点用
    :param fourpoints: 四个点,排列为:[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :return:
    """
    x=fourpoints[0][0]+fourpoints[1][0]+fourpoints[2][0]+fourpoints[3][0]
    y=fourpoints[0][1]+fourpoints[1][1]+fourpoints[2][1]+fourpoints[3][1]
    center=(int(round(x/4)),int(round(y/4)))
    return center

def sort_four_points(fourpoints,center=None):
    """
    对旋转矩形的四个值进行排序
    送入的point是列行排布,即x,y排布
    四个点顺时针排布,左下角为0,
    :param fourpoints: 四个点,排布为:[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :param center:
    :return:
    """
    if DEBUG_FLAG:
        print("center:",center)
        print("fourpoint",fourpoints)

    correct_points=[None,None,None,None]

    if center is None:
        center=get_center_fromfourpoints(fourpoints)

    for point in fourpoints:
        if point[1]<center[1]:
            if point[0]<center[0]:
                correct_points[1]=point
            else:
                correct_points[2]=point

        else:
            if point[0]<center[0]:
                correct_points[0]=point
            else:
                correct_points[3]=point

    for i,point in enumerate(correct_points):#
        if point is None:
            correct_points[i]=(0,0)

    return correct_points

def is_in_rect(point,rect):
    """
    用于判断point是否在rect中
    :param point:
    :param rect:
    :return:
    """
    fourpoints=cv.boxPoints(rect)
    fourpoints=sort_four_points(fourpoints,rect[0])#获取正确的四个矩形点
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]=fourpoints#获取到4个点的xy值,从左下开始
    point_x,point_y=point

    if min(x1,x2)<point_x<max(x3,x4) and min(y2,y3)<point_y<max(y1,y4):
        return True
    else:
        return False



if __name__ == '__main__':
    check_distance_better()