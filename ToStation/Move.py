"""
基于Detect的信息,进行目标的移动
"""
import cv2 as cv
import tools
from Camera import RS
from Detect import Detect
from Detect import DetectStation
from Message import MessageProcesser
import math
import serial
import numpy as np
import time

#用于进行不同的debug模式
global DEBUG_FLAG
global SHOW_IMAGE_FLAG
global USE_SERIAL_FLAG

DEBUG_FLAG=True
SHOW_IMAGE_FLAG=True
USE_SERIAL_FLAG=False


class Move:
    def __init__(self):
        self.detectStation=DetectStation()
        if USE_SERIAL_FLAG:
            self.messageProcesser=MessageProcesser()

        #pid的记录参数
        self.pid_x_i=[]#x的积分
        self.pid_y_i=[]#y的积分
        self.pid_z_i=[]#z的积分

    def process_rect(self,station_rect,station_xyz):
        """
        用于处理资源岛的矩形
        最终得到x,y,z需要更改的值,速度先不管
        :param station_rect:矩形的四个点(而非2个点四个值)
        :param station_xyz:矩形的四个点对应的xyz
        :return:
        """
        leftdown,leftup,rightup,rightdown=station_rect#这里得到的是xy,而非wh
        leftdown_xyz,leftup_xyz,rightup_xyz,rightdown_xyz=station_xyz

        #获取中心点的xyz值
        left_center=tools.get_middle(leftdown,leftup)
        up_center=tools.get_middle(leftup,rightup)
        # if DEBUG_FLAG:
        #     print("左边中心:",left_center,"右边中心为:",up_center)
        center=(up_center[0],left_center[1])
        center_xyz=self.detectStation.camera.get_xyz(center)


        #获取左右的转角
        x1,y1,z1=leftup_xyz
        x2,y2,z2=rightup_xyz

        distance=tools.get_distance(leftup_xyz,rightup_xyz)
        temp_z=z1-z2
        xita_rad=math.asin(temp_z/distance)

        xita=xita_rad*180/math.pi

        return center,center_xyz,xita

    def pid_x(self,x,target=0,K=10,I=1):
        """
        进行pid的x的调节
        这里面先加入一个PI控制
        :return:
        """
        x=x-target

        if len(self.pid_x_i)>100:
            self.pid_x_i.pop(0)
            self.pid_x_i.append(x)#筛入一个新的pid

        pid_out=K*x+I*sum(self.pid_x_i)
        return pid_out

    def pid_y(self,y,target=0,K=10,I=1):
        """
        进行pid的x的调节
        这里面先加入一个PI控制
        :return:
        """
        y=y-target

        if len(self.pid_x_i)>100:
            self.pid_y_i.pop(0)
            self.pid_y_i.append(y)#筛入一个新的pid

        pid_out=K*y+I*sum(self.pid_y_i)
        return pid_out

    def pid_z(self,z,target=0,K=10,I=1):
        """
        进行pid的x的调节
        这里面先加入一个PI控制
        :return:
        """
        z=z-target

        if len(self.pid_z_i)>100:
            self.pid_z_i.pop(0)
            self.pid_z_i.append(z)#筛入一个新的pid

        pid_out=K*z+I*sum(self.pid_z_i)
        return pid_out

    def move_to_station(self):
        """
        这个函数用于移动到目标的信息发送
        采用PID进行控制,3个方向单独pid到0就是目标
        :return:
        """
        #1:寻找资源岛
        color_image,depth_image=self.detectStation.camera.get_data()
        color_map=self.detectStation.camera.get_color_map()
        find_station_flag,station_rect,station_xyz=self.detectStation.get_station(depth_image)

        #找到资源岛的情况下,返回资源岛xyz
        if find_station_flag:
            center,center_xyz,xita=self.process_rect(station_rect,station_xyz)

            x,y,z=center_xyz[2],center_xyz[0],xita
            if DEBUG_FLAG:
                print("资源岛的XYZ为:x:{:.2f},y:{:.2f},z:{:.2f}".format(x,y,z))

            x_out=self.pid_x(x,target=600)#500mm之后就检测不到资源岛了
            y_out=-self.pid_y(y,K=5)
            z_out=self.pid_z(z)

            puttext="x:{:.2f},y:{:.2f},z:{:.2f}".format(x_out,y_out,z_out)
            print("移动的目标为:",puttext)
            if USE_SERIAL_FLAG:
                send_msg=self.messageProcesser.get_send_msg(function_word=1,x=x_out,y=y_out,z=0,max_value=3000)#先控制x
                self.messageProcesser.USB0.write(send_msg)

            if SHOW_IMAGE_FLAG:
                cv.circle(color_map,center,3,(0,0,255),2)
                cv.putText(color_map,puttext,(30,30),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                cv.putText(color_map,"{:.2f},{:.2f},{:.2f}".format(tuple(center_xyz)[0],tuple(center_xyz)[1],tuple(center_xyz)[2]),center,cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                cv.putText(color_map,"{:.3f}".format(xita),(center[0],center[1]+100),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                for i in range(4):
                    cv.line(color_map,tuple(station_rect[i]),tuple(station_rect[(i+1)%4]),(255,255,255),2)

        else:
            print("没有发现目标")
            if USE_SERIAL_FLAG:
                send_msg=self.messageProcesser.get_send_msg(function_word=1,x=0,y=0,z=0)#先控制x
                self.messageProcesser.USB0.write(send_msg)

        if SHOW_IMAGE_FLAG:
            show_image=cv.hconcat([color_image,color_map])
            cv.namedWindow("result",cv.WINDOW_NORMAL)
            cv.imshow("result",show_image)
            cv.waitKey(1)

if __name__ == '__main__':
    move=Move()
    while True:
        print("****************开始执行****************************")
        move.move_to_station()



