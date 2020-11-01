"""
对位的信息传输类
"""
import serial
import serial.tools.list_ports
import struct
import time
from ctypes import c_ubyte,c_float,Structure

import threading


global DEBUG_FLAG
global SHOW_IMAGE_FLAG
global USE_SERIAL_FLAG

DEBUG_FLAG=True
SHOW_IMAGE_FLAG=False
USE_SERIAL_FLAG=False


#用于定义数据发送类型的结构体
class MessageType(Structure):
    _fields_=[
        ("head", c_ubyte),
        ("function_word", c_ubyte),
        ("add_zero1", c_ubyte),
        ("add_zero2",c_ubyte),
        ("x", c_float),#vx或x
        ("y", c_float),#vx或x
        ("z", c_float),#vx或x
        ("sum_check", c_ubyte),
        ("end", c_ubyte),
    ]


#定义数据发送接收类
class MessageProcesser:
    def __init__(self):
        #1:初始化串口
        bps=460800
        port_list=list(serial.tools.list_ports.comports())
        port_list.sort(key=lambda x:x.device)#确保排列顺序为USB0,USB1
        
        #如果一个串口,则是使用模式
        if len(port_list)==1:
            self.USB0=serial.Serial(port=port_list[0].device,baudrate=bps,timeout=0.0001)#采用0.1ms的等待间隔

        #如果两个串口,则是测试串口模式
        elif len(port_list)==2:
            self.USB0=serial.Serial(port=port_list[0].device,baudrate=bps,timeout=0.0001)
            self.USB1=serial.Serial(port=port_list[1].device,baudrate=bps,timeout=0.0001)


        #2:初始化发送通信数据
        self.messageType=MessageType()
        self.messageType.head=c_ubyte(255)
        self.messageType.function_word=c_ubyte(1)
        self.messageType.add_zero1=c_ubyte(0)
        self.messageType.add_zero2=c_ubyte(0)
        self.messageType.x=c_float(0)
        self.messageType.y=c_float(0)
        self.messageType.z=c_float(0)
        self.messageType.sum_check=c_ubyte(0)
        self.messageType.end=c_ubyte(13)

        self.send_list=[self.messageType.head,self.messageType.function_word,self.messageType.add_zero1,self.messageType.add_zero2,self.messageType.x,self.messageType.y,self.messageType.z,self.messageType.sum_check,self.messageType.end]
        self.send_type='BBBBfffBB'
        self.send_msg=None

        #3:初始化信息接收
        self.read_msg_array=None
        #还需要不断地进行状态更新,之后更新状态
        self.robot_vx=0
        self.robot_vy=0
        self.robot_z=0

        #4:开始一个线程,用于接收串口信息,进而更新机器人状态
        Read_thread=threading.Thread(target=self.get_message)
        Read_thread.start()

    def get_send_msg(self,function_word=1,x=0,y=0,z=0,max_value=500):
        """
        用于串口发送信息,指定功能字与xyz三个信息,进而生成发送信息
        :param function_word:功能字
        :param x: 发送x/vx
        :param y: 发送y/vy
        :param z: 发送z/vz
        :param max_value: 速度中发送的最大速度
        :return:
        """
        #1:确保速度不要太大
        if function_word==1:
            x=min(x,max_value)
            y=min(y,max_value)
            z=min(z,max_value)


        #2:生成数据
        self.messageType.function_word=c_ubyte(function_word)
        self.messageType.x=c_float(x)
        self.messageType.y=c_float(y)
        self.messageType.z=c_float(z)
        self.messageType.sum_check=c_ubyte(0)#每一次进行一次和校验


        #3:生成发送数据
        self.send_list[1]=self.messageType.function_word
        self.send_list[4]=self.messageType.x
        self.send_list[5]=self.messageType.y
        self.send_list[6]=self.messageType.z
        self.send_list[-2]=self.messageType.sum_check
        no_sum_check=struct.pack(self.send_type,*self.send_list)

        #进行和校验
        sum=0
        for data in no_sum_check:
            sum=sum+data

        check_data=sum%256
        self.messageType.sum_check=c_ubyte(check_data)
        self.send_list[-2]=self.messageType.sum_check#和校验重新赋值

        #4:打包需要发送的数据
        assert len(self.send_list)==9,"生成的尺寸不足9个,不进行发送"
        self.send_msg=struct.pack(self.send_type,*self.send_list)
        return self.send_msg

    def get_message(self):
        """
        这里面的逻辑中,没有测试过读取信息较慢的情况下能否正常接收,之后串口出问题很大可能就是这个函数里面出问题
        get_message的函数不断地进行robot_vx,robot_vy,robot_z三个参数的更新,在串口类定义的时候就开启了这个线程
        长数据发送应该是没有大问题的
        :return:
        """
        #1:不断地更新读取的矩阵
        while True:
            #在这里面进行函数的执行
            data=self.USB0.read(100)

            #2:进行串口接送到的信息处理
            list_data=list(data)#data变成Byte
            if len(list_data)>=18:#只处理数据超过18个的情况
                #2.1:刚刚好是18个字长的状态
                if len(list_data)==18:
                    if list_data[0]==255 and list_data[-1]==13:#帧头帧尾确定
                        list_sum=sum(list_data)-list_data[-2]
                        if list_sum%256==list_data[-2]:#通过和校验
                            analysis_data=struct.unpack(self.send_type,data)#解析数据完成
                            if DEBUG_FLAG:
                                print("解析出来的数据为:",analysis_data)
                            self.robot_vx=analysis_data[4]
                            self.robot_vy=analysis_data[5]
                            self.robot_z=analysis_data[6]
                            continue#不在进行其他更多的解析

                #2.2:字长超过18个,需要开始进行解析
                else:
                    for begin_number in range(len(list_data)-18):
                        if list_data[begin_number]==255 and list_data[begin_number+17]==13:
                            list_sum=sum(list_data[begin_number:begin_number+18])-list_data[begin_number+16]
                            if list_sum%256==list_data[begin_number+16]:
                                analysis_data=struct.unpack(self.send_type,bytes(list_data[begin_number:begin_number+18]))
                                if DEBUG_FLAG:
                                    print("解析出来的数据为:",analysis_data)
                                self.robot_vx=analysis_data[4]
                                self.robot_vy=analysis_data[5]
                                self.robot_z=analysis_data[6]
                                break#不在进行其他更多的解析

                            else:
                                if DEBUG_FLAG:
                                    print("和校验出错")

            time.sleep(0.001)#进行每一轮的休息,避免接收太快出问题

    ############################整体功能性函数###################################
    @staticmethod
    def get_and_send():
        """
        这个是执行收发的操作,与工程车通信,返回收到的数据
        :return:
        """
        messageProcesser=MessageProcesser()
        x=500
        while True:
            x=x-1
            if x<0:
                x=500
            print("发送的数据为:x:{},vy:{},vz:{}".format(x,messageProcesser.robot_vy,messageProcesser.robot_z))
            send_msg=messageProcesser.get_send_msg(x=x,y=messageProcesser.robot_vy,z=messageProcesser.robot_z)#robot_vy和robot_z是接收的信息,x是自动更新的数据
            if DEBUG_FLAG:
                print("发送的数据为:",send_msg)
            messageProcesser.USB0.write(send_msg)#这里面不断地进行发送任务
            time.sleep(0.005)


if __name__ == '__main__':
    messageProcesser=MessageProcesser()
    x=500
    while True:
        x=x-1
        if x<0:
            x=500
        print("发送的数据为:x:{},vy:{},vz:{}".format(x,messageProcesser.robot_vy,messageProcesser.robot_z))
        send_msg=messageProcesser.get_send_msg(x=x,y=messageProcesser.robot_vy,z=messageProcesser.robot_z)
        # print("发送的数据为:",send_msg)
        messageProcesser.USB0.write(send_msg)#这里面不断地进行发送任务

        time.sleep(0.005)