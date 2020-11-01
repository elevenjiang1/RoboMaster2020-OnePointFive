"""
用于生成GUI图像,里面包含了EasyGui和GUI两个类
EasyGui负责基本GUI的显示
GUI负责整个GUI的绘制
这里面之后还需要补充一个不同命令的中断任务
"""
import tkinter as tk
from tkinter import filedialog
import os
import sys
# from BBoxGenerater import BBoxGenerater
from DataAugment import DataAugment
from DataProcesser import DataProcesser,GenerateDataset
from Annotation import Annotation
import tools
import threading
import cv2 as cv
class EasyGui:
    def __init__(self):
        """
        用于绘制GUI,简化tk调用的复杂度
        """
        self.father_windows = tk.Tk()
        self.font = ('Arial', 15)#字体设置
        self.writing_line=0#用于处理写的行数

    def setWindowName(self, name=''):
        """
        设置窗口名
        :param name: 窗口名称
        :return:
        """
        self.father_windows.title(name)

    def add_line(self):
        """
        用于增加一行
        :return:
        """
        self.writing_line=self.writing_line+1
        tk.Label(self.father_windows,text="").grid(row=self.writing_line)

    def new_line(self):
        """
        用于开新的一行
        :return:
        """
        self.writing_line=self.writing_line+1

    def setLabel(self, text='', width=1, row=None, column=0, columnspan=1):
        """
        放置一个按键对象
        :param text: Label输出内容
        :param width: Label宽度
        :param row: Label方式行,默认为None,这时为self.writing_line
        :param column: Label方式列
        :param columnspan: 合并多少个类
        :return:
        """
        if row is None:
            row =self.writing_line
        tk.Label(self.father_windows, text=text, width=width, font=self.font).grid(row=row, column=column,columnspan=columnspan )

    def setButton(self, text='', width=1, command=None, row=None, column=0, columnspan=1):
        """
        生成一个按键对象
        :param text:
        :param width: 按键长度
        :param command: 命令执行函数
        :param row: 放置的行数,默认为None,这时为self.writing_line
        :param column:放置列
        :param columnspan:合并整体
        :return:
        """
        if row is None:
            row =self.writing_line

        b = tk.Button(self.father_windows, text=text, width=width, command=command, font=self.font)
        b.grid(row=row, column=column, columnspan=columnspan)
        return b

    def setEntry(self, width=1, row=None, column=0, columnspan=1):
        """
        设置一个显示行
        :param width:
        :param row: 放置的行数,默认为None,这时为self.writing_line
        :param column:
        :param columnspan:
        :return:
        """
        if row is None:
            row =self.writing_line
        e = tk.Entry(self.father_windows, width=width, font=self.font)
        e.grid(row=row, column=column, columnspan=columnspan)
        return e

    def setText(self, width=1, height=1, row=None, column=0, columnspan=1):
        """
        多行文本输入
        :param width:
        :param height:
        :param row: 放置的行数,默认为None,这时为self.writing_line
        :param column:
        :param columnspan:
        :return:
        """
        if row is None:
            row =self.writing_line
        t = tk.Text(self.father_windows, width=width, height=height, font=self.font)
        t.grid(row=row, column=column, columnspan=columnspan)
        return t

    def setRadiobutton(self, text='', value='', var=None, row=None, column=0, columnspan=1):
        """
        设置选项按键
        :param text:
        :param value:
        :param var:
        :param row: 放置的行数,默认为None,这时为self.writing_line
        :param column:
        :param columnspan:
        :return:
        """
        if row is None:
            row =self.writing_line
        tk.Radiobutton(self.father_windows, text=text, variable=var, value=value, font=self.font).grid(row=row,
                                                                                                     column=column,
                                                                                                     columnspan=columnspan)

    def setCheckButton(self,text='',var=None,row=None,column=0,columnspan=1):
        """
        用于CheckButton选择,选中了为1,没选中为0
        :param text:
        :param var:
        :param row: 放置的行数,默认为None,这时为self.writing_line
        :param column:
        :param columnspan:
        :return:
        """
        if row is None:
            row =self.writing_line
        tk.Checkbutton(self.father_windows,text=text,variable=var,onvalue="True",offvalue="False",font=self.font).grid(row=row,column=column,columnspan=columnspan)

    def run(self):
        self.father_windows.mainloop()

class GUI(EasyGui):
    def __init__(self,Dataset_Name):
        """
        用于生成全局GUI,GUI只负责生成一个配置文件,之后的逻辑处理还是交给opencv进行处理
        """
        super().__init__()
        #1:参数初始化
        self.Dataset_Name=Dataset_Name
        self.mission_name=tk.StringVar()

        #2:文件路径初始化
        #主目录定义
        ROOT_DIR=os.path.dirname(os.path.realpath(__file__))#ROOT_DIR是lib的文件路径
        self.target_file=ROOT_DIR+"/../../"+str(self.Dataset_Name)#总文件夹指定
        self.target_file=os.path.abspath(self.target_file)

        #生成原数据集路径
        self.dataset_path=self.target_file+"/dataset"#用于之后图片路径生成
        self.images_path=self.dataset_path+"/images"
        self.annotations_path=self.dataset_path+"/image_annotations"

        #生成aug数据集路径
        self.aug_path=self.target_file+"/aug"
        self.aug_images_path=self.aug_path+"/images"
        self.aug_annotations_path=self.aug_path+"/image_annotations"

        #如果是一个新的项目,则生成所有需要文件
        if not os.path.exists(self.target_file):
            os.mkdir(self.target_file)
            #存放原图的位置
            os.mkdir(self.dataset_path)
            os.mkdir(self.images_path)
            os.mkdir(self.annotations_path)

            #存放数据增强的位置
            os.mkdir(self.aug_path)
            os.mkdir(self.aug_images_path)
            os.mkdir(self.aug_annotations_path)

        else:
            print("已经存在{}数据集,基于这个进行处理".format(self.Dataset_Name))

        #绘制GUI,初始化bbox生成器
        self.drawGUI()

        #功能类初始化
        self.dataAugment=DataAugment(dataset_path=self.dataset_path,aug_path=self.aug_path)
        self.dataProcesser=DataProcesser(dataset_path=self.dataset_path)
        self.generateDataset=GenerateDataset(self.aug_path)#生成数据集依赖于增强的数据结果
        self.annotation=Annotation(self.Dataset_Name)

    def beginMission(self,mission_name):
        self.mission_name.set(mission_name)
        self.Mission_Manager()

    def setMissionButton(self,text="",mission_name="",width=1,row=None,column=0,columnspan=1):
        """
        生成一个启动任务的按键
        :param text: 按键名称
        :param mission_name: 任务名称
        :param width: Button宽度
        :param row: Button所在行
        :param column: Button所在列
        :return:
        """

        if row is None:
            row=self.writing_line
        b = tk.Button(self.father_windows, text=text, width=width, command=lambda:self.beginMission(mission_name), font=self.font)
        b.grid(row=row, column=column, columnspan=columnspan)

        return b

    def drawGUI(self):
        """
        进行GUI的绘制工作
        :return:
        """
        #GUI名称
        self.setWindowName("数据标注软件2.0")
        self.writing_line=0

        #确定数据来源
        self.setLabel('数据源类型',width=10,column=0,columnspan=5)#选择数据来源
        self.dataType = tk.StringVar()
        self.setRadiobutton(text='已有数据集', value='image', var=self.dataType,column=6,columnspan=5)#两个圆圈选择,将image赋值给dataType
        self.setRadiobutton(text='视频',value='video',var=self.dataType,column=12,columnspan=2)
        self.setLabel('采样间隔',width=8,column=15,columnspan=4)
        self.samplingEntry=self.setEntry(width=2,column=20,columnspan=2)#采样间隔输入
        self.samplingEntry.insert('end', 10)#默认采样率为10

        self.new_line()
        self.setLabel('数据源路径',width=10,column=0,columnspan=5)#数据源路径输入
        self.dataPath=self.setEntry(width=20,column=6,columnspan=10)
        self.setMissionButton(text="选择路径",mission_name="select_file",width=8,column=17,columnspan=4)
        self.new_line()
        self.setMissionButton(text="生成图片",mission_name="begin_videotopictures",width=8,column=10,columnspan=4)
        self.add_line()

        #标注方式
        self.new_line()
        self.setLabel("标注方式",width=8,column=0,columnspan=4)
        self.setLabel("目标跟踪",width=8,column=5,columnspan=4)
        self.TrackCar=tk.StringVar()
        #目标跟踪只有 有和没有的区别,有的话统一采用KCF
        self.setCheckButton(text="使用跟踪",var=self.TrackCar,column=10,columnspan=5)

        self.new_line()
        #目标检测分为车子和装甲板
        self.DetectCar=tk.StringVar()
        self.DetectArmor=tk.StringVar()
        self.setLabel("目标检测",width=8,column=5,columnspan=4)
        self.setCheckButton(text="识别车子",var=self.DetectCar,column=10,columnspan=5)#是否识别车子
        self.setCheckButton(text="识别装甲板",var=self.DetectArmor,column=16,columnspan=5)#是否识别装甲板
        self.new_line()
        #是否展示标注信息
        self.flag_show_annotation=tk.StringVar()
        self.setRadiobutton(text="显示已标注",value="True",var=self.flag_show_annotation,column=10,columnspan=4)
        self.setRadiobutton(text="不显示",value="False",var=self.flag_show_annotation,column=15,columnspan=3)
        self.new_line()
        self.setMissionButton("开始标注",mission_name="begin_annotation",width=8,column=10,columnspan=4)
        self.add_line()


        #数据增强方式
        self.new_line()
        self.Augment_color=tk.StringVar()
        self.Augment_scale=tk.StringVar()
        self.setLabel("数据增强",width=8,column=0,columnspan=4)
        self.setCheckButton(text="亮暗操作",var=self.Augment_color,column=5,columnspan=5)
        self.setCheckButton(text="目标增多",var=self.Augment_scale,column=11,columnspan=5)

        self.new_line()
        self.setMissionButton(text="开始增强",mission_name="begin_dataaugment",width=8,column=10,columnspan=4)
        self.add_line()


        #不同数据格式生成
        self.new_line()
        self.setLabel("数据格式生成类型",width=16,column=0,columnspan=8)
        self.GenerateDatesetType=tk.StringVar()
        self.setRadiobutton(text="txt",value="txt",var=self.GenerateDatesetType,column=9,columnspan=3)#目标跟踪只有一个算法
        self.setRadiobutton(text="xml",value="xml",var=self.GenerateDatesetType,column=13,columnspan=3)
        self.setRadiobutton(text="json",value="json",var=self.GenerateDatesetType,column=17,columnspan=3)
        self.setRadiobutton(text="csv",value="csv",var=self.GenerateDatesetType,column=21,columnspan=3)
        self.new_line()
        self.setMissionButton(text="生成文件",mission_name="begin_makedata",width=8,column=10,columnspan=4)
        self.add_line()


        #数据清洗
        self.new_line()
        self.setLabel("数据清洗",width=16,column=0,columnspan=8)
        self.add_line()


        #消息框
        self.new_line()
        self.setLabel("信息输出框",width=10,column=8,columnspan=10)
        self.new_line()
        self.messageText=self.setText(width=44,height=10,column=0,columnspan=22)


        #窗口状态栏
        menubar=tk.Menu(self.father_windows)
        seeResultMenu=tk.Menu()
        seeResultMenu.add_command(label="查看原始数据",command=lambda:self.beginMission("begin_seeresult_o"))
        seeResultMenu.add_command(label="查看增强数据",command=lambda:self.beginMission("begin_seeresult_a"))

        menubar.add_cascade(label="查看标注数据",menu=seeResultMenu)
        self.father_windows.config(menu=menubar)

    def printText(self,info_output):
        """
        用于在GUI界面中进行信息输出
        :return:
        """
        self.messageText.insert('end',info_output+"\n")

    def Mission_Manager(self):
        """
        这个类用于进行多个任务管理,由这个产生任务,然后调用不同的类进行执行
        :return:
        """
        mission=self.mission_name.get()
        if mission=="select_file":
            self.beginGetDataPath()

        elif mission=="begin_videotopictures":
            Thread_videoToPictures=threading.Thread(target=self.beginToPicture)
            Thread_videoToPictures.run()

        elif mission=="begin_annotation":
            Thread_Annotation=threading.Thread(target=self.beginAnnotation)#,name="annotation")
            Thread_Annotation.run()

        elif mission=="begin_dataaugment":
            Thread_DataAugment=threading.Thread(target=self.beginDataAugment())#,name="dataaugment")
            Thread_DataAugment.run()

        elif mission=="begin_makedata":
            Thread_MakeData=threading.Thread(target=self.beginMakeData)
            Thread_MakeData.run()

        elif mission[:-2]=="begin_seeresult":
            Thread_SeeResult=threading.Thread(target=self.beginSeeResult)
            Thread_SeeResult.run()

        elif mission=="stop":
            print("这个功能等黄哥哥回来加")

        else:
            print("你这个按键还没有定义功能")

    def beginGetDataPath(self):
        """
        获取图片路径,需要提前确定数据源是image还是video
        :return:
        """
        self.printText("执行任务: 文件路径选择")
        self.dataPath.delete(0, 'end')#把上一次路径信息进行删除

        if self.dataType.get() == 'image':
            path = filedialog.askdirectory()#要求是文件夹

        elif self.dataType.get() == 'video':
            path = filedialog.askopenfilename()#要求是图片名称

        else:
            self.printText("  Warning:请先选择数据源类型")
            return
        self.dataPath.insert(0, path)#最后最开始插入选择的路径
        self.printText("!完成数据选择!")
        self.printText(" ")

    def beginToPicture(self):
        """
        用于基于视频生成图片的函数
        所有的图片生成在了/Dataset/images中
        包含了视频到Picture和文件路径到Picture
        :return:
        """
        #1:基于视频路径开始进行处理
        self.printText("执行任务: 视频转图片")
        data_path=self.dataPath.get()#获取数据源路径
        sample_rate=int(self.samplingEntry.get())#获取采样数目

        #2:当目标是文件夹的时,导入文件夹图片
        if os.path.isdir(data_path):

            new_images_path=data_path+"/images"
            new_annotations_path=data_path+"/image_annotations"
            assert os.path.exists(new_images_path),"图片路径不存在,不进行执行"
            assert os.path.exists(new_annotations_path),"标注路径不存在,不进行执行"
            self.printText("  送入的是文件夹,将其内部的文件进行导入")
            self.dataProcesser.File_To_Picture(new_dataset_path=data_path)

        #3:当目标是视频时,基于采样频率进行视频导入
        else:
            #3.1:确定导入路径
            images_path=self.dataset_path+"/images"
            annotations_path=self.dataset_path+"/image_annotations"

            #3.2:基于目标文件夹导入数据集
            if not os.path.exists(self.dataset_path):#不存在数据集
                self.printText("  没有已经存在的图片,自动生成在了{}".format(self.dataset_path))
                os.mkdir(self.dataset_path)
                os.mkdir(images_path)
                os.mkdir(annotations_path)
                begin_number=0

            else:#存在数据集
                self.printText("  已经存在文件,将会在后面继续生成")
                list_image=os.listdir(images_path)
                if len(list_image)==0:
                    begin_number=0
                else:
                    list_image.sort(key=lambda x:int(x[:-4]))
                    begin_number=int(list_image[-1][:-4])+1
            self.dataProcesser.Video_To_Picture(data_path,images_path,sample_rate,begin_number)

        self.printText("!完成视频转图片任务!")
        self.printText(" ")#完成一次生成图片,换行

    def beginAnnotation(self):
        """
        用于标注任务
        :return:
        """
        self.printText("执行任务: 开始标注")
        flag_trackcar=self.TrackCar.get()
        flag_detectcar=self.DetectCar.get()
        flag_detectarmor=self.DetectArmor.get()
        flag_show_annotation=self.flag_show_annotation.get()

        if not os.path.exists(self.dataset_path):
            self.printText(" 请创建路径文件夹")
            return

        if flag_show_annotation=="":
            self.printText('  请选择是否展示已标注的内容')
            return

        annotation_info={"dataset_path":self.dataset_path,"flag_trackcar":flag_trackcar,"flag_detectcar":flag_detectcar,"flag_detectarmor":flag_detectarmor,"flag_show_annotation":flag_show_annotation}

        self.annotation.Annotation_Mission(annotation_info)

        self.printText("完成了数据标注任务")
        self.printText(" ")
        cv.destroyAllWindows()

    def beginDataAugment(self):
        #1:先进行一次数据确保
        self.printText("执行任务: 开始数据增强")
        self.dataProcesser.checkData(dataset_path=self.dataset_path)

        #2:将数据移动到aug中
        self.dataAugment.move_origin_to_aug()

        #根据选项进行数据增强
        if self.Augment_color.get()=="True":
            self.dataAugment.aug_color_motion()
            self.printText("   完成了颜色增强")
        if self.Augment_scale.get()=="True":
            self.dataAugment.aug_more_bboxes()
            self.printText("   完成了bbox的增多")

        self.printText("!完成了数据集增强任务!")
        self.printText(" ")
        cv.destroyAllWindows()

    def beginMakeData(self):
        self.dataProcesser.checkData(dataset_path=self.aug_path)
        generateDatasettype=self.GenerateDatesetType.get()
        self.printText("执行任务: 开始生成{}数据".format(generateDatasettype))
        if generateDatasettype=="txt":
            print("需要补充txt数据格式生成")
        elif generateDatasettype=="json":
            self.generateDataset.getJsonFile()
        elif generateDatasettype=="xml":
            print("需要补充xml数据格式生成")
        elif generateDatasettype=="csv":
            self.generateDataset.getCsvFile()
        else:
            self.printText("请选择需要生成的数据格式")
            return

        self.printText("!完成数据集生成任务!")
        self.printText(" ")

    def beginSeeResult(self):
        self.printText("执行任务: 查看标注结果")
        mission=self.mission_name.get()
        if mission=="begin_seeresult_a":
            self.annotation.See_Annotation_Mission(self.aug_path)
        else:
            self.annotation.See_Annotation_Mission(self.dataset_path)


if __name__ == '__main__':
    # gui=GUI("SmallMap")
    # gui=GUI("HitRotate")
    gui=GUI("TEST1")
    gui.run()