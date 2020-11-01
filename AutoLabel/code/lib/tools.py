"""
工具类,各种文件处理
"""
import cv2 as cv
import os
import sys
import numpy as np
def show_text(show_text_info,point,image,text_size=0.8,color=(0,255,0)):
    """
    专门用于信息输出
    :param show_text_info: 需要绘制的信息
    :param point: 绘制点
    :param image: 图片
    :return:
    """
    x1,y1=point
    text_box_size=cv.getTextSize(show_text_info,cv.FONT_HERSHEY_DUPLEX,text_size,2)
    image[y1-text_box_size[0][1]:y1,x1:x1+text_box_size[0][0]]=color
    cv.putText(image,show_text_info,point,cv.FONT_HERSHEY_DUPLEX,text_size,(255,255,255),2,lineType=cv.LINE_AA)

def return_score(result):
    return result[4]

def videoToPicture(video_path, picture_path,sample_rate,begin_number=0,stop_number=0,type=".jpg"):
    """
    将视频变换成单张图片,存放图片的名称为(数字.jpg)
    :param video_path: 视频路径
    :param picture_path: 存放图片路径
    :param begin_number: 图片开始名称
    :param stop_number: 一共转换图片数
    :param type: 保存图片类型
    :return:
    """
    assert os.path.exists(video_path),video_path+" 路径下没有视频,不能进行视频帧的转换"
    assert os.path.exists(picture_path),"图片保存路径"+picture_path+"不存在,因此不能够进行文件保存"

    #读取image,进行转换
    cap = cv.VideoCapture(video_path)
    i = begin_number
    all_number=0#用于进行采样频率的跳过
    flag = False
    print("正在将视频转换为图片")
    while True:
        ret, frame = cap.read()
        if ret:
            all_number=all_number+1
            if all_number%sample_rate!=0:
                continue#不同采样率
            image_filename = picture_path + "/" + str(i) +type
            cv.imwrite(image_filename, frame)
            flag = True

        #避免视频读取完毕仍然在读取
        else:
            if flag:
                break
            else:
                print("该路径下没有视频")

        #用于单行输出
        sys.stdout.write("\r当前已经处理了 {} 帧视频".format(str(i-begin_number)))  # 为了进行单行输出
        sys.stdout.flush()

        #大于暂停数则停止
        if stop_number!=0:
            if stop_number+begin_number<i:#从开始图片算起
                break


        i = i + 1
    print(" ")
    print("完成视频解析,文件夹下共有{}帧图片,存放的路径为:{}".format(str(i), picture_path))
    print("这个视频一共有:{}帧".format(all_number))

def drawAnnotation(txt_path, image_show):
    """
    用于进行标注文件的显示(标注信息是绿色的)
    :param txt_path:标注文件名称
    :param image_show:进行展示图像使用
    :return: True:存在标注文件/False:不存在标注文件
    """
    #首先判断是否存在标注文件
    if os.path.exists(txt_path):
        temp_txt = open(txt_path, 'r')
        txt_info = temp_txt.readlines()

        # 这一块的数据变换有时间需要重新写一下
        for line in txt_info:
            ann=line.split(',')[:-1]#用于去除掉最后的\n
            ann_int=map(int,ann)
            number_ann=list(ann_int)
            category_id, x1, y1, x2, y2 = number_ann
            put_text_info=str(category_id)
            cv.rectangle(image_show, (x1, y1), (x2, y2), (0, 0, 255), 1)
            box_size=cv.getTextSize(put_text_info,cv.FONT_HERSHEY_SIMPLEX,0.5,2)
            image_show[y1-box_size[0][1]:y1+box_size[1],x1:x1+box_size[0][0]]=(0,0,255)
            cv.putText(image_show,put_text_info,(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

        return True

    else:
        return False

def draw_bbox(image,ann_info):
    """
    对bbox的绘制
    :param image:
    :param ann_info: 单条标注信息
    :return:
    """
    category_id,x1,y1,x2,y2=ann_info
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)


    cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)#绘制bbox
    name_dict={1:"car"}
    try:
        show_info="{}".format(name_dict[category_id])
    except:
        show_info="wrong"
    ((text_width,text_height),_)=cv.getTextSize(show_info,cv.FONT_HERSHEY_SIMPLEX,0.8,2)
    cv.rectangle(image,(x1,y1-text_height),(x1+text_width,y1),(0,255,0),-1)
    cv.putText(image,show_info,(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

def visualize(anno_infos,image,window_name="visualize_image",show_image=True):
    """
    基于标注信息和图片进行绘制
    :param anno_infos: 标注信息
    :param image: 图片
    :return:
    """
    image_show=image.copy()
    for ann in anno_infos:
        draw_bbox(image_show,ann)

    if show_image:
        cv.imshow(window_name,image_show)

def draw_bbox2(image,ann_info):
    """
    对bbox的绘制
    :param image:
    :param ann_info: 单条标注信息
    :return:
    """
    x1,y1,x2,y2,score,category_id=ann_info
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)


    cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)#绘制bbox
    name_dict={1:"car"}
    try:
        show_info="{}".format(name_dict[category_id])
    except:
        show_info="wrong"
    ((text_width,text_height),_)=cv.getTextSize(show_info,cv.FONT_HERSHEY_SIMPLEX,0.8,2)
    cv.rectangle(image,(x1,y1-text_height),(x1+text_width,y1),(0,255,0),-1)
    cv.putText(image,show_info,(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

def visualize_fordetect_car(anno_infos,image,window_name="visualize_image",show_image=True):
    image_show=image.copy()
    for ann in anno_infos:
        draw_bbox2(image_show,ann)

    if show_image:
        cv.namedWindow(window_name,cv.WINDOW_NORMAL)
        cv.imshow(window_name,image_show)
        cv.waitKey(0)

    return image_show

def check_path(images_path,annotations_path):
    if not os.path.exists(images_path):
        print("不存在{}的图片路径,请进行确定")
        assert False
    if not os.path.exists(annotations_path):
        print("不存在{}的标注路径,请进行确定")
        assert False

def checkInBox(armor_bbox,car_bbox):
    """
    用于判断armor_bbox是否在car_bbox中,此处的是xmin,ymin,xmax,ymax的操作
    :param armor_bbox: 装甲板的bbox
    :param car_bbox: 车子的bbox
    :return: 在车子的bbox中,返回True,否则返回False
    """

    xmin_a=armor_bbox[0]
    ymin_a = armor_bbox[1]
    xmax_a=armor_bbox[2]
    ymax_a=armor_bbox[3]

    xmin_c = car_bbox[0]
    ymin_c = car_bbox[1]
    xmax_c = car_bbox[2]
    ymax_c = car_bbox[3]

    if xmin_c<xmin_a and xmax_c>xmax_a and ymin_c<ymin_a  and ymax_c>ymax_a:
        return True
    else:
        return False

def show_track_result(trackbbox,image):
    x1,y1,x2,y2,score,category_id=trackbbox
    cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
    show_text_info="track"
    #这个的需要把信息放在左下角,因此自己写xyz
    text_box_size=cv.getTextSize(show_text_info,cv.FONT_HERSHEY_DUPLEX,0.8,2)
    image[y2:y2+text_box_size[0][1]+text_box_size[1],x1:x1+text_box_size[0][0]]=(0,255,0)
    cv.putText(image,show_text_info,(x1,y2+text_box_size[0][1]),cv.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)

def show_network_result(networkbboxes,image):
    sorted(networkbboxes,key=return_score)#进行从高到底的排序
    for i,networkbbox in enumerate(networkbboxes):
        x1,y1,x2,y2,score,category_id=networkbbox
        cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
        show_text_info="{}  {:.2f}".format(i,score)
        show_text(show_text_info,(x1,y1),image)
        if i>8:#一张图只进行置信度最高的9个显示
            break

def generate_kernel(x,y):
    return np.ones((x,y),dtype=np.uint8)

def filter_results(results,confidence=0.8):
    """
    对于识别结果进行过滤
    :param results: 识别结果
    :param confidence: 置信度
    :return:
    """
    return_results=[]
    for result in results:
        x1,y1,x2,y2,score,category_id=result
        if score>confidence:
            return_results.append([x1,y1,x2,y2,score,category_id])

    return return_results



