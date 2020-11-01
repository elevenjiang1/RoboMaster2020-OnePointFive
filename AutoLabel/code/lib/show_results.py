import cv2 as cv
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

def show_track_result(trackbboxes,image):
    for trackbbox in trackbboxes:
        x1,y1,x2,y2,score,category_id=trackbbox
        cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
        show_text_info="track"
        #这个的需要把信息放在左下角,因此自己写xyz
        text_box_size=cv.getTextSize(show_text_info,cv.FONT_HERSHEY_DUPLEX,0.8,2)
        image[y2:y2+text_box_size[0][1]+text_box_size[1],x1:x1+text_box_size[0][0]]=(0,255,0)
        cv.putText(image,show_text_info,(x1,y2+text_box_size[0][1]),cv.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)




def show_network_result(networkbboxes,image):
    if len(networkbboxes)==0:
        return

    sorted(networkbboxes,key=return_score)#进行从高到底进行分数排序
    for i,networkbbox in enumerate(networkbboxes):
        x1,y1,x2,y2,score,category_id=networkbbox
        cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
        show_text_info="{}  {:.2f}".format(i,score)
        show_text(show_text_info,(x1,y1),image)
        if i>8:#一张图只进行置信度最高的9个显示
            break