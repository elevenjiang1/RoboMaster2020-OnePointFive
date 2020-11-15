# 好用工具

> 这里主要是记录RoboMaster中使用到的好用工具,主要包括
>
> - OpenCV使用GPU
> - Numpy的Cpp接口与Torch的Python接口联合编程

# 一 OpenCV使用GPU

> 项目开始是想要测试一下是否可能全部小主机换到TX2上面跑,要对比TX2与intel i5的速度差距
>
> 最终发现一张图片到GPU的显存上时间都需要5ms以上,这肯定是不能接收的,因此最终放弃此方案,但是进行OpenCV GPU的编程肯定是有必要的,在此进行记录

















# 二 Numpy-Cpp<->Torch-Python

> 项目开始是希望使用OpenVINO跑神经网络,希望在OpenVINO上面部署CenterNet,看看是否效果会更好.
>
> 但是CenterNet里面最后heatmap的操作在Cpp中实在是写不了,一张图片期望变成一个tensor很麻烦,基于此,就希望直接在Torch-python可以和Numpy-Cpp进行通信,完成这一套的任务













