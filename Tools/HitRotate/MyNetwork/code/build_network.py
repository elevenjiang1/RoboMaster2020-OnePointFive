"""
这里面用于生成网络模型
"""
import torch
import torch.nn as nn

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        #3x3卷积,之后跟着一个bn,最后来一个relu
        self.conv1 = conv3x3(inplanes, planes, stride)#第一层进行扩充
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        #第二层3x3的卷积核数不增加
        self.conv2 = conv3x3(planes, planes)#第二层两个相同
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x#一开始从x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)#补充上去的为x或者为降采样的x

        out += residual
        out = self.relu(out)

        return out


class My_Network(nn.Module):

    def __init__(self,block,layers,heads,head_conv):
        """
        block就是基本的模块,heads说明了网络输出的层数,head_conv为-1
        @param block:
        @param layers:
        @param heads:
        @param head_conv:
        """
        super(My_Network,self).__init__()#这句话之后要去看看到底是什么意思
        self.inplanes=64
        self.deconv_with_bias=False#默认不进行操作
        self.heads=heads

        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)#3->64卷积
        self.bn1=nn.BatchNorm2d(num_features=64,momentum=BN_MOMENTUM)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)#第一层是3->64的卷积,基本的stride和padding
        #    layers=[2,2,2,2]#layers不同的resnet不同

        #完成了卷积的生成
        self.layer1 = self._make_layer(block=block,planes=64,blocks=layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.deconv_layers=self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 256, 256],
            num_kernels=[4, 4, 4],
        )


        #这里给这个类生成了heads中的几个dict的名称
        for head in sorted(self.heads):
            num_output = self.heads[head]#每一个head生成一个对应的参数

            if head_conv > 0:#256->缓冲的head_conv->最终的卷积核
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv,kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(head_conv, num_output,kernel_size=1, stride=1, padding=0))#单个进行生成

            else:#小于0,则默认是这种卷积,这里是直接处目标
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)#相当于是给类中添加了一个head的操作,只是这里的送入的可以是string,这样子,类中多了一个self.head:fc,(这里的head是对应他下面的string)

    def _make_layer(self,block,planes,blocks,stride=1):
        """
        生成网络,两个block
        @param block:
        @param planes:
        @param blocks:
        @param stride:
        @return:
        """
        downsample=None


        #用于生成降采样模块
        if stride!=1 or self.inplanes!=planes*block.expansion:#Bottleneck.expansion=4,BasicBlock.expansion=1
            downsample=nn.Sequential(
                #这里是如果输入的网络层和planes*block.expansion不同,则进行一次这个操作
                nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=False),#把inplanes层变为planes*block.expansion层
                nn.BatchNorm2d(planes*block.expansion,momentum=BN_MOMENTUM),
            )

        #用于生成基本网络块
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))#看看是否包含了将采样操作(当对不齐的时候进行降采样)
        self.inplanes=planes*block.expansion#新的数据变成了planes*block.expansion层

        for i in range(1,blocks):#blocks决定了生成几次这个网络层,从1开始
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        #这里面好像index没有用
        #对应不同的上采样kernel,padding的尺度不同
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0


        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1


        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []

        # self.deconv_layers=self._make_deconv_layer(
        #     num_layers=3,
        #     num_filters=[256, 256, 256],
        #     num_kernels=[4, 4, 4],
        # )

        for i in range(num_layers):#进行3次上卷积
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)#这里面的index没啥用,直接送入

            planes = num_filters[i]#256层
            layers.append(
                nn.ConvTranspose2d(#反卷积
                    in_channels=self.inplanes,#上一次的卷积核数
                    out_channels=planes,#输出层数为256
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,#送入的padding
                    output_padding=output_padding,#输出的padding
                    bias=self.deconv_with_bias))

            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes


        return nn.Sequential(*layers)

    def foward(self,x):
        #先记性一次基本的卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        #再来过一个res18
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        #最后用一个上卷积,和目标一样的层
        x = self.deconv_layers(x)#进行了上卷积的操作


        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)#这里面应该就是不同的输出层


        return [ret]#最后是再对应到了输出层




#总体网络构建
# resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
#                34: (BasicBlock, [3, 4, 6, 3]),
#                50: (Bottleneck, [3, 4, 6, 3]),
#                101: (Bottleneck, [3, 4, 23, 3]),
#                152: (Bottleneck, [3, 8, 36, 3])}
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2])}

def get_network(num_layers,heads,head_conv):
    # block_class,layers=resnet_spec[num_layers]#指定多少层,确定网络模型

    block_class=BasicBlock
    layers=[2,2,2,2]
    heads={
        'hm':1,#识别的种类
        'wh':2,#长款的数量(还有一个中一个类别对应一个2维的wh
        'reg':2#细调的回归
    }

    """
     help='conv layer channels for output head'
      '0 for no conv layer'
      '-1 for default setting: '
      '64 for resnets and 256 for dla.')
    """

    head_conv=-1

    model=My_Network(block_class,layers,heads=heads,head_conv=head_conv)

    model.init_weights(num_layers,pretrained=False)

    return model



