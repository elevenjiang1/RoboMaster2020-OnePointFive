import json
import os
import cv2 as cv
import csv
from tqdm import tqdm
import random
import sys
import shutil
import tools
import pyzed.sl as sl
import numpy as np
"""
使用说明:
    这个类用于确定数据无误,同时生成一个迭代器用于进行图片和标注的返回
    
if __name__ == '__main__':
    checkData=CheckData(dataset_path="/home/elevenjiang/Documents/Project/RM/Code/SmallMap/MakeData/Code/AutoLabel2/target_file/dataset")#送入dataset路径和是否进行数据检测
    all_data=checkData.getAllData(shuffle=False)#获取所有数据迭代器,且不进行打乱

    #进行结果读取
    for info in all_data:
        image=info["image"]#获取图片
        ann_infos=info["ann_infos"]#获取标注信息,为一个list,每个元素为(category_id,x1,y1,x2,y2)
        tools.visualize(ann_infos,image)
        cv.waitKey(0)
        
"""

class DataProcesser:
    def __init__(self, dataset_path=None):
        # dataset_path="/home/elevenjiang/Documents/Project/RM/Code/SmallMap/MakeData/Code/AutoLabel2/target_file/dataset"
        assert dataset_path is not None, "需要初始化DataProcesser的路径"
        self.dataset_path = dataset_path
        self.images_path = self.dataset_path + "/images"
        self.annotations_path = self.dataset_path + "/image_annotations"

    def checkBBox(self, image_size, bbox):
        """
        送入图片的长宽和bbox的点,判断bbox是否为合格的bbox
        :param image_size:
        :param bbox:
        :return:
        """
        image_w, image_h = image_size

        x1, y1, x2, y2 = bbox
        if not 0 < x1 < image_w:
            return False
        if not x1 < x2 < image_w:
            return False
        if not 0 < y1 < image_h:
            return False
        if not y1 < y2 < image_h:
            return False

        return True

    def checkData(self,dataset_path=None):
        """
        用于确定txt和image之间是否对应良好
        :return:
        """
        delete_name = []
        if dataset_path is None:
            dataset_path=self.dataset_path

        images_path=dataset_path+"/images"
        annotations_path=dataset_path+"/image_annotations"
        assert os.path.exists(images_path), "图片路径不存在"
        assert os.path.exists(annotations_path), "标注路径不存在"

        # 1:先删除image存在而txt不存在的文件
        print("开始检查 {} 的数据集".format(dataset_path))
        for image_name in tqdm(os.listdir(images_path)):
            filename = image_name[:-4]
            image_path = os.path.join(images_path, image_name)
            txt_path = os.path.join(annotations_path, filename + ".txt")

            if not os.path.exists(txt_path):
                delete_name.append(image_path)
                os.remove(image_path)  # 进行image删除
                continue

            #1.1:进行txt内部参数的确定
            image = cv.imread(image_path)
            image_h, image_w, image_c = image.shape
            txt_file = open(txt_path)
            txt_info = txt_file.readlines()
            txt_file.close()

            for line in txt_info:
                ann = line.split(',')[:-1]
                ann_int = map(int, ann)
                number_ann = list(ann_int)
                category_id, x1, y1, x2, y2 = number_ann
                check_bbox_flag = self.checkBBox((image_w, image_h), (x1, y1, x2, y2))
                if not check_bbox_flag:
                    print("删除的bbox为::", x1, y1, x2, y2)
                    delete_name.append(txt_path + image_path)
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    continue

        # 2:避免txt文件有的而image没有的情况
        image_list = os.listdir(images_path)
        txt_list = os.listdir(annotations_path)

        #当txt有多的时候进行image删除,否则就是一一对应的关系了
        if len(image_list) < len(txt_list):
            print("存在有txt而没有对应图片的情况,开始检查图片情况")
            for txt_name in tqdm(os.listdir(annotations_path)):
                filename = txt_name[:-4]
                image_path = os.path.join(images_path, filename + ".jpg")
                txt_path = os.path.join(annotations_path, txt_name)

                if not os.path.exists(image_path):
                    delete_name.append(txt_path)
                    os.remove(txt_path)

        # 3:最后进行删除文件的显示
        if len(delete_name) == 0:
            print("数据集无误,可以放心使用")
        else:
            print("删除了{}个文件,他们分别为:".format(len(delete_name)))
            for data in delete_name:
                print("         " + data)
            print("剩下的文件可以放心使用")

    def getAllData(self,dataset_path=None,shuffle=False):
        """
        生成数据读取方式,返回一个迭代器,使用方法见文件开始
        :param shuffle:
        :return:
        """
        class ALL_DATA:
            def __init__(self, dataset_path, shuffle=shuffle):
                self.dataset_path = dataset_path
                self.images_path = self.dataset_path + "/images"
                self.image_annotations_path = self.dataset_path + "/image_annotations"
                self.image_list = os.listdir(self.images_path)

                if not shuffle:  # 不进行打乱
                    self.image_list.sort(key=lambda x: int(x[:-4]))

                self.id = -1

            def __iter__(self):
                return self

            def __len__(self):
                return len(self.image_list)

            def __next__(self):
                if self.id < len(self.image_list) - 1:
                    self.id = self.id + 1
                    image_name = self.image_list[self.id]
                    filename = image_name[:-4]
                    image_path = os.path.join(self.images_path, image_name)
                    txt_path = os.path.join(self.image_annotations_path, filename + ".txt")
                    image = cv.imread(image_path)
                    txt_file = open(txt_path)
                    txt_info = txt_file.readlines()
                    ann_infos = []
                    for line in txt_info:
                        ann = line.split(',')[:-1]
                        ann_int = map(int, ann)
                        number_ann = list(ann_int)
                        category_id, x1, y1, x2, y2 = number_ann

                        ann_infos.append((category_id, x1, y1, x2, y2))

                    return {"image": image, "ann_infos": ann_infos}

                else:
                    raise StopIteration
        if dataset_path is None:
            dataset_path=self.dataset_path

        self.all_data = ALL_DATA(dataset_path)
        self.all_data = iter(self.all_data)
        return self.all_data

    def clean_alldata(self,dataset_path=None):
        """
        用于对数据中进行清洗
        生成在他的同路径下面的temp文件夹中
        @return:
        """

        #创建清洗后的文件夹
        save_path=dataset_path+"_cleaned"
        if os.path.exists(save_path):
            assert False,"已经存在了清洗过的文件,请进行手动文件修改"
        else:
            os.mkdir(save_path)
            save_path_images=save_path+"/images"
            save_path_annotations=save_path+"/image_annotations"
            os.mkdir(save_path_images)
            os.mkdir(save_path_annotations)

        if dataset_path is None:
            dataset_path=self.dataset_path


        #确保本身文件夹正确
        self.checkData(dataset_path)#进行一次数据清洗

        images_path=dataset_path+"/images"
        annotations_path=dataset_path+"/image_annotations"


        print("开始生成在新的文件夹中")


        #直接采用复制的方式转移文件
        images_list=os.listdir(images_path)
        images_list.sort(key=lambda x:int(x[:-4]))

        for all_i,image_name in tqdm(enumerate(images_list)):
            filename=image_name[:-4]
            txt_name=filename+".txt"

            #复制图片
            save_name_image=str(all_i)+".jpg"


            image_path=os.path.join(images_path,image_name)
            save_path_image=os.path.join(save_path_images,save_name_image)
            shutil.copyfile(image_path,save_path_image)



            save_name_txt=str(all_i)+".txt"
            txt_path=os.path.join(annotations_path,txt_name)
            save_txt_path=os.path.join(save_path_annotations,save_name_txt)
            shutil.copyfile(txt_path,save_txt_path)

        print("完成数据清洗")

    def Video_To_Picture(self,video_path, images_path,sample_rate,begin_number=0,type=".jpg"):
        """
        将视频变换成单张图片,存放图片的名称为(数字.jpg)
        如果是.svo文件,需要进行一下配置
        :param video_path: 视频路径
        :param images_path: 存放图片路径
        :param begin_number: 图片开始名称
        :param stop_number: 一共转换图片数
        :param type: 保存图片类型
        :return:
        """
        #1:确保有视频文件
        assert os.path.exists(video_path),video_path+" 路径下没有视频,不能进行视频帧的转换"
        assert os.path.exists(images_path),"图片保存路径"+images_path+"不存在,因此不能够进行文件保存"

        #2:初始化导入参数
        i=begin_number#用于计算转换视频
        all_number=0#用于跳过采样频率
        read_flag=False#用于确定是导入的问题还是读完的问题

        #3:导入视频
        #3.1:处理.svo文件
        if video_path[-4:]==".svo":
            input_type = sl.InputType()
            input_type.set_from_svo_file(video_path)
            init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
            cam = sl.Camera()
            status = cam.open(init)
            if status!=sl.ERROR_CODE.SUCCESS:
                print(".svo文件导入失败,错误原因为:",repr(status))
                return
            runtime=sl.RuntimeParameters()
            sl_image=sl.Mat()

            print("正在将.svo文件转换为图片")
            while True:
                err=cam.grab(runtime)
                if err==sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(sl_image)
                    image=sl_image.get_data()[:,:,:3].copy()
                    image=image.astype(np.uint8)

                    all_number=all_number+1
                    if all_number%sample_rate!=0:
                        continue
                    image_filename=images_path+"/"+str(i)+type
                    cv.imwrite(image_filename,image)
                    read_flag=True

                else:
                    if read_flag:
                        break
                    else:
                        print("导入.svo文件出错,错误码为:",err)


                sys.stdout.write("\r当前已经处理了 {} 帧视频".format(str(i-begin_number)))  # 为了进行单行输出
                sys.stdout.flush()
                i = i + 1
            print(" ")
            print("完成视频解析,文件夹下共有{}帧图片,存放的路径为:{}".format(str(i), images_path))
            print("这个视频一共有:{}帧".format(all_number))


        #3.2:处理其他视频文件
        else:
            cap = cv.VideoCapture(video_path)

            print("正在将视频转换为图片")
            while True:
                ret, frame = cap.read()
                if ret:
                    all_number=all_number+1
                    if all_number%sample_rate!=0:
                        continue#不同采样率
                    image_filename = images_path + "/" + str(i) +type
                    cv.imwrite(image_filename, frame)
                    read_flag = True

                #避免视频读取完毕仍然在读取,同时确保是视频
                else:
                    if read_flag:
                        break
                    else:
                        print("该文件不是视频")

                #用于单行输出
                sys.stdout.write("\r当前已经处理了 {} 帧视频".format(str(i-begin_number)))  # 为了进行单行输出
                sys.stdout.flush()
                i = i + 1
            print(" ")
            print("完成视频解析,文件夹下共有{}帧图片,存放的路径为:{}".format(str(i), images_path))
            print("这个视频一共有:{}帧".format(all_number))

    def File_To_Picture(self,new_dataset_path,target_dataset_path=None):
        """
        把一个新文件夹路径移动到dataset中
        这里面并不支持DJI的数据集的文件名称的导入
        @param new_dataset_path:新的数据集路径
        @param target_dataset_path: 移动的目标数据集.如果为None,则为目标的文件
        @return:
        """
        #1:确定移动的目标文件夹
        if target_dataset_path is None:
            target_dataset_path=self.dataset_path
        else:
            assert os.path.exists(target_dataset_path),"移动的目标路径不存在"
        print("送入的是{}的路径,将这个文件夹下的images和image_annotations移动到{}的路径".format(new_dataset_path,target_dataset_path))
        #移动的目标文件
        images_path=target_dataset_path+"/images"
        annotations_path=target_dataset_path+"/image_annotations"

        #2:配置新进来的文件夹
        new_images_path=new_dataset_path+"/images"
        new_annotations_path=new_dataset_path+"/image_annotations"
        #对新进来的文件夹做一次checkData操作
        assert os.path.exists(new_images_path),"图片路径不存在,不进行执行"
        assert os.path.exists(new_annotations_path),"标注路径不存在,不进行执行"
        self.checkData(dataset_path=new_dataset_path)

        #3:获取保存的开始文件数字
        #3.1:没有目标文件存在,则直接创建目标文件夹
        if not os.path.exists(target_dataset_path):
            print("  没有已经存在的图片,自动生成在了{}".format(target_dataset_path))
            os.mkdir(target_dataset_path)
            os.mkdir(images_path)
            os.mkdir(annotations_path)
            begin_number=0

        #3.2:存在目标文件,查找最后文件的数字(这个不适用于DJI的数据集)
        else:
            print("  已经存在文件,将会在后面继续生成")
            images_list=os.listdir(images_path)
            txts_list=os.listdir(annotations_path)
            if len(images_list)==0:
                begin_number=0

            else:
                images_list.sort(key=lambda x:int(x[:-4]))
                begin_number_image=int(images_list[-1][:-4])+1
                txts_list.sort(key=lambda  x:int(x[:-4]))
                begin_number_txt=int(txts_list[-1][:-4])+1

                if begin_number_image==begin_number_txt:#末尾开始数字相同时
                    begin_number=begin_number_image

                else:#如果末尾数字不同,则进行目标文件的数据删除
                    self.checkData(target_dataset_path)
                    images_list=os.listdir(images_path)#保护image
                    images_list.sort(key=lambda x:int(x[:-4]))
                    begin_number=int(images_list[-1][:-4])+1

        #4:移动目标文件
        print("开始转移数据集")
        images_list=os.listdir(new_images_path)
        images_list.sort(key=lambda x:int(x[:-4]))
        for image_name in tqdm(images_list):
            file_name=image_name[:-4]
            txt_name=file_name+".txt"
            #复制image
            save_name_image=str(begin_number)+".jpg"
            new_image_path=os.path.join(new_images_path,image_name)
            save_path_image=os.path.join(images_path,save_name_image)
            shutil.copyfile(new_image_path,save_path_image)

            #复制txt
            save_name_txt=str(begin_number)+".txt"
            new_txt_path=os.path.join(new_annotations_path,txt_name)
            save_txt_path=os.path.join(annotations_path,save_name_txt)
            shutil.copyfile(new_txt_path,save_txt_path)

            begin_number=begin_number+1

        print("完成数据添加工作")

class GenerateDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataProcesser = DataProcesser(self.dataset_path)  # 不使用数据检查

    def getJsonFile(self, train=0.7, val=0.25, test=0.04):
        """
        在dataset目录下生成一个JsonFile的文件,下面包含train2017.json,val2017.json,test2017.json三个json文件
        另外,有instances_train2017,instances_val2017,instances_test2017,分别存放了对应的图片
        使用样例:
        generate_dataset = Generate_Dataset(dataset_path="/home/elevenjiang/Documents/Project/RM/Code/SmallMap/MakeData/Code/AutoLabel2/target_file/dataset")#初始化路径
        generate_dataset.getJsonFile()#生成Json文件
        :param train:训练占比
        :param val: 验证占比
        :param test: 测试占比
        :return:
        """
        #1:Json需要的一系列参数初始化
        #确定比例类型
        assert train + val + test <= 1.0, "请输入正确占比"
        val = train + val
        if not os.path.exists(self.dataset_path):
            print("生成Json文件需要{}文件夹存在")
            assert False,"生成Json文件失败"


        #三种训练集生成
        cats = ['car']  # 默认只有一个类
        cat_info = []
        for i, cat in enumerate(cats):
            cat_info.append({'name': cat, 'id': i + 1})
        train_ret = {"images": [], "annotations": [], "categories": cat_info}
        val_ret = {"images": [], "annotations": [], "categories": cat_info}
        test_ret = {"images": [], "annotations": [], "categories": cat_info}

        #2:生成文件夹
        JsonFile_Path = os.path.join(self.dataset_path, "../JsonFile")#上一个路径进行文件保存
        JsonFile_Path=os.path.abspath(JsonFile_Path)
        if os.path.exists(JsonFile_Path):
            print("已经有了Json文件,请将 {} 移走,从而生成新的数据集".format(JsonFile_Path))#为了避免覆盖掉已有的数据
            return
        else:
            try:
                os.mkdir(JsonFile_Path)
                print("生成了Json文件,在{}".format(JsonFile_Path))
            except:
                print("生成文件失败,{}中间欠缺文件".format(JsonFile_Path))
                return

        #2.1之后生成3个文件夹存放图片
        train_path = os.path.join(JsonFile_Path, "train2017")
        val_path = os.path.join(JsonFile_Path, "val2017")
        test_path = os.path.join(JsonFile_Path, "test2017")
        annotations_path=os.path.join(JsonFile_Path,"annotations")
        os.mkdir(train_path)
        os.mkdir(val_path)
        os.mkdir(test_path)
        os.mkdir(annotations_path)

        #3:生成json文件
        image_id = 0
        all_data=self.dataProcesser.getAllData(shuffle=True)

        print("开始生成Json数据集")
        for all_i, data in tqdm(enumerate(all_data)):
            # 开始生成文件
            image = data["image"]
            ann_infos = data["ann_infos"]
            save_image_name = str(all_i) + ".jpg"

            if 0 <= all_i / len(all_data) < train:
                save_image_path = os.path.join(train_path, save_image_name)
                # 1:进行图片保存
                cv.imwrite(save_image_path, image)

                # 2:进行ret文件写入
                ret_image = {'file_name': save_image_name, 'id': image_id}
                train_ret['images'].append(ret_image)
                for ann in ann_infos:
                    category_id, x1, y1, x2, y2 = ann
                    ret_ann = {'image_id': image_id, 'id': len(train_ret['annotations']) + 1,
                               'category_id': category_id, 'bbox': [x1, y1, x2 - x1, y2 - y1]}
                    train_ret['annotations'].append(ret_ann)
                image_id = image_id + 1

            elif train <= all_i / len(all_data) < val:
                save_image_path = os.path.join(val_path, save_image_name)
                cv.imwrite(save_image_path, image)
                ret_image = {'file_name': save_image_name, 'id': image_id}
                val_ret['images'].append(ret_image)

                for ann in ann_infos:
                    category_id, x1, y1, x2, y2 = ann
                    ret_ann = {'image_id': image_id, 'id': len(train_ret['annotations']) + 1,
                               'category_id': category_id, 'bbox': [x1, y1, x2 - x1, y2 - y1]}
                    val_ret['annotations'].append(ret_ann)
                image_id = image_id + 1

            elif val <= all_i / len(all_data) <= 1:
                save_image_path = os.path.join(test_path, save_image_name)
                cv.imwrite(save_image_path, image)
                ret_image = {'file_name': save_image_name, 'id': image_id}
                test_ret['images'].append(ret_image)

                for ann in ann_infos:
                    category_id, x1, y1, x2, y2 = ann
                    ret_ann = {'image_id': image_id, 'id': len(train_ret['annotations']) + 1,
                               'category_id': category_id, 'bbox': [x1, y1, x2 - x1, y2 - y1]}
                    test_ret['annotations'].append(ret_ann)
                image_id = image_id + 1

            else:
                print(all_i / len(all_data))
                print("数据集出问题了")
                return

        json_train_file_name = "instances_train2017.json"
        json_val_file_name = "instances_val2017.json"
        json_test_file_name = "instances_test2017.json"

        json_train_file_path = os.path.join(annotations_path, json_train_file_name)
        json_val_file_path = os.path.join(annotations_path, json_val_file_name)
        json_test_file_path = os.path.join(annotations_path, json_test_file_name)

        json.dump(train_ret, open(json_train_file_path, 'w'))
        json.dump(val_ret, open(json_val_file_path, 'w'))
        json.dump(test_ret, open(json_test_file_path, 'w'))

    def getCsvFile(self,trainRate=0.7,loc=True):
        """
        用于生成Csv训练数据
        :param trainRate:
        :return:
        """
        #1:生成对应文件夹
        CsvFile_Path=os.path.join(self.dataset_path,"../CsvFile")
        CsvFile_Path=os.path.abspath(CsvFile_Path)
        images_Path=os.path.join(CsvFile_Path,"images")
        if os.path.exists(CsvFile_Path):
            print("已经存在了CsvFile,请将 {} 移走,从而生成新的数据集".format(CsvFile_Path))
            return
        else:
            try:
                os.mkdir(CsvFile_Path)
                os.mkdir(images_Path)
            except:
                print("生成数据集失败")

        if not os.path.exists(self.dataset_path):
            print("生成Json文件需要{}文件夹存在")
            assert False,"生成Json文件失败"


        #2:生成csv文件
        name_dict = {1: "car"}
        title = 'filename','width','height','class','xmin','ymin','xmax','ymax'#设置表头
        trainWriter = open(os.path.join(CsvFile_Path, 'train.csv'), 'w', newline='', encoding='utf-8')
        trainCsvWriter = csv.writer(trainWriter)
        testWriter = open(os.path.join(CsvFile_Path, 'test.csv'), 'w', newline='', encoding='utf-8')
        testCsvWriter = csv.writer(testWriter)
        trainCsvWriter.writerow(title)
        testCsvWriter.writerow(title)

        all_data=self.dataProcesser.getAllData(shuffle=False)
        num=0
        dividing = int(trainRate * 100)#用于随机读取数据
        print("开始生成Csv数据文件夹")
        for info in tqdm(all_data):
            image = info['image']
            ann_infos = info['ann_infos']
            filename = str(num) + '.jpg'
            num += 1
            cv.imwrite(os.path.join(images_Path, filename), image)
            H = image.shape[0]
            W = image.shape[1]
            for ann in ann_infos:
                category_id, x1, y1, x2, y2 = ann
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                objectName = name_dict[category_id]
                if not loc:
                    x1, y1, x2, y2 = x1 / W, y1 / H, x2 / W, y2 / H
                row = [filename, W, H, objectName, x1, y1, x2, y2]
                if random.randint(1, 100) < dividing:
                    trainCsvWriter.writerow(row)
                else:
                    testCsvWriter.writerow(row)


        trainWriter.close()
        testWriter.close()

    def getArmorFile(self,trainRate=0.7):
        """
        用于生成适合YOLO训练的数据集,需要把图片生成到新的位置里面
        需要生成归一化的YOLO训练集,还需要生成train.txt,valid.txt,这里面要包含image的路径
        @param trainRate: 生成的训练比例
        @return:
        """
        assert trainRate<1,"请输入正确的trianRate"
        self.dataProcesser.checkData(dataset_path=self.dataset_path)

        #1:生成Armor路径
        Armor_Path=os.path.join(self.dataset_path,"../Armor")
        Armor_Path=os.path.abspath(Armor_Path)

        if os.path.exists(Armor_Path):
            print("已经存在了Armor的文件,请将{}移走,从而生成新的数据".format(Armor_Path))
            return
        else:
            try:
                os.mkdir(Armor_Path)
                print("生成了Armor文件夹,在{}".format(Armor_Path))

            except:
                print("生成Armor文件夹失败,{}中间欠缺文件".format(Armor_Path))


        #2:生成保存文件夹
        #生成images和labels两个文件夹
        images_path=os.path.join(Armor_Path,"images")
        annotations_path=os.path.join(Armor_Path,"labels")
        os.mkdir(images_path)
        os.mkdir(annotations_path)


        #3:开始生成数据
        #直接生成images和labels,最后的train.txt和valid.txt再进行规整
        all_data=self.dataProcesser.getAllData(shuffle=True)

        save_number=0
        for all_i,data in tqdm(enumerate(all_data)):
            #每一张图片进行处理
            image=data['image']
            anno_infos=data["ann_infos"]
            save_image_name=str(save_number)+".jpg"
            save_annotation_name=str(save_number)+".txt"

            car_annotations=[]
            armor_annotations=[]
            #3.1:解析标注数据
            for ann in anno_infos:
                category_id,x1,y1,x2,y2=ann

                if category_id==1:
                    car_annotations.append({'bbox':[x1,y1,x2,y2]})

                else:
                    armor_annotations.append({'category_id':category_id,'bbox':[x1,y1,x2,y2]})

            #3.2:基于每一张车图进行处理
            for i in range(len(car_annotations)):
                save_image_path=os.path.join(images_path,save_image_name)
                save_txt_path=os.path.join(annotations_path,save_annotation_name)
                save_infos=[]

                for j in range(len(armor_annotations)):
                    #保存在这个车bbox里的装甲板
                    car_bbox=car_annotations[i]['bbox']
                    car_x1,car_y1,car_x2,car_y2=car_bbox
                    armor_bbox=armor_annotations[j]['bbox']
                    id=armor_annotations[j]['category_id']

                    result=tools.checkInBox(armor_bbox,car_bbox)

                    if result:
                        absolute_armor_x1,absolute_armor_y1,absolute_armor_x2,absolute_armor_y2=armor_bbox#整张图绝对的像素

                        #基于carimage生成图片,而不是相对于整张图的点
                        #保存对应的装甲板信息
                        armor_x1=absolute_armor_x1-car_x1
                        armor_y1=absolute_armor_y1-car_y1
                        armor_x2=absolute_armor_x2-car_x1
                        armor_y2=absolute_armor_y2-car_y1


                        #保存对应的图片
                        save_center_x=(armor_x1+armor_x2)/2
                        save_center_y=(armor_y1+armor_y2)/2
                        w=armor_x2-armor_x1
                        h=armor_y2-armor_y1

                        #再按照YOLO的要求进行归一化
                        save_center_x=save_center_x/(car_x2-car_x1)
                        save_center_y=save_center_y/(car_y2-car_y1)
                        w=w/(car_x2-car_x1)
                        h=h/(car_y2-car_y1)

                        save_info="{} {} {} {} {}\n".format(str(id),str(save_center_x),str(save_center_y),str(w),str(h))
                        save_infos.append(save_info)

                if len(save_infos)>0:
                    txt_file=open(save_txt_path,'w')
                    for save_info in save_infos:
                        txt_file.write(save_info)#保存txt信息
                    txt_file.close()

                    car_image=image[car_y1:car_y2,car_x1:car_x2]
                    cv.imwrite(save_image_path,car_image)
                    save_number=save_number+1#保存数目


        #4:生成train.txt和valid.txt
        images_name=os.listdir(images_path)
        print("一共有{}张图片".format(len(images_name)))
        train_txt_path=os.path.join(Armor_Path,"train.txt")
        valid_txt_path=os.path.join(Armor_Path,"valid.txt")
        train_txtfile=open(train_txt_path,'w')
        vaild_txtfile=open(valid_txt_path,'w')

        for count,jpg_name in enumerate(images_name):
            all_data_length=len(images_name)
            if count/all_data_length<trainRate:
                location="data/custom/images/"+jpg_name+"\n"
                train_txtfile.write(location)
            else:
                location="data/custom/images/"+jpg_name+"\n"
                vaild_txtfile.write(location)

        train_txtfile.close()
        vaild_txtfile.close()
        print("完成最终的生成")












