#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<iostream>
#include<chrono>
#include<torch/torch.h>

using namespace std;
using namespace chrono;
// using namespace cv;
// using namespace cv::dnn;
int main()
{
    //目前OpenVINO可以用了,每一次需要ov一下就好了

    string xml="correct_shape.xml";
    string bin="correct_shape.bin";
    cv::dnn::Net net=cv::dnn::readNetFromModelOptimizer(xml,bin);
    

    cv::Mat image=cv::imread("3.jpg");
    cout<<image.size<<endl;

    while (true)
    {

        cout<<"**************************************************"<<endl;
        cv::imshow("image",image);
        cv::Mat resized_image;
        cv::resize(image,resized_image,cv::Size(512,512));
        cv::imshow("resized_image",resized_image);
        auto start=system_clock::now();
        // cv::Mat blob=cv::dnn::blobFromImage(image,1,cv::Size(512,512),cv::Scalar(100,100,100),false,false,5);//这句话也要好好看看
        cv::Mat blob=cv::dnn::blobFromImage(image,1,cv::Size(512,512),cv::Scalar(),false,false,5);//这句话也要好好看看
        cout<<"blob.size"<<blob.size<<endl;
        cout<<"blob.data"<<blob.data<<endl;

        vector<cv::Mat> images;
        cv::dnn::imagesFromBlob(blob,images);
        // cout<<"images.size"<<images.size()<<endl;
        cv::Mat image_blob=images[0];
        cout<<"iamge_blob"<<image_blob.size<<endl;
        cout<<"image_blob.type"<<image_blob.type()<<endl;
        image_blob.convertTo(image_blob,CV_8UC3);

        cout<<"origin_image.size"<<image.size<<endl;
        cv::imshow("bolb_image",image_blob);
        cv::waitKey(1);

        // cv::Mat show_image;
        // vector<cv::Mat> split_images;
        // cv::split(blob,split_images);

        // cout<<split_images.size()<<endl;
        // cout<<split_images[0].size<<endl;

        // for(int i=0;i<split_images.size();i++)
        // {
        //     cout<<"size:"<<split_images[i]<<endl;
        // }

        // cout<<"show_image.size"<<show_image.size<<endl;


        
        
        net.setInput(blob);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        
        cv::Mat detection=net.forward();
        cout<<"detection size"<<detection.size<<endl;
        vector<cv::Mat> detection_images;
        cv::dnn::imagesFromBlob(detection,detection_images);
        cv::Mat detection_image=detection_images[0];
        cout<<"detection_image.size"<<detection_image.size<<endl;
        cv::Mat sigle_Mat[5];
        cv::split(detection_image,sigle_Mat);

        //生成对应图层
        cv::Mat heatmap=sigle_Mat[0];
        cv::imshow("output",heatmap);
        cv::resize(heatmap,heatmap,cv::Size(512,512));

        double minval;
        double maxval;
        

        cv::minMaxLoc(heatmap,&maxval,&minval,NULL,NULL);
        cout<<"max:"<<maxval<<endl;
        cout<<"min:"<<minval<<endl;

        cv::Mat outputmap;
        // cv::normalize(heatmap,outputmap,minval,maxval,cv::NORM_MINMAX);
        cv::normalize(heatmap,outputmap,255,0,cv::NORM_MINMAX);
        // cv::normalize(heatmap,outputmap,255,0,cv::NORM_MINMAX);
        outputmap.convertTo(outputmap,CV_8UC1);
        cv::Mat color_map;

        cv::applyColorMap(outputmap, color_map, cv::COLORMAP_AUTUMN);

        cv::imshow("temp_iamge",color_map);

        
        // cout<<"heamap max"<<cv::minMaxLoc(heatmap)<<endl;
        




        auto end=system_clock::now();
        auto time=duration_cast<milliseconds>(end-start);
        cout<<"识别一次的耗时为:"<<double(time.count())<<"ms"<<endl;

        cv::waitKey(1);


    }
    

    cv::imshow("image",image);
    cv::waitKey(0);
    std::cout<<CV_VERSION<<std::endl;
}
