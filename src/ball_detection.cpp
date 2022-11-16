#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Header.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PointStamped.h"
#include "sensor_msgs/CompressedImage.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>


#include <iostream>
#include <fstream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core.hpp>
// #include <opencv4/opencv2/highgui/highgui.hpp>
#include "yolo_v2_class.hpp" 



class PingpongDetector {
  public:
    std::string cfg_file;
    std::string weights_file;
    std::string topic_sub;
    std::string topic_pub;
    std::string cam_id;

    float *image_data = new float[3932160];
    image_t yolo_image = {1024,1280,3,image_data};
    int count = 0;
    ros::Subscriber img_subscriber;
    ros::Publisher position_publisher;
    Detector *detector;

    PingpongDetector(ros::NodeHandle *);
    void  callback_img(const sensor_msgs::CompressedImageConstPtr& );
    void show_console_result(std::vector<bbox_t> const );
    void publish_best_detection(std::vector<bbox_t> const ,std_msgs::Header header);
    void compressed_to_image_t(uint8_t*);

     ~PingpongDetector()
    {
        delete[] image_data;
    }
};


PingpongDetector::PingpongDetector(ros::NodeHandle *n){
  std::cout<<"constructor"<<std::endl;
  for(int i=0;i<3932160;i++){
    image_data[i] = 0;
  }
  n->getParam("/yolo_cfg", cfg_file);
  n->getParam("/yolo_weights", weights_file);
  n->getParam(ros::this_node::getName() +"/cam_id", cam_id);

  detector =  new Detector(cfg_file,weights_file);


  topic_sub= "/camera_"+cam_id+"/image_color/compressed";
  img_subscriber = n->subscribe<sensor_msgs::CompressedImage>(topic_sub, 1, &PingpongDetector::callback_img,this);
  topic_pub = "/camera_"+cam_id+"/image_color/uv_ball";
  position_publisher = n->advertise<geometry_msgs::PointStamped>(topic_pub, 1);
}

  
void PingpongDetector::compressed_to_image_t(uint8_t* pixelPtr){
  float* pixelPtr_imt;
    pixelPtr_imt = image_data;


    for(int i =2; i< 3932160; i=i+3){
      *pixelPtr_imt = int(pixelPtr[i])/255.;
      if (*pixelPtr_imt>1.0){
        std::cout << "overflowed" <<std::endl;
        return;
      }
      pixelPtr_imt ++;
    }

    for(int i =1; i< 3932160; i=i+3){
      *pixelPtr_imt = int(pixelPtr[i])/255.;
      if (*pixelPtr_imt>1.0){
        std::cout << "overflowed" <<std::endl;
        return;
      }
      pixelPtr_imt ++;
    }
    for(int i =0; i< 3932160; i=i+3){
      *pixelPtr_imt = int(pixelPtr[i])/255.;
      if (*pixelPtr_imt>1.0){
        std::cout << "overflowed" <<std::endl;
        return;
      }
      pixelPtr_imt ++;
    }
}


void PingpongDetector::show_console_result(std::vector<bbox_t> const result_vec) {
    int x = -1;
    int y = -1;
    float prob_prev = 0.0;
    for (auto &rst: result_vec){
        if (rst.prob > prob_prev && rst.prob > 0.55){
            prob_prev = rst.prob;
            x = rst.x + rst.w/2;
            y = rst.y + rst.h/2;
        }
    }
    std::cout<< "u = " << x << "\tv = " << y << std::endl;

}

void PingpongDetector::publish_best_detection(std::vector<bbox_t> const result_vec,std_msgs::Header header) {
    int x = -1;
    int y = -1;
    float prob_prev = 0.0;
    for (auto &rst: result_vec){
        if (rst.prob > prob_prev && rst.prob > 0.55){
            prob_prev = rst.prob;
            x = rst.x + rst.w/2;
            y = rst.y + rst.h/2;
        }
    }
    geometry_msgs::Point p;
    geometry_msgs::PointStamped p_stamped;
    p.x= float(x);
    p.y = float(y);
    p.z = 0.0;
    p_stamped.header = header;
    p_stamped.point = p;
    position_publisher.publish(p_stamped);

}

void  PingpongDetector::callback_img(const sensor_msgs::CompressedImageConstPtr& msg){
 try
  {
    double cvConvert_begin_secs =ros::Time::now().toSec();
    cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_UNCHANGED);//convert compressed image data to cv::Mat
    double cvConvert_end_secs =ros::Time::now().toSec();

    uint8_t* pixelPtr = (uint8_t*)image.data;

    double convert_begin_secs =ros::Time::now().toSec();
    compressed_to_image_t(pixelPtr);
    double convert_end_secs =ros::Time::now().toSec();


    double yolo_begin_secs =ros::Time::now().toSec();
    std::vector<bbox_t> result_vec = detector->detect(yolo_image);
    double yolo_end_secs =ros::Time::now().toSec();

    // std::cout<<"Read from ROS to cv::Mat time = " << cvConvert_end_secs-cvConvert_begin_secs<<std::endl;
    // std::cout<<"Convert cv::Mat to image_t time = " << convert_end_secs-convert_begin_secs<<std::endl;
    // std::cout<<"YOLO detection time = " << yolo_end_secs-yolo_begin_secs<<std::endl;

    // show_console_result(result_vec);

    std_msgs::Header header;
    header.seq = (msg->header).seq;
    header.stamp = msg->header.stamp;
    header.frame_id = (msg->header).frame_id;

    publish_best_detection(result_vec,header);

  }
  catch (cv_bridge::Exception& e)
  {
    delete detector;
    detector = nullptr;
    ROS_ERROR("Could not convert to image!");
  }
}


// main

int main(int argc, char **argv)
{
  ros::init(argc, argv, "ball_detection_node");
  ros::NodeHandle n;
  PingpongDetector ppd = PingpongDetector(&n);
  ros::spin();
  return 0; 
}