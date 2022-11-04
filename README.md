## Getting Started
* install [opencv](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
* install [cuda](https://developer.nvidia.com/cuda-toolkit)
* Compile [DarkNet](https://github.com/AlexeyAB/darknet), put the library in ./lib/

## Download Pretrained Model
* Download the pretrain weights file from [here](https://u.pcloud.link/publink/show?code=XZ6qqfVZR4O5OuXKdDRBkPKBToTYzXwGRyWy).

## Build The Package
use `catkin_make` or `catkin build`

## Set Configuration
set correct path in ./config/yolo_cfg_path.yaml

## start the node
`roslaunch ball_detection_ros ball_detection' 

