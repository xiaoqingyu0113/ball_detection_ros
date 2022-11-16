#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point,PointStamped
from std_msgs.msg import Header
import numpy as np
import message_filters
import util
import os 


def load_params():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    eight_points_cam1 = util.read_register_points(dir_path+'/camera_calibration_data/eight_points_cam1.txt')
    eight_points_cam2 = util.read_register_points(dir_path+'/camera_calibration_data/eight_points_cam2.txt')
    eight_points_cam3 = util.read_register_points(dir_path+'/camera_calibration_data/eight_points_cam3.txt')

    K = np.array([1292.87465, 0., 638.506509,
        0., 1292.59105, 533.065748,
        0., 0., 1.]).reshape(3,3) 

    # Extrinsics from calibration
    R_1w, t_1w_1 =  util.read_extrinsics(dir_path+'/camera_calibration_data/ex1.txt')
    R_2w, t_2w_2 =  util.read_extrinsics(dir_path+'/camera_calibration_data/ex2.txt')
    R_3w, t_3w_3 =  util.read_extrinsics(dir_path+'/camera_calibration_data/ex3.txt')

    R_w1 = R_1w.T; t_w1_w = -R_w1 @ t_1w_1
    R_w2 = R_2w.T; t_w2_w = -R_w2 @ t_2w_2
    R_w3 = R_3w.T; t_w3_w = -R_w3 @ t_3w_3

    # ======================== Register Points =======================
    register_pts_list = []
    for uv1,uv2,uv3 in zip(eight_points_cam1,eight_points_cam2,eight_points_cam3):
        closest_pts,std = util.triangulation(uv1,uv2,uv3,util.CameraParam(K,R_1w,t_w1_w), util.CameraParam(K,R_2w,t_w2_w), util.CameraParam(K,R_3w,t_w3_w))
        register_pts_list.append(closest_pts)
    
    register_pts_list = np.r_[register_pts_list]
    
    # real table dimension
    short_side_length = 1.525 # m
    long_side_length = 2.749 # m

    # compute scale, and relative pose
    scale = short_side_length/ np.linalg.norm(register_pts_list[0,:] - register_pts_list[1,:])
    offset = register_pts_list[0,:]
    p_v1 = register_pts_list[0,:];p_v2 = register_pts_list[1,:]

    eta_vec = register_pts_list[1,:] - register_pts_list[0,:]; eta_vec = eta_vec/np.linalg.norm(eta_vec)
    tau_vec = np.array([eta_vec[1], -eta_vec[0],0]); tau_vec = tau_vec/np.linalg.norm(tau_vec)
    ksi_vec = np.cross(tau_vec,eta_vec)
    R_offset = np.c_[tau_vec,eta_vec,ksi_vec]

    return util.CameraParam(K,R_1w,t_w1_w),util.CameraParam(K,R_2w,t_w2_w),util.CameraParam(K,R_3w,t_w3_w), R_offset,offset,scale

class TriangulationTransformer:
    def __init__(self) -> None:

        cam_param1, cam_param2, cam_param3,R_offset, offset, scale = load_params()
        self.cam_param1 = cam_param1
        self.cam_param2 = cam_param2
        self.cam_param3 = cam_param3
        self.R_offset = R_offset
        self.offset = offset
        self.scale = scale

    def compute(self,uv1,uv2,uv3):
        mean_p, std_c = util.triangulation(uv1,uv2,uv3,self.cam_param1, self.cam_param2, self.cam_param3)   
        mean_p_transform = self.R_offset.T @ (mean_p - self.offset)*self.scale
        if mean_p[0] <-100:
            p = Point()
            p.x = mean_p[0]
            p.y = mean_p[1]
            p.z = mean_p[2]
        else:
            p = Point()
            p.x = mean_p_transform[0]
            p.y = mean_p_transform[1]
            p.z = mean_p_transform[2]
        return p


class MultiBallSync:
    def __init__(self, debug=False, init_ros_node=False):

        rospy.init_node('image_sync', anonymous=True)

        self.image_sub1 = message_filters.Subscriber("/camera_1/image_color/uv_ball", PointStamped)
        self.image_sub2 = message_filters.Subscriber("/camera_2/image_color/uv_ball",PointStamped)
        self.image_sub3 = message_filters.Subscriber("/camera_3/image_color/uv_ball", PointStamped)

        self.tss = message_filters.ApproximateTimeSynchronizer([self.image_sub1, self.image_sub2,self.image_sub3],
                                                               queue_size=30, slop=0.030)
        self.tss.registerCallback(self.callback)

        self.pub = rospy.Publisher('/ball_xyz', PointStamped, queue_size=1)

        self.tt = TriangulationTransformer()


    def callback(self,msg1,msg2,msg3):
        t = rospy.Time.now().to_sec()
        uv1= np.array([msg1.point.x,msg1.point.y]).astype('int')
        uv2= np.array([msg2.point.x,msg2.point.y]).astype('int')
        uv3= np.array([msg3.point.x,msg3.point.y]).astype('int')

        p = self.tt.compute(uv1,uv2,uv3)
        header = Header()
        header.stamp = rospy.Time.now()
        p_stamped = PointStamped()
        p_stamped.header = header
        p_stamped.point = p
        self.pub.publish(p_stamped)
        # print(rospy.Time.now().to_sec() - t)


if __name__ == '__main__':
    try:
        mbs = MultiBallSync()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass