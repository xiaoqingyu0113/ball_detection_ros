import numpy as np
import matplotlib.pyplot as plt
import glob
import util
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import csv
import matplotlib.animation as manimation
import time

# ====================== obtain the three view images ============

imgs1 = glob.glob("./data/multiview_pingpong/cam1/*jpg")
imgs1.sort()

imgs2 = glob.glob("./data/multiview_pingpong/cam2/*jpg")
imgs2.sort()

imgs3 = glob.glob("./data/multiview_pingpong/cam3/*jpg")
imgs3.sort()


#  ====================  read corressponding points =====================

eight_points_cam1 = util.read_register_points('./data/multiview_pingpong/eight_points_cam1.txt')
eight_points_cam2= util.read_register_points('./data/multiview_pingpong/eight_points_cam2.txt')
eight_points_cam3 = util.read_register_points('./data/multiview_pingpong/eight_points_cam3.txt')



# ====================== YOLO results =======================================

yolo_rst1 = np.loadtxt('./data/multiview_pingpong/cam1_rst/results.txt')
yolo_rst2 = np.loadtxt('./data/multiview_pingpong/cam2_rst/results.txt')
yolo_rst3 = np.loadtxt('./data/multiview_pingpong/cam3_rst/results.txt')



# Intrinsics from calibration
K = np.array([1292.87465, 0., 638.506509,
          0., 1292.59105, 533.065748,
          0., 0., 1.]).reshape(3,3) 

# Extrinsics from calibration
R_1w, t_1w_1 =  util.read_extrinsics('./data/multiview_pingpong/ex1.txt')
R_2w, t_2w_2 =  util.read_extrinsics('./data/multiview_pingpong/ex2.txt')
R_3w, t_3w_3 =  util.read_extrinsics('./data/multiview_pingpong/ex3.txt')






if __name__ == '__main__':
    idx = 100

    R_w1 = R_1w.T; t_w1_w = -R_w1 @ t_1w_1
    R_w2 = R_2w.T; t_w2_w = -R_w2 @ t_2w_2
    R_w3 = R_3w.T; t_w3_w = -R_w3 @ t_3w_3

    fig = plt.figure(figsize=(14,5))
    ax_img = fig.add_subplot(121)
    ax = fig.add_subplot(122,projection='3d')






# ======================== Table Plane ==================================


    # xx, yy = (np.array([[-0.040,-0.097],[0.156,0.099]]),np.array([[-0.094,0.015],[0.0086,0.1176]]) )
    # zz = np.ones_like(xx)*(-0.0652)
    # ax.plot_surface(xx, yy, zz, color='blue',alpha=0.3)


# ======================== Register Points =======================
    register_pts_list = []
    for uv1,uv2,uv3 in zip(eight_points_cam1,eight_points_cam2,eight_points_cam3):

        start_w1, end_w1 = util.projection_img2world_line(uv=uv1,camera_params=util.CameraParam(K,R_1w,t_w1_w),z=-.2)
        start_w2, end_w2 = util.projection_img2world_line(uv=uv2,camera_params=util.CameraParam(K,R_2w,t_w2_w),z=-.2)
        start_w3, end_w3 = util.projection_img2world_line(uv=uv3,camera_params=util.CameraParam(K,R_3w,t_w3_w),z=-.2)

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
    

    # table plane
    # xx, yy = (np.array([[-0.040,-0.097],[0.156,0.099]]) ,np.array([[-0.094,0.015],[0.0086,0.1176]]) )
    # zz = np.ones_like(xx)*(-0.0652) 
    x_table = np.array([[p_v1[0],p_v2[0]],[p_v1[0]+tau_vec[0]*long_side_length/scale,p_v2[0]+tau_vec[0]*long_side_length/scale]])
    y_table = np.array([[p_v1[1],p_v2[1]],[p_v1[1]+tau_vec[1]*long_side_length/scale,p_v2[1]+tau_vec[1]*long_side_length/scale]])
    z_table = np.ones_like(x_table)*offset[2]

    tempt = np.r_[x_table.reshape(1,-1),y_table.reshape(1,-1),z_table.reshape(1,-1)]
    
    x_table_transform = [];y_table_transform = [];z_table_transform = []

    tempt_trans = R_offset.T @ (tempt - offset[:,np.newaxis])*scale
    for t in tempt_trans.T:
        x_table_transform.append(t[0])
        y_table_transform.append(t[1])
        z_table_transform.append(t[2])
    x_table_transform = np.array(x_table_transform).reshape(2,2)
    y_table_transform = np.array(y_table_transform).reshape(2,2)
    z_table_transform = np.array(z_table_transform).reshape(2,2)

    # visualize table
    # ax.plot_surface(x_table*scale, y_table*scale, z_table*scale, color='blue',alpha=0.3)
    ax.plot_surface(x_table_transform, y_table_transform, z_table_transform, color='blue',alpha=0.3)

    # visualize points on table
    # ax.scatter(register_pts_list[:,0]*scale,register_pts_list[:,1]*scale,register_pts_list[:,2]*scale,c='blue',marker='x')


# ======================= visualize camera pose ======================

    cam_scale = 0.1
    # util.draw_camera(R_w1,t_w1_w*scale,scale=cam_scale,color='blue',ax=ax)
    # util.draw_camera(R_w2,t_w2_w*scale,scale=cam_scale,color='red',ax=ax)
    # util.draw_camera(R_w3,t_w3_w*scale,scale=cam_scale,color='green',ax=ax)
    util.draw_camera(R_offset.T@R_w1,R_offset.T@(t_w1_w-offset)*scale,scale=cam_scale,color='blue',ax=ax)
    util.draw_camera(R_offset.T@R_w2,R_offset.T@(t_w2_w-offset)*scale,scale=cam_scale,color='red',ax=ax)
    util.draw_camera(R_offset.T@R_w3,R_offset.T@(t_w3_w-offset)*scale,scale=cam_scale,color='green',ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    util.axis_equal(ax,[-0.3,0.3],[-0.2,0.35],[0,0.11])



# ========================== Pingpong 3D Estimation =========================

    img_name = f"./data/multiview_pingpong/cam1/{idx:05d}.jpg"
    img = plt.imread(img_name)
    ax_img.imshow(img)

    # probability filter
    if yolo_rst1[idx,-1] < 0.5 or yolo_rst2[idx,-1] < 0.5  or yolo_rst3[idx,-1] < 0.5:
        print('prob filtered')
    
    uv1 = yolo_rst1[idx,:2].astype(int)
    uv2 = yolo_rst2[idx,:2].astype(int)
    uv3 = yolo_rst3[idx,:2].astype(int)

    mean_p, std_c = util.triangulation(uv1,uv2,uv3,util.CameraParam(K,R_1w,t_w1_w), util.CameraParam(K,R_2w,t_w2_w), util.CameraParam(K,R_3w,t_w3_w))   
    mean_p_transform = R_offset.T @ (mean_p - offset)*scale
    # outlier filter
    if std_c.max() > 0.050:
        print('std filtered')

    start_w1, end_w1 = util.projection_img2world_line(uv=uv1,camera_params=util.CameraParam(K,R_1w,t_w1_w),z=-.2)
    # util.plot_line_3d(ax,start_w1,end_w1,color='blue',linewidth=0.5)
    start_w1_transform = R_offset.T @ (start_w1 - offset)*scale
    end_w1_transform = R_offset.T @ (end_w1 - offset)*scale
    util.plot_line_3d(ax,start_w1_transform,end_w1_transform,color='blue',linewidth=0.5)

    start_w2, end_w2 = util.projection_img2world_line(uv=uv2,camera_params=util.CameraParam(K,R_2w,t_w2_w),z=-.2)
    # util.plot_line_3d(ax,start_w2,end_w2,color='red',linewidth=0.5)    start_w1_transform = R_offset.T @ (start_w1 - offset)*scale
    start_w2_transform = R_offset.T @ (start_w2 - offset)*scale
    end_w2_transform = R_offset.T @ (end_w2 - offset)*scale
    util.plot_line_3d(ax,start_w2_transform,end_w2_transform,color='blue',linewidth=0.5)

    start_w3, end_w3 = util.projection_img2world_line(uv=uv3,camera_params=util.CameraParam(K,R_3w,t_w3_w),z=-.2)
    # util.plot_line_3d(ax,start_w3,end_w3,color='green',linewidth=0.5)
    start_w3_transform = R_offset.T @ (start_w3 - offset)*scale
    end_w3_transform = R_offset.T @ (end_w3 - offset)*scale
    util.plot_line_3d(ax,start_w3_transform,end_w3_transform,color='blue',linewidth=0.5)



    ax.scatter(mean_p_transform[0],mean_p_transform[1],mean_p_transform[2],s=10,color='red')

    ax.set_ylim([-0.2*scale,0.2*scale])
    ax.set_xlim([-0.1*scale,0.3*scale])


    plt.show()
