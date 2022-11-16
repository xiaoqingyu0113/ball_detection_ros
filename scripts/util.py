import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import csv

def read_extrinsics(txtfile):
    data = np.loadtxt(txtfile)
    return data[:3,:], data[3,:]

def read_register_points(txtfile):

    eight_points_cam2= []
    with open(txtfile) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            eight_points_cam2.append(row)
        eight_points_cam2= np.array(eight_points_cam2).astype(int)

    return eight_points_cam2 

def draw_camera(R, t, color='blue', scale=1, ax=None):
    points = np.array([[0,0,0],[1,1,2],[0,0,0],[-1,1,2],[0,0,0],[1,-1,2],[0,0,0],[-1,-1,2],
                        [-1,1,2],[1,1,2],[1,-1,2],[-1,-1,2]]) * scale
    
    x,y,z = R @ points.T + t[:,np.newaxis]

    if ax is not None:
        ax.plot(x,y,z)
        return ax
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(x,y,z)
        return ax

def axis_equal(ax,X,Y,Z):
    X = np.array(X);Y = np.array(Y);Z = np.array(Z)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

class CameraParam:
    def __init__(self,K,R,t):
        self.K = K
        self.R = R
        self.t =t


def projection_img2world_line(uv,camera_params,z=-2.0):
    # t = np.array(camera_params.t)
    # t = np.expand_dims(t,axis = 1)
    # R = np.array(camera_params.R)
    # K = np.array(camera_params.K)
    # M = K @ np.block([R,-R@t])
    M = camParam2proj(camera_params)
    
    u = uv[0];v = uv[1]
    A = np.array([
        [M[0,0] - u*M[2,0], M[0,1] - u*M[2,1]],
        [M[1,0] - v*M[2,0], M[1,1] - v*M[2,1]]
        ])
    b1 = np.array([
        [M[0,2] - u*M[2,2]],
        [M[1,2] - v*M[2,2]]
        ])
    b2 = np.array([
        [M[0,3] - u*M[2,3]],
        [M[1,3] - v*M[2,3]]
        ])
    xy = np.linalg.solve(A,-b1*z-b2)

    start = camera_params.t
    end = np.append(xy,z)

    return start, end

def plot_line_3d(ax,start, end, **kwargs):
    ax.plot([start[0],end[0]], [start[1],end[1]],[start[2],end[2]],**kwargs)


def camParam2proj(cam_param):
    t = np.array(cam_param.t)
    t = np.expand_dims(t,axis = 1)
    R = np.array(cam_param.R)
    K = np.array(cam_param.K)
    M = K @ np.block([R,-R@t])
    return M

def closest_points_twoLine(p0,p1,q0,q1): # https://blog.csdn.net/Hunter_pcx/article/details/78577202
    A = np.c_[p0-p1,q0-q1]
    b = q1-p1

    # solve normal equation
    x = np.linalg.solve(A.T@A, A.T@b)
    
    a = x[0]
    n = -x[1]

    pc = a*p0 + (1-a)*p1
    qc = n*q0 + (1-n)*q1

    return pc, qc

def triangulation(uv1,uv2,uv3,cam_param1, cam_param2, cam_param3):
    THRESH = 0.005
    closest_pts = []

    false_detection = np.array([uv1[0],uv2[0],uv3[0]]) == -1
    available_idx = np.arange(3)
    available_idx = available_idx[~false_detection]
    N_available = len(available_idx)

    uv_list = [uv1,uv2,uv3]
    cam_param_list = [cam_param1, cam_param2, cam_param3]

    if N_available <2:
        return np.array([-111,-111,-111]),-111

    elif N_available == 2:
        p0,p1 =  projection_img2world_line(uv_list[available_idx[0]],cam_param_list[available_idx[0]],z=-1.0)
        q0,q1 =  projection_img2world_line(uv_list[available_idx[1]],cam_param_list[available_idx[1]],z=-1.0)
        pc,qc = closest_points_twoLine(p0,p1,q0,q1)
        if np.linalg.norm(pc-qc) > THRESH:
            return np.array([-222,-222,-222]),-222
        else:
            return (pc+qc)/2.,  np.linalg.norm(pc-qc)/2.

    else:

        p0,p1 = projection_img2world_line(uv1,cam_param1,z=-1.0)
        q0,q1 = projection_img2world_line(uv2,cam_param2,z=-1.0)
        r0,r1 = projection_img2world_line(uv3,cam_param3,z=-1.0)

        
        pc,qc = closest_points_twoLine(p0,p1,q0,q1)
        closest_pts.append([pc]); closest_pts.append([qc])
        pc,rc = closest_points_twoLine(p0,p1,r0,r1)
        closest_pts.append([pc]); closest_pts.append([rc])
        qc,rc = closest_points_twoLine(q0,q1,r0,r1)
        closest_pts.append([qc]); closest_pts.append([rc])

        closest_pts = np.array(closest_pts).reshape((-1,3))

        knn_dist = pairwise_dist(closest_pts,closest_pts)

        idx_filtered = []
        for idx in range(6):
            dist = knn_dist[idx]
            if len(dist<THRESH) <2:
                continue    
            else:
                idx_filtered.append(idx)
        
        if len(idx_filtered) < 2:
            return np.array([-333,-333,-333]),-333
        else:
            p_return  =  closest_pts.mean(0)
            s_return = closest_pts.std(0)
            if s_return.max()  > THRESH:
                return np.array([-444,-444,-444]),-444
            else:
                return p_return, s_return


def pairwise_dist(x,y):
    """
    x: N,D
    y: M,D

    return N,M
    """
    d_sq = np.sum(x**2,axis=1)[:,np.newaxis] + np.sum(y**2,axis=1)[np.newaxis,:] - 2*x @ y.T

    d_sq[d_sq<0] = 0

    return np.sqrt(d_sq)