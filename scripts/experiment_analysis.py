import csv
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import os

import util

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def yolo_stationary_test():
    # read data
    with open(DIR_PATH +"/experiment_data/yolo_stationary.txt", 'r') as f:
        lines = f.readlines()
    points_exp = []
    for line in lines:
        p = line.replace(' ', '') .replace('\n','')
        p = p.split(',')
        points_exp.append(p)
    points_exp = np.array(points_exp).astype(float)

    # ground truth
    x_gt,y_gt = np.meshgrid([0.2,0.4,0.6,0.8],[0.2,0.4,0.6,0.8])
    points_gt = np.c_[x_gt.reshape(-1),y_gt.reshape(-1), 0.020*np.ones(np.prod(x_gt.shape))]

    # total rmse
    total_rmse = np.sqrt(np.mean(np.sum((points_exp - points_gt)**2,axis=1)))
    print("total RMSE is ",total_rmse*1e3, '(mm)')

    # bias
    points_bias = (points_exp - points_gt).mean(0)
    print("bias = ", points_bias)

    # rmse_unbiased
    unbiased_rmse = np.sqrt(np.mean(np.sum((points_exp - points_gt - points_bias)**2,axis=1)))
    print("unbiased RMSE is ",unbiased_rmse*1e3, '(mm)')

    # visualize
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_exp[:,0],points_exp[:,1],points_exp[:,2],color='blue')
    ax.scatter(points_gt[:,0],points_gt[:,1],points_gt[:,2],color='red')
    util.axis_equal(ax,[0.1,0.9],[0.1,0.9],[0,0.11])
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')

    plt.show()

def yolo_straightline_test():
    # read data
    with open(DIR_PATH +"/experiment_data/yolo_output_1.csv", 'r') as f:
        lines = f.readlines()
    points_exp = []
    for line in lines:
        p = line.replace(' ', '') .replace('\n','')
        p = p.split(',')
        points_exp.append(p)
    points_exp = np.array(points_exp[1:]).astype(float)
    points_exp = points_exp[points_exp[:,1]>-100]
    points_exp = points_exp[:,1:]


    # visualize
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_exp[:,0],points_exp[:,1],points_exp[:,2],color='blue')
    util.axis_equal(ax,[0.1,0.9],[0.1,0.9],[0,0.11])
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')
    plt.show()

def zed_straightline_test():
    # read data
    with open(DIR_PATH +"/experiment_data/zed_output_1.csv", 'r') as f:
        lines = f.readlines()
    points_exp = []
    for line in lines:
        p = line.replace(' ', '') .replace('\n','')
        p = p.split(',')
        points_exp.append(p)
    points_exp = np.array(points_exp[1:]).astype(float)
    points_exp = points_exp[points_exp[:,1]>-100]
    points_exp = points_exp[:,1:]

    yolo_origin = np.array([0,0.6,0.02])
    points_exp = points_exp - points_exp[0] + yolo_origin
    points_exp[:,0]  = -points_exp[:,0]

    # visualize
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_exp[:,0],points_exp[:,1],points_exp[:,2],color='blue')
    util.axis_equal(ax,[0.1,0.9],[0.1,0.9],[0,0.11])
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')

    plt.show()


if __name__ == "__main__":
    # yolo_stationary_test()
    yolo_straightline_test()
    # zed_straightline_test() 