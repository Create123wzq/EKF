# -*- coding: utf-8 -*-
'''
kalman_mousetracker.py - OpenCV mouse-tracking demo using TinyEKF
'''

# This delay will affect the Kalman update rate
DELAY_MSEC = 20  # 卡尔曼滤波计算时间间隔,单位为ms

WINDOW_NAME = 'Kalman Mousetracker [ESC to quit]'  # 窗口名称
WINDOW_SIZE = 500  # 窗口大小

import cv2
import numpy as np
from sys import exit
from EKF.tinyekf import EKF


class TrackerEKF(EKF):
    '''
    An EKF for mouse tracking
    '''

    def __init__(self):
        EKF.__init__(self, 2, 2, pval=1, qval=0.001, rval=0.1)

    def f(self, x):
        # State-transition function is identity
        return np.copy(x)

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        return np.eye(2)

    def h(self, x):
        # Observation function is identity
        return x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return np.eye(2)


class MouseInfo(object):
    '''
    A class to store X,Y points
    '''

    def __init__(self):
        self.x, self.y = -1, -1

    # If you print an object then its __str__ method will get called
    # The __str__ is intended to be as human-readable as possible
    def __str__(self):
        return '%4d %4d' % (self.x, self.y)


def mouseCallback(event, x, y, flags, mouse_info):
    '''
    Callback to update a MouseInfo object with new X,Y coordinates
    '''
    mouse_info.x = x
    mouse_info.y = y


def drawCross(img, center, r, g, b):
    '''
    Draws a cross a the specified X,Y coordinates with color RGB
    '''
    d = 5  # 调整d改变X标记大小
    thickness = 2  # 线宽
    color = (r, g, b)  # 标记颜色
    ctrx = center[0]  # 标记中心点的x坐标
    ctry = center[1]  # 标记中心点的y坐标

    # Python： cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift ] ] ])--> None
    # lineType参数之一： CV_AA - antialiased line
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, thickness, cv2.LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, thickness, cv2.LINE_AA)


def drawLines(img, points, r, g, b):
    '''
    Draws lines
    '''
    # Python: cv2.polylines(img, pts, isClosed, color[, thickness[, lineType[, shift ] ] ]) -->None
    # 参数pts: Array of polygonal curves
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def newImage():
    '''
    Returns a new image
    '''
    return np.zeros((500, 500, 3), np.uint8)  # 创建矩阵，用于保存图像内容


if __name__ == '__main__':
    # Create a new image in a named window
    img = newImage()
    cv2.namedWindow(WINDOW_NAME)

    # Create an X,Y mouse info object and set the window's mouse callback to modify it
    mouse_info = MouseInfo()  # mouse_info用于存贮当前鼠标位置

    # 设置鼠标事件回调函数
    # 参数1：name – Window name
    # 参数2：onMouse – Mouse callback.
    # 参数3：param – The optional parameter passed to the callback.
    cv2.setMouseCallback(WINDOW_NAME, mouseCallback, mouse_info)

    # Loop until mouse inside window
    while True:
        if mouse_info.x > 0 and mouse_info.y > 0:  # 鼠标进入窗口内
            break
        cv2.imshow(WINDOW_NAME, img)  # 鼠标没进入窗口内则一直显示黑色背景
        if cv2.waitKey(1) == 27:  # 检测是否按下ESC键
            exit(0)

    # These will get the trajectories for mouse location and Kalman estiamte
    measured_points = []  # 测量值列表
    kalman_points = []  # 估计值列表

    # Create a new Kalman filter for mouse tracking
    kalfilt = TrackerEKF()

    # Loop till user hits escape
    while True:
        # Serve up a fresh image
        img = newImage()

        # Grab current mouse position and add it to the trajectory
        measured = (mouse_info.x, mouse_info.y)
        measured_points.append(measured)  # 注意：程序运行时间越长（或者计算间隔越小）列表长度会越大

        # Update the Kalman filter with the mouse point, getting the estimate.
        estimate = kalfilt.step([[mouse_info.x], [mouse_info.y]])

        # Add the estimate to the trajectory
        estimated = [int(c) for c in estimate]
        kalman_points.append(estimated)  # kalman_points为2D point列表,存放每次计算出的估计值坐标

        # Display the trajectories and current points
        drawLines(img, kalman_points, 0, 255, 0)  # 绘制跟踪点移动路径
        drawCross(img, estimated, 255, 255, 255)  # X标记点，代表卡尔曼滤波估计位置
        drawLines(img, measured_points, 255, 255, 0)  # 绘制鼠标移动路径
        drawCross(img, measured, 0, 0, 255)  # X标记点，代表鼠标当前位置

        # Delay for specified interval, quitting on ESC
        cv2.imshow(WINDOW_NAME, img)  # image每隔DELAY_MSEC毫秒就刷新一次
        if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
            break

    # close the window and de-allocate any associated memory usage.
    cv2.destroyAllWindows()