# -*- coding: utf-8 -*-
'''
kalman_motionestimation.py - demo using TinyEKF
'''
import cv2
import numpy as np
import math
from EKF.tinyekf import EKF
from EKF.utils import fromCameraToImage, draw, drawQuaternions, getImagePosition2, fromImageToPixel, drawPoint, getQuaternion


DELAY_MSEC = 1 # 卡尔曼滤波计算时间间隔,单位为s
F = 36 # 焦距 mm
CCD_SIZE = 11.264 # 传感器尺寸 mm
WIDTH = 2048
HEIGHT = 2048

M = 4 # 卡尔曼滤波器跟踪的点的个数

class TrackerEKF(EKF):

    def __init__(self, n, m, s0, q0, points0, w0, v0, p, q, r, pre=None, ifCre=True):
        EKF.__init__(self, n, m, pval=p, qval=q, rval=r)
        #初始状态估计
        if ifCre:
            self.x[0][0] = s0[0]/s0[2]
            self.x[1][0] = s0[1]/s0[2]
            self.x[2][0] = v0[0]/s0[2]
            self.x[3][0] = v0[1]/s0[2]
            self.x[4][0] = v0[2]/s0[2]
            self.x[5][0] = q0[0]
            self.x[6][0] = q0[1]
            self.x[7][0] = q0[2]
            self.x[8][0] = q0[3]
            self.x[9][0] = w0[0]
            self.x[10][0] = w0[1]
            self.x[11][0] = w0[2]
            for i in range(0, M):
                self.x[3*i+12][0] = points0[i][0]/s0[2]
                self.x[3*i+13][0] = points0[i][1]/s0[2]
                self.x[3*i+14][0] = points0[i][2]/s0[2]
        else:
            self.x = pre[:-3]
        print("s0:")
        print(self.x)

    def f(self, x):
        # 状态转移函数
        st = np.zeros((3*M+12, 1))
        s1 = x[0][0]
        s2 = x[1][0]
        s3 = x[2][0]
        s4 = x[3][0]
        s5 = x[4][0]
        s6 = x[5][0]
        s7 = x[6][0]
        s8 = x[7][0]
        s9 = x[8][0]
        s10 = x[9][0]
        s11 = x[10][0]
        s12 = x[11][0]
        st[0][0] = s3 - s1*s5
        st[1][0] = s4 - s2*s5
        st[2][0] = -s3*s5
        st[3][0] = -s4*s5
        st[4][0] = -s5**2
        st[5][0] = 0.5*(s12*s7 - s11*s8 + s10*s9)
        st[6][0] = 0.5*(-s12*s6 + s10*s8 + s11*s9)
        st[7][0] = 0.5*(s11*s6 - s10*s7 + s12*s9)
        st[8][0] = 0.5*(-s10*s6 - s11*s7 - s12*s8)
        st[9][0] = 0
        st[10][0] = 0
        st[11][0] = 0
        for i in range(0, 3*M):
            st[i+12][0] = -x[i+12][0]*s5

        return DELAY_MSEC * st + x

    def getF(self, x):
        # 状态转移函数的雅克比矩阵
        s1 = x[0][0]
        s2 = x[1][0]
        s3 = x[2][0]
        s4 = x[3][0]
        s5 = x[4][0]
        s6 = x[5][0]
        s7 = x[6][0]
        s8 = x[7][0]
        s9 = x[8][0]
        s10 = x[9][0]
        s11 = x[10][0]
        s12 = x[11][0]

        jac = np.zeros((3*M+12, 3*M+12))
        jac[0][0] = -s5
        jac[0][2] = 1
        jac[0][4] = -s1
        jac[1][1] = -s5
        jac[1][3] = 1
        jac[1][4] = -s2
        jac[2][2] = -s5
        jac[2][4] = -s3
        jac[3][3] = -s5
        jac[3][4] = -s4
        jac[4][4] = -2*s5
        jac[5][6] = 0.5*s12
        jac[5][7] = -0.5*s11
        jac[5][8] = 0.5*s10
        jac[5][9] = 0.5*s9
        jac[5][10] = -0.5*s8
        jac[5][11] = 0.5*s7
        jac[6][5] = -0.5*s12
        jac[6][7] = 0.5*s10
        jac[6][8] = 0.5*s11
        jac[6][9] = 0.5*s8
        jac[6][10] = 0.5*s9
        jac[6][11] = -0.5*s6
        jac[7][5] = 0.5*s11
        jac[7][6] = -0.5*s10
        jac[7][8] = 0.5*s12
        jac[7][9] = -0.5*s7
        jac[7][10] = 0.5*s6
        jac[7][11] = 0.5*s9
        jac[8][5] = -0.5*s10
        jac[8][6] = -0.5*s11
        jac[8][7] = -0.5*s12
        jac[8][9] = -0.5*s6
        jac[8][10] = -0.5*s7
        jac[8][11] = -0.5*s8
        for i in range(0, 3*M):
            jac[i+12][i+12] = -s5
            jac[i+12][4] = -x[i+12][0]

        return DELAY_MSEC*jac + np.eye(3*M+12)

    def h(self, x):
        # 观测函数
        s1 = x[0][0]
        s2 = x[1][0]
        s6 = x[5][0]
        s7 = x[6][0]
        s8 = x[7][0]
        s9 = x[8][0]
        ob = np.zeros((2*M,1))
        for i in range(0, M):
            n1 = s1 + np.dot(np.array([[s6**2-s7**2-s8**2+s9**2, 2*(s6*s7+s8*s9), 2*(s6*s8-s7*s9)]]),
                                  np.array([[x[i*3+12][0]], [x[i*3+13][0]], [x[i*3+14][0]]]))[0][0]

            n2 = s2 + np.dot(np.array([[2*(s6*s7-s8*s9), -s6**2+s7**2-s8**2+s9**2, 2*(s7*s8+s6*s9)]]),
                                  np.array([[x[i*3+12][0]], [x[i*3+13][0]], [x[i*3+14][0]]]))[0][0]

            d = 1 + np.dot(np.array([[2*(s6*s8+s7*s9), 2*(s7*s8-s6*s9), -s6**2-s7**2+s8**2+s9**2]]),
                           np.array([[x[i*3+12][0]], [x[i*3+13][0]], [x[i*3+14][0]]]))[0][0]
            ob[i*2][0] = n1/d
            ob[i*2+1][0] = n2/d

        return F*ob

    def getH(self, x):
        # 观测函数的雅克比矩阵
        s1 = x[0][0]
        s2 = x[1][0]
        s6 = x[5][0]
        s7 = x[6][0]
        s8 = x[7][0]
        s9 = x[8][0]

        jac = np.zeros((2*M, 3*M+12))
        for i in range(0, M):
            s13 = x[i*3+12][0]
            s14 = x[i*3+13][0]
            s15 = x[i*3+14][0]
            jac[i*2][0] = 1/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)
            jac[i*2][5] = (2*s13*s6 + 2*s14*s7 + 2*s15*s8)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1) + (-2*s13*s8 + 2*s14*s9 + 2*s15*s6)*(s1 + s13*(s6**2 - s7**2 - s8**2 + s9**2) + 2*s14*(s6*s7 + s8*s9) + 2*s15*(s6*s8 - s7*s9))/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2
            jac[i*2][6] = (-2*s13*s7 + 2*s14*s6 - 2*s15*s9)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1) + (-2*s13*s9 - 2*s14*s8 + 2*s15*s7)*(s1 + s13*(s6**2 - s7**2 - s8**2 + s9**2) + 2*s14*(s6*s7 + s8*s9) + 2*s15*(s6*s8 - s7*s9))/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2
            jac[i*2][7] = (-2*s13*s6 - 2*s14*s7 - 2*s15*s8)*(s1 + s13*(s6**2 - s7**2 - s8**2 + s9**2) + 2*s14*(s6*s7 + s8*s9) + 2*s15*(s6*s8 - s7*s9))/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2 + (-2*s13*s8 + 2*s14*s9 + 2*s15*s6)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)
            jac[i*2][8] = (-2*s13*s7 + 2*s14*s6 - 2*s15*s9)*(s1 + s13*(s6**2 - s7**2 - s8**2 + s9**2) + 2*s14*(s6*s7 + s8*s9) + 2*s15*(s6*s8 - s7*s9))/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2 + (2*s13*s9 + 2*s14*s8 - 2*s15*s7)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)
            jac[i*2][i*3+12] = (-2*s6*s8 - 2*s7*s9)*(s1 + s13*(s6**2 - s7**2 - s8**2 + s9**2) + 2*s14*(s6*s7 + s8*s9) + 2*s15*(s6*s8 - s7*s9))/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2 + (s6**2 - s7**2 - s8**2 + s9**2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)
            jac[i*2][i*3+13] = (2*s6*s7 + 2*s8*s9)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1) + (2*s6*s9 - 2*s7*s8)*(s1 + s13*(s6**2 - s7**2 - s8**2 + s9**2) + 2*s14*(s6*s7 + s8*s9) + 2*s15*(s6*s8 - s7*s9))/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2
            jac[i*2][i*3+14] = (2*s6*s8 - 2*s7*s9)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1) + (s1 + s13*(s6**2 - s7**2 - s8**2 + s9**2) + 2*s14*(s6*s7 + s8*s9) + 2*s15*(s6*s8 - s7*s9))*(s6**2 + s7**2 - s8**2 - s9**2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2

            jac[i*2+1][1] = 1/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)
            jac[i*2+1][5] = (2*s13*s7 - 2*s14*s6 + 2*s15*s9)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1) + (-2*s13*s8 + 2*s14*s9 + 2*s15*s6)*(2*s13*(s6*s7 - s8*s9) + s14*(-s6**2 + s7**2 - s8**2 + s9**2) + 2*s15*(s6*s9 + s7*s8) + s2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2
            jac[i*2+1][6] = (2*s13*s6 + 2*s14*s7 + 2*s15*s8)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1) + (-2*s13*s9 - 2*s14*s8 + 2*s15*s7)*(2*s13*(s6*s7 - s8*s9) + s14*(-s6**2 + s7**2 - s8**2 + s9**2) + 2*s15*(s6*s9 + s7*s8) + s2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2
            jac[i*2+1][7] = (-2*s13*s6 - 2*s14*s7 - 2*s15*s8)*(2*s13*(s6*s7 - s8*s9) + s14*(-s6**2 + s7**2 - s8**2 + s9**2) + 2*s15*(s6*s9 + s7*s8) + s2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2 + (-2*s13*s9 - 2*s14*s8 + 2*s15*s7)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)
            jac[i*2+1][8] = (-2*s13*s7 + 2*s14*s6 - 2*s15*s9)*(2*s13*(s6*s7 - s8*s9) + s14*(-s6**2 + s7**2 - s8**2 + s9**2) + 2*s15*(s6*s9 + s7*s8) + s2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2 + (-2*s13*s8 + 2*s14*s9 + 2*s15*s6)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)
            jac[i*2+1][i*3+12] = (2*s6*s7 - 2*s8*s9)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1) + (-2*s6*s8 - 2*s7*s9)*(2*s13*(s6*s7 - s8*s9) + s14*(-s6**2 + s7**2 - s8**2 + s9**2) + 2*s15*(s6*s9 + s7*s8) + s2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2
            jac[i*2+1][i*3+13] = (2*s6*s9 - 2*s7*s8)*(2*s13*(s6*s7 - s8*s9) + s14*(-s6**2 + s7**2 - s8**2 + s9**2) + 2*s15*(s6*s9 + s7*s8) + s2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2 + (-s6**2 + s7**2 - s8**2 + s9**2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)
            jac[i*2+1][i*3+14] = (2*s6*s9 + 2*s7*s8)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1) + (s6**2 + s7**2 - s8**2 - s9**2)*(2*s13*(s6*s7 - s8*s9) + s14*(-s6**2 + s7**2 - s8**2 + s9**2) + 2*s15*(s6*s9 + s7*s8) + s2)/(2*s13*(s6*s8 + s7*s9) + 2*s14*(-s6*s9 + s7*s8) + s15*(-s6**2 - s7**2 + s8**2 + s9**2) + 1)**2


        return F*jac

if __name__ == '__main__':
    # 小正方体特征点索引
    indexs = [1, 7, 3, 5]
    cube_points = [(-100, 100, -100), (-100, -100, -100), (100, 100, -100), (100, -100, -100), (100, 100, 100), (100, -100, 100), (-100, 100, 100), (-100, -100, 100)]
    # 测量值列表
    measured_points = []
    quaternions = []
    for i in range(180):
        temp = []
        for index in indexs:
            pos = getImagePosition2(cube_points[index][0], cube_points[index][1], cube_points[index][2], 0, 0, i)
            # 相机坐标系转换到图像坐标系
            x, y = fromCameraToImage(pos[0], pos[1], pos[2])
            # temp.append((x, y))
            temp.append([x])
            temp.append([y])

        quaternions.append(getQuaternion(-45, 0, -i))
        measured_points.append(temp)

    # 验证坐标是否正确
    # for i in range(180):
    #     prefix = "/Users/apple/Desktop/test/img/"
    #     filename = "img" + "0" * (4 - len(str(i))) + str(i) + ".jpg"
    #
    #     img = cv2.imread(prefix + filename)
    #     result = []
    #     for p in measured_points[i]:
    #         result.append((fromImageToPixel(p[0], p[1])))
    #     drawPoint(img, result)
    #     cv2.imwrite("test/" + filename, img)


    # 估计值列表
    kalman_points = []
    # 初值:小正方体坐标系原点在相机坐标系下的位置(y值要取反！！！！！！！！！！)
    s0 = [0., 0., 2121]
    # 初值:小正方体坐标系到相机坐标系坐标系旋转矩阵(四元数)
    q0 = [-0.383, 0., 0., 0.924]
    # M个特征点在小行星坐标系下的坐标（不随时间变化,y值要取反！！！！！！！！！！)
    points = [[-100, -100, -100], [-100, -100, 100], [100, -100, -100], [100, -100, 100]]
    # 初值：小行星坐标系x,y,z方向的角速度初值
    w0 = [0., 0., 0.]
    # 初值：小行星坐标原点在相机坐标系下x,y,z方向速度初值
    v0 = [0., 0., 0.]

    # kalfilt = TrackerEKF(s0, q0, points, w0, v0, p=0.2,  q=0.0002, r=0.000002)
    kalfilt = TrackerEKF(3*M+12, 2*M, s0, q0, points, w0, v0, p=0.02, q=0.0002, r=0.000002)

    frameIndex = 86
    for i in range(1, frameIndex):
        # kalfilt.x[14][0] = 0
        # 更新卡尔曼滤波器，获取预测值
        estimate = kalfilt.step(measured_points[i])

        # 四元数在update之后要归一化！！！！
        num = math.sqrt(kalfilt.x[5][0]**2 + kalfilt.x[6][0]**2 + kalfilt.x[7][0]**2 + kalfilt.x[8][0]**2)

        kalfilt.x[5][0] /= num
        kalfilt.x[6][0] /= num
        kalfilt.x[7][0] /= num
        kalfilt.x[8][0] /= num
        # kalfilt.x[14][0] = 0
        np.savetxt("result/kalman-"+str(i), kalfilt.x)
        kalman_points.append(kalfilt.x)

    # 测试丢失一个特征点的结果
    M = 3
    kalfilt = TrackerEKF(3 * M + 12, 2 * M, s0, q0, points, w0, v0, p=0.02, q=0.0002, r=0.000002, pre=kalman_points[-1], ifCre=False)
    for i in range(frameIndex, 180):
        # kalfilt.x[14][0] = 0
        # 更新卡尔曼滤波器，获取预测值

        estimate = kalfilt.step(measured_points[i][:-2])

        # 四元数在update之后要归一化！！！！
        num = math.sqrt(kalfilt.x[5][0]**2 + kalfilt.x[6][0]**2 + kalfilt.x[7][0]**2 + kalfilt.x[8][0]**2)

        kalfilt.x[5][0] /= num
        kalfilt.x[6][0] /= num
        kalfilt.x[7][0] /= num
        kalfilt.x[8][0] /= num
        # kalfilt.x[14][0] = 0
        np.savetxt("result/kalman-"+str(i), kalfilt.x)
        kalman_points.append(kalfilt.x)

    draw(kalman_points, [0, 1], ['X', 'Y'])
    draw(kalman_points, [2, 3, 4], ['Vx', 'Vy', 'Vz'])
    draw(kalman_points, [5, 6, 7, 8], ['q1', 'q2', 'q3', 'q4'])
    draw(kalman_points, [9, 10, 11], ['Wx', 'Wy', 'Wz'])
    # for i in range(M):
    #     draw(kalman_points, [12+i*3, 13+i*3, 14+i*3], ['X-'+str(i+1), 'Y-'+str(i+1), 'Z-'+str(i+1)])
    for i in range(M+1):
        if i != M:
            draw(kalman_points, [12+i*3, 13+i*3, 14+i*3], ['X-'+str(i+1), 'Y-'+str(i+1), 'Z-'+str(i+1)])
        else:
            draw(kalman_points[:frameIndex-1], [12 + i * 3, 13 + i * 3, 14 + i * 3], ['X-' + str(i + 1), 'Y-' + str(i + 1), 'Z-' + str(i + 1)])
    drawQuaternions(kalman_points, [5, 6, 7, 8], ['q1', 'q2', 'q3', 'q4'], quaternions)




