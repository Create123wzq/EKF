import cv2
import numpy as np
import math
from EKF.tinyekf import EKF
from EKF.utils import draw, drawQuaternions2, getImagePosition2, getImagePosition3, getQuaternion2, changeToFrame1


DELAY_MSEC = 1 # 卡尔曼滤波计算时间间隔,单位为s

M = 4 # 卡尔曼滤波器跟踪的点的个数

class TrackerEKF(EKF):

    def __init__(self, s0, q0, points0, w0, v0, p, q, r):
        EKF.__init__(self, 3*M+13, 3*M, pval=p, qval=q, rval=r)
        #初始状态估计
        self.x[0][0] = s0[0]
        self.x[1][0] = s0[1]
        self.x[2][0] = s0[2]

        self.x[3][0] = v0[0]
        self.x[4][0] = v0[1]
        self.x[5][0] = v0[2]

        self.x[6][0] = q0[0]
        self.x[7][0] = q0[1]
        self.x[8][0] = q0[2]
        self.x[9][0] = q0[3]

        self.x[10][0] = w0[0]
        self.x[11][0] = w0[1]
        self.x[12][0] = w0[2]
        for i in range(0, M):
            self.x[3*i+13][0] = points0[i][0]
            self.x[3*i+14][0] = points0[i][1]
            self.x[3*i+15][0] = points0[i][2]
        print("s0:")
        print(self.x)

    def f(self, x):
        # 状态转移函数
        st = np.zeros((3*M+13, 1))

        s4 = x[3][0]
        s5 = x[4][0]
        s6 = x[5][0]
        s7 = x[6][0]
        s8 = x[7][0]
        s9 = x[8][0]
        s10 = x[9][0]
        s11 = x[10][0]
        s12 = x[11][0]
        s13 = x[12][0]
        st[0][0] = s4
        st[1][0] = s5
        st[2][0] = s6

        st[6][0] = 0.5*(-s11 * s8 - s12 * s9 - s13 * s10)
        st[7][0] = 0.5*(-s13*s9 + s11*s7 + s12*s10)
        st[8][0] = 0.5*(s13*s8 - s11*s10 + s12*s7)
        st[9][0] = 0.5*(-s12*s8 + s11*s9 + s13*s7)



        return DELAY_MSEC * st + x

    def getF(self, x):
        # 状态转移函数的雅克比矩阵
        s7 = x[6][0]
        s8 = x[7][0]
        s9 = x[8][0]
        s10 = x[9][0]
        s11 = x[10][0]
        s12 = x[11][0]
        s13 = x[12][0]

        jac = np.zeros((3*M+13, 3*M+13))
        jac[0][3] = 1
        jac[1][4] = 1
        jac[2][5] = 1

        jac[6][7] = -0.5 * s11
        jac[6][8] = -0.5 * s12
        jac[6][9] = -0.5 * s13
        jac[6][10] = -0.5 * s8
        jac[6][11] = -0.5 * s9
        jac[6][12] = -0.5 * s10

        jac[7][6] = 0.5 * s11
        jac[7][8] = -0.5 * s13
        jac[7][9] = 0.5 * s12
        jac[7][10] = 0.5 * s7
        jac[7][11] = 0.5 * s10
        jac[7][12] = -0.5 * s9

        jac[8][6] = 0.5 * s12
        jac[8][7] = 0.5 * s13
        jac[8][9] = -0.5 * s11
        jac[8][10] = -0.5 * s10
        jac[8][11] = 0.5 * s7
        jac[8][12] = 0.5 * s8

        jac[9][6] = 0.5 * s13
        jac[9][7] = -0.5 * s12
        jac[9][8] = 0.5 * s11
        jac[9][10] = 0.5 * s9
        jac[9][11] = -0.5 * s8
        jac[9][12] = s7


        return DELAY_MSEC*jac + np.eye(3*M+13)

    def h(self, x):
        # 观测函数
        s1 = x[0][0]
        s2 = x[1][0]
        s3 = x[2][0]

        s7 = x[6][0]
        s8 = x[7][0]
        s9 = x[8][0]
        s10 = x[9][0]
        ob = np.zeros((3*M,1))
        for i in range(0, M):
            s14 = x[3*i+13][0]
            s15 = x[3*i+14][0]
            s16 = x[3*i+15][0]
            ob[3*i][0] = s1 + (1-2*s9**2-2*s10**2)*s14 + (2*s8*s9-2*s7*s10)*s15 + (2*s8*s10+2*s7*s9)*s16
            ob[3*i+1][0] = s2 + (2*s8*s9+2*s7*s10)*s14 + (1-2*s8**2-2*s10**2)*s15 + (2*s9*s10-2*s7*s8)*s16
            ob[3*i+2][0] = s3 +(2*s8*s10-2*s7*s9)*s14 + (2*s9*s10+2*s7*s8)*s15 + (1-2*s8**2-2*s9**2)*s16

        return ob

    def getH(self, x):
        # 观测函数的雅克比矩阵

        s7 = x[6][0]
        s8 = x[7][0]
        s9 = x[8][0]
        s10 = x[9][0]

        jac = np.zeros((3*M, 3*M+13))
        for i in range(0, M):
            s14 = x[i*3+13][0]
            s15 = x[i*3+14][0]
            s16 = x[i*3+15][0]

            jac[i*3][0] = 1
            jac[i*3][6] = -2*s10*s15 + 2*s16*s9
            jac[i*3][7] = 2*s10*s16 + 2*s15*s9
            jac[i*3][8] = -4*s14*s9 + 2*s15*s8 + 2*s16*s7
            jac[i*3][9] = -4*s10*s14 - 2*s15*s7 + 2*s16*s8
            jac[i*3][i*3+13] = -2*s10**2 - 2*s9**2 + 1
            jac[i*3][i*3+14] = -2*s10*s7 + 2*s8*s9
            jac[i*3][i*3+15] = 2*s10*s8 + 2*s7*s9

            jac[i*3+1][1] = 1
            jac[i*3+1][6] = 2*s10*s14 - 2*s16*s8
            jac[i*3+1][7] = 2*s14*s9 - 4*s15*s8 - 2*s16*s7
            jac[i*3+1][8] = 2*s10*s16 + 2*s14*s8
            jac[i*3+1][9] = -4*s10*s15 + 2*s14*s7 + 2*s16*s9
            jac[i*3+1][i*3+13] = 2*s10*s7 + 2*s8*s9
            jac[i*3+1][i*3+14] = -2*s10**2 - 2*s8**2 + 1
            jac[i*3+1][i*3+15] = 2*s10*s9 - 2*s7*s8

            jac[i*3+2][2] = 1
            jac[i*3+2][6] = -2*s14*s9 + 2*s15*s8
            jac[i*3+2][7] = 2*s10*s14 + 2*s15*s7 - 4*s16*s8
            jac[i*3+2][8] = 2*s10*s15 - 2*s14*s7 - 4*s16*s9
            jac[i*3+2][9] = 2*s14*s8 + 2*s15*s9
            jac[i*3+2][i*3+13] = 2*s10*s8 - 2*s7*s9
            jac[i*3+2][i*3+14] = 2*s10*s9 + 2*s7*s8
            jac[i*3+2][i*3+15] = -2*s8**2 - 2*s9**2 + 1


        return jac




if __name__ == '__main__':
    # 小正方体特征点索引
    indexs = [0, 1, 3, 2]
    cube_points = [(-100, 100, -100), (-100, -100, -100), (100, 100, -100), (100, -100, -100), (100, 100, 100), (100, -100, 100), (-100, 100, 100), (-100, -100, 100)]
    # 测量值列表
    measured_points = []
    quaternions = []
    for i in range(200):
        temp = []
        for index in indexs:
            # 设计相机继续绕自身x轴负方向旋转，并且不断向世界坐标系下z轴负方向移动， 得到每一帧相机坐标系下的坐标
            pos = getImagePosition3(cube_points[index][0], cube_points[index][1], cube_points[index][2], 0, 0, i, i*10, -i)
            # 假设通过传感器可以得到相邻两帧间的位姿关系，则可以通过计算得到每个点在第一帧相机下的位置
            pos_calculate = changeToFrame1(pos[0], pos[1], pos[2], i*10, -i)
            # 和真值进行比较，验证计算的正确性
            pos_real = getImagePosition2(cube_points[index][0], cube_points[index][1], cube_points[index][2], 0, 0, i)
            # print(pos_calculate)
            # print("---")
            # print(pos_real)

            temp.append([pos_calculate[0]])
            temp.append([pos_calculate[1]])
            temp.append([pos_calculate[2]])

        quaternions.append(getQuaternion2(-45, 0, -i))
        measured_points.append(temp)

    # 估计值列表
    kalman_points = []
    # 初值:小正方体坐标系原点在相机坐标系下的位置(y值要取反！！！！！！！！！！)
    s0 = [0., 0., 2121]
    # 初值:小正方体坐标系到相机坐标系坐标系旋转矩阵(四元数)
    q0 = [-0.924, -0.383, 0., 0.]
    # M个特征点在小行星坐标系下的坐标（不随时间变化,y值要取反！！！！！！！！！！)
    points = [[-100., 100., -100.], [-100., -100., -100.], [100., -100., -100.], [100., 100., -100.]]
    # 初值：小行星坐标系x,y,z方向的角速度初值
    w0 = [0., 0., 0.]

    # 初值：小行星坐标原点在相机坐标系下x,y,z方向速度初值
    v0 = [0., 0., 0.]

    # kalfilt = TrackerEKF(s0, q0, points, w0, v0, p=0.2,  q=0.002, r=0.000002)
    kalfilt = TrackerEKF(s0, q0, points, w0, v0, p=0.02, q=0.0002, r=0.000002)

    for i in range(1, 180):
        # 更新卡尔曼滤波器，获取预测值
        estimate = kalfilt.step(measured_points[i])

        # 四元数在update之后要归一化！！！！
        num = math.sqrt(kalfilt.x[6][0]**2 + kalfilt.x[7][0]**2 + kalfilt.x[8][0]**2 + kalfilt.x[9][0]**2)

        kalfilt.x[6][0] /= num
        kalfilt.x[7][0] /= num
        kalfilt.x[8][0] /= num
        kalfilt.x[9][0] /= num
        np.savetxt("result/kalman-"+str(i), kalfilt.x)
        kalman_points.append(kalfilt.x)


    draw(kalman_points, [0, 1, 2], ['X', 'Y', 'Z'])
    draw(kalman_points, [3, 4, 5], ['Vx', 'Vy', 'Vz'])
    draw(kalman_points, [6, 7, 8, 9], ['q0', 'q1', 'q2', 'q3'])
    draw(kalman_points, [10, 11, 12], ['Wx', 'Wy', 'Wz'])
    for i in range(M):
        draw(kalman_points, [13+i*3, 14+i*3, 15+i*3], ['X-'+str(i+1), 'Y-'+str(i+1), 'Z-'+str(i+1)])
    drawQuaternions2(kalman_points, [6, 7, 8, 9], ['q0', 'q1', 'q2', 'q3'], quaternions)




