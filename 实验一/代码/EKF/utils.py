import matplotlib.pyplot as plt
import math
import numpy as np
import cv2


DELAY_MSEC = 1  # 卡尔曼滤波计算时间间隔,单位为s
F = 36 # 焦距 mm
CCD_SIZE = 11.264 # 传感器尺寸 mm
WIDTH = 2048
HEIGHT = 2048

M = 4 # 卡尔曼滤波器跟踪的点的个数

def fromPixelToImage(u, v):
    dx = CCD_SIZE/WIDTH
    dy = CCD_SIZE/HEIGHT
    x = dx*(u - WIDTH/2)
    y = dy*(v - HEIGHT/2)
    return x,y

def fromImageToPixel(x, y):
    dx = CCD_SIZE / WIDTH
    dy = CCD_SIZE / HEIGHT
    u = x/dx + WIDTH/2
    v = y/dy + HEIGHT/2
    return int(u), int(v)


def fromCameraToImage(x, y, z):
    x_new = (F * x) / z
    y_new = (F * y) / z
    return x_new, y_new

def drawQuaternions(kalman_points, indexList, labelList, quaternions):
    for i in range(len(indexList)):
        index = indexList[i]
        y = []
        for j in range(len(kalman_points)):
            y.append(abs(kalman_points[j][index][0]-quaternions[j][i]))
        plt.plot(y, label=labelList[i])

    plt.legend(loc='upper right')
    plt.title("quaternion error")
    plt.show()

def drawQuaternions2(kalman_points, indexList, labelList, quaternions):
    for i in range(len(indexList)):
        index = indexList[i]
        y = []
        for j in range(len(kalman_points)):
            y.append(abs(abs(kalman_points[j][index][0])-abs(quaternions[j][i])))
        plt.plot(y, label=labelList[i])

    plt.legend(loc='upper right')
    plt.title("quaternion error")
    plt.show()


def draw(kalman_points, indexList, labelList):
    for i in range(len(indexList)):
        index = indexList[i]
        y = []
        for j in range(len(kalman_points)):
            y.append(kalman_points[j][index][0])
        plt.plot(y, label=labelList[i])

    plt.legend(loc='upper right')
    plt.show()

def drawPoint(img, points_list):
    point_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 165, 0)]
    point_size = 6
    thickness = 4  # 可以为 0 、4、8

    for i,point in enumerate(points_list):
        cv2.circle(img, point, point_size, point_colors[i], thickness)


def getImagePosition(x, y, z, p, h, b, v):

    h = math.radians(h)
    p = math.radians(p)
    b = math.radians(b)
    # 这是坐标的旋转矩阵，得到的是旋转后的坐标在原始坐标系下的新坐标
    Rx  = np.array([[1, 0, 0],
                    [0, math.cos(p), -math.sin(p)],
                    [0, math.sin(p), math.cos(p)]])


    Ry = np.array([[math.cos(h), 0, math.sin(h)],
                   [0, 1, 0],
                   [-math.sin(h), 0, math.cos(h)]])

    Rz = np.array([[math.cos(b), -math.sin(b), 0],
                   [math.sin(b), math.cos(b), 0],
                   [0, 0, 1]])

    point = np.array([[x], [y], [z]])
    rM = (Rx.dot(Ry)).dot(Rz)
    # p1 = (np.linalg.inv(rM)).dot(point)
    p1 = rM.dot(point)
    return p1[0][0]+v*0.9, p1[1][0]+v*0.8, p1[2][0]+2000+v

def getImagePosition2(x, y, z, p, h, b):

    h = math.radians(h)
    p = math.radians(p)
    b = math.radians(b)
    # 这是坐标的旋转矩阵，得到的是旋转后的坐标在原始坐标系下的新坐标
    Rx  = np.array([[1, 0, 0],
                    [0, math.cos(p), -math.sin(p)],
                    [0, math.sin(p), math.cos(p)]])


    Ry = np.array([[math.cos(h), 0, math.sin(h)],
                   [0, 1, 0],
                   [-math.sin(h), 0, math.cos(h)]])

    Rz = np.array([[math.cos(b), -math.sin(b), 0],
                   [math.sin(b), math.cos(b), 0],
                   [0, 0, 1]])

    point = np.array([[x], [y], [z]])
    rM = (Rx.dot(Ry)).dot(Rz)
    # p1 = (np.linalg.inv(rM)).dot(point)
    # 得到点的世界坐标
    p1 = rM.dot(point)

    p1[1][0] += 1500
    p1[2][0] += 1500

    h = math.radians(0)
    p = math.radians(-45)
    b = math.radians(0)

    # 这是坐标系的旋转矩阵，得到的是不动的点在动了的坐标系下的表示
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(p), math.sin(p)],
                   [0, -math.sin(p), math.cos(p)]])

    Ry = np.array([[math.cos(h), 0, -math.sin(h)],
                   [0, 1, 0],
                   [math.sin(h), 0, math.cos(h)]])

    Rz = np.array([[math.cos(b), math.sin(b), 0],
                   [-math.sin(b), math.cos(b), 0],
                   [0, 0, 1]])

    rM = (Rx.dot(Ry)).dot(Rz)

    p2 = rM.dot(p1)
    return p2[0][0], p2[1][0], p2[2][0]

def getImagePosition3(x, y, z, p, h, b, v, r):

    h = math.radians(h)
    p = math.radians(p)
    b = math.radians(b)
    # 这是坐标的旋转矩阵，得到的是旋转后的坐标在原始坐标系下的新坐标
    Rx  = np.array([[1, 0, 0],
                    [0, math.cos(p), -math.sin(p)],
                    [0, math.sin(p), math.cos(p)]])


    Ry = np.array([[math.cos(h), 0, math.sin(h)],
                   [0, 1, 0],
                   [-math.sin(h), 0, math.cos(h)]])

    Rz = np.array([[math.cos(b), -math.sin(b), 0],
                   [math.sin(b), math.cos(b), 0],
                   [0, 0, 1]])

    point = np.array([[x], [y], [z]])
    rM = (Rx.dot(Ry)).dot(Rz)
    # p1 = (np.linalg.inv(rM)).dot(point)
    # 得到点的世界坐标
    p1 = rM.dot(point)

    p1[1][0] += 1500
    p1[2][0] += 1500 + v

    h = math.radians(0)
    p = math.radians(-45+r)
    b = math.radians(0)

    # 这是坐标系的旋转矩阵，得到的是不动的点在动了的坐标系下的表示
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(p), math.sin(p)],
                   [0, -math.sin(p), math.cos(p)]])

    Ry = np.array([[math.cos(h), 0, -math.sin(h)],
                   [0, 1, 0],
                   [math.sin(h), 0, math.cos(h)]])

    Rz = np.array([[math.cos(b), math.sin(b), 0],
                   [-math.sin(b), math.cos(b), 0],
                   [0, 0, 1]])

    rM = (Rx.dot(Ry)).dot(Rz)

    p2 = rM.dot(p1)
    return p2[0][0], p2[1][0], p2[2][0]

def changeToFrame1(x, y, z, v, r):
    h = math.radians(0)
    p = math.radians(-r)
    b = math.radians(0)
    # 这是坐标系的旋转矩阵，得到的是不动的点在动了的坐标系下的表示
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(p), math.sin(p)],
                   [0, -math.sin(p), math.cos(p)]])

    Ry = np.array([[math.cos(h), 0, -math.sin(h)],
                   [0, 1, 0],
                   [math.sin(h), 0, math.cos(h)]])

    Rz = np.array([[math.cos(b), math.sin(b), 0],
                   [-math.sin(b), math.cos(b), 0],
                   [0, 0, 1]])

    point = np.array([[x], [y], [z]])
    rM = (Rx.dot(Ry)).dot(Rz)

    p1 = rM.dot(point)

    return p1[0][0], p1[1][0]+v*1.414/2, p1[2][0]-v*1.414/2

def getQuaternion(x, y, z):
    x = math.radians(x)
    y = math.radians(y)
    z = math.radians(z)
    c1 = math.cos(x/2)
    c2 = math.cos(y/2)
    c3 = math.cos(z/2)
    s1 = math.sin(x/2)
    s2 = math.sin(y/2)
    s3 = math.sin(z/2)

    result = []
    result.append(s1*c2*c3-c1*s2*s3)
    result.append(c1*s2*c3+s1*c2*s3)
    result.append((c1*c2*s3-s1*s2*c3))
    result.append(c1*c2*c3+s1*s2*s3)

    return result

def getQuaternion2(x, y, z):
    x = math.radians(x)
    y = math.radians(y)
    z = math.radians(z)
    c1 = math.cos(x / 2)
    c2 = math.cos(y / 2)
    c3 = math.cos(z / 2)
    s1 = math.sin(x / 2)
    s2 = math.sin(y / 2)
    s3 = math.sin(z / 2)

    result = []
    result.append(c1 * c2 * c3 + s1 * s2 * s3)
    result.append(s1 * c2 * c3 - c1 * s2 * s3)
    result.append(c1 * s2 * c3 + s1 * c2 * s3)
    result.append((c1 * c2 * s3 - s1 * s2 * c3))

    return result

# indexs = [0, 1, 3, 2]
# cube_points = [(-100, 100, -100), (-100, -100, -100), (100, 100, -100), (100, -100, -100), (100, 100, 100), (100, -100, 100), (-100, 100, 100), (-100, -100, 100)]
# print(getImagePosition2(cube_points[0][0], cube_points[0][1], cube_points[0][2], 0, 0, 80))
# print(getQuaternion(-45, 0, -80))
# print(getImagePosition2(0, 0, 0, 0, 0, 0))
# print(getQuaternion(-45, 0, 0))
# print(4.587054710452746786e-04 * 2179)
# print(np.random.randn()*.01)
# print(5000%360)
# print(math.degrees(-3.137499))
# print(math.degrees(-3.128717))