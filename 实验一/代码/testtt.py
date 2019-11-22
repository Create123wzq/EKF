import numpy as np
import math
from EKF.utils import getImagePosition2

def f1():
    p1 = np.array([[0], [0], [0]])
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

    print(p2)

def f2():
    p1 = np.array([[0], [0], [0]])

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
    p2[2][0] += 2121

    print(p2)

f1()
f2()