import numpy as np
import cv2
import sympy
import math
sympy.init_printing(use_latex=True)

# 利用sympy求雅克比矩阵
# s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 = sympy.symbols('s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15')
# H = sympy.Matrix([(s1+s13*(s6**2-s7**2-s8**2+s9**2)+2*s14*(s6*s7+s8*s9)+2*s15*(s6*s8-s7*s9))/
#                   (1+2*s13*(s6*s8+s7*s9)+2*s14*(s7*s8-s6*s9)+s15*(-s6**2-s7**2+s8**2+s9**2))])
# state = sympy.Matrix([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15])
# print(H.jacobian(state))

# H = sympy.Matrix([(s2+2*s13*(s6*s7-s8*s9)+s14*(-s6**2+s7**2-s8**2+s9**2)+2*s15*(s7*s8+s6*s9))/
#                   (1 + 2 * s13 * (s6 * s8 + s7 * s9) + 2 * s14 * (s7 * s8 - s6 * s9) + s15 * (
#                   -s6 ** 2 - s7 ** 2 + s8 ** 2 + s9 ** 2))])
# state = sympy.Matrix([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15])
# print(H.jacobian(state))

# 利用sympy求雅克比矩阵
# s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 = sympy.symbols('s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15')
# H = sympy.Matrix([(s1+s13*(s6**2-s7**2-s8**2+s9**2)+2*s14*(s6*s7-s8*s9)+2*s15*(s6*s8+s7*s9))/
#                   (1+2*s13*(s6*s8-s7*s9)+2*s14*(s7*s8+s6*s9)+s15*(-s6**2-s7**2+s8**2+s9**2))])
# state = sympy.Matrix([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15])
# print(H.jacobian(state))

# H = sympy.Matrix([(s2+2*s13*(s6*s7+s8*s9)+s14*(-s6**2+s7**2-s8**2+s9**2)+2*s15*(s7*s8-s6*s9))/
#                   (1 + 2 * s13 * (s6 * s8 - s7 * s9) + 2 * s14 * (s7 * s8 + s6 * s9) + s15 * (
#                   -s6 ** 2 - s7 ** 2 + s8 ** 2 + s9 ** 2))])
# state = sympy.Matrix([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15])
# print(H.jacobian(state))



# 在图上标注特征点位置
def drawPoint(img, points_list):
    point_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 165, 0)]
    point_size = 1
    thickness = 4  # 可以为 0 、4、8

    for i,point in enumerate(points_list):
        cv2.circle(img, point, point_size, point_colors[i], thickness)

# sift 特征提取（点并没有一直跟踪到）
# sift = cv2.xfeatures2d.SIFT_create()
#
# for i in range(31):
#     prefix = "image/"
#     if i < 10:
#         filename = "img000" + str(i) + ".jpg"
#     else:
#         filename = "img00" + str(i) + ".jpg"
#
#     img = cv2.imread(prefix + filename)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     kp,des = sift.detectAndCompute(img, None)
#
#     img1=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
#     cv2.imwrite("sift/"+filename, img1)

f = 36
q = [0.924,  0.000, -0.383,  0.000]
#matrix = Matrix(v1: (1, 0, 0); v2: (0, 0.707, 0.707); v3: (0, -0.707, 0.707); off: (0, 5000, -5000))
CCD_SIZE = 11.264 # 传感器尺寸 mm
WIDTH = 2048
HEIGHT = 2048

def fromCameraToPixel(points):
    result = []
    dx = CCD_SIZE / WIDTH
    dy = CCD_SIZE / HEIGHT
    for point in points:
        x = f * point[0] / point[2]
        y = f * point[1] / point[2]
        u = int(x/dx + WIDTH/2)
        v = int(y/dy + HEIGHT/2)
        result.append((u, v))

    return result

def fromPixelToImage(u, v):
    dx = CCD_SIZE/WIDTH
    dy = CCD_SIZE/HEIGHT
    x = dx*(u - WIDTH/2)
    y = dy*(v - HEIGHT/2)
    return x,y


# 经过测试，y要取负值！！！！！！！！！！！！！！
# points = [(18.884042, 11.206219, 6778.044135), (-15.845593, -128.087664, 6917.338018), (181.115958, -152.648934, 6941.899288), (215.845593, -13.355051, 6802.605405)]
# img=cv2.imread('image/img0001.jpg')
# cv2.namedWindow('input_image', cv2.WINDOW_NORMAL)
# drawPoint(img, fromCameraToPixel(points))
# cv2.imshow('input_image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 小正方体特征点索引
indexs = [1, 7, 5, 3]
# 测量值列表


def check():
    points = []
    for i in range(200):
        fileName = "/Users/apple/Desktop/test/Cube/pos"+str(i)+".txt"
        with open(fileName, "r") as file:
            temp = []
            lines = file.readlines()
            for index in indexs:
                line = lines[index].strip("\n").split()
                temp.append((float(line[0]), -float(line[1]), float(line[2])))

            points.append(temp)

    for i in range(20):
        prefix = "/Users/apple/Desktop/test/img/"
        filename = "img" + "0"*(4-len(str(i))) + str(i) + ".jpg"

        img = cv2.imread(prefix + filename)
        drawPoint(img, fromCameraToPixel(points[i]))
        cv2.imwrite("test/"+filename, img)


