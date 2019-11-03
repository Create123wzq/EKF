import numpy as np
import math
from EKF.utils import fromCameraToImage

def spaceResection(image_list, ground_list, phi=0.0, omega=0.0, kappa=0.0, Xs=0.0, Ys=0.0, Zs=0, f=36.0, max_step=60, threshold=1, threshold2=2e-7):
    num_ite = 0
    # 方程内的矩阵
    rotate = np.mat(np.zeros((3, 3)))  # 旋转矩阵
    A = np.mat(np.zeros((len(image_list) * 2, 6)))
    L = np.mat(np.zeros((len(image_list) * 2, 1)))
    # 存改正数的
    delta = [0] * 6
    # 存中误差
    m = [0] * 6

    # 开始处理
    while (True):

        # 旋转矩阵
        a1 = math.cos(phi) * math.cos(kappa) - math.sin(phi) * math.sin(omega) * math.sin(kappa)
        a2 = (-1.0) * math.cos(phi) * math.sin(kappa) - math.sin(phi) * math.sin(omega) * math.cos(kappa)
        a3 = (-1.0) * math.sin(phi) * math.cos(omega)
        b1 = math.cos(omega) * math.sin(kappa)
        b2 = math.cos(omega) * math.cos(kappa)
        b3 = (-1.0) * math.sin(omega)
        c1 = math.sin(phi) * math.cos(kappa) + math.cos(phi) * math.sin(omega) * math.sin(kappa)
        c2 = (-1.0) * math.sin(phi) * math.sin(kappa) + math.cos(phi) * math.sin(omega) * math.cos(kappa)
        c3 = math.cos(phi) * math.cos(omega)

        for i in range(0, len(image_list)):
            x = image_list[i][0]
            y = image_list[i][1]
            X = ground_list[i][0]
            Y = ground_list[i][1]
            Z = ground_list[i][2]

            Xp = a1 * (X - Xs) + b1 * (Y - Ys) + c1 * (Z - Zs)
            Yp = a2 * (X - Xs) + b2 * (Y - Ys) + c2 * (Z - Zs)
            Zp = a3 * (X - Xs) + b3 * (Y - Ys) + c3 * (Z - Zs)

            A[i * 2, 0] = 1.0 * (a1 * f + a3 * x) / Zp
            A[i * 2, 1] = 1.0 * (b1 * f + b3 * x) / Zp
            A[i * 2, 2] = 1.0 * (c1 * f + c3 * x) / Zp
            A[i * 2, 3] = y * math.sin(omega) - (x * (x * math.cos(kappa) - y * math.sin(kappa)) / f + f * math.cos(
                kappa)) * math.cos(omega)
            # A[i*2, 3] = b1*x*y/f - b2*(f+x**2/f) - b3*y
            A[i * 2, 4] = -f * math.sin(kappa) - x * (x * math.sin(kappa) + y * math.cos(kappa)) / f
            A[i * 2, 5] = y

            A[i * 2 + 1, 0] = 1.0 * (a2 * f + a3 * y) / Zp
            A[i * 2 + 1, 1] = 1.0 * (b2 * f + b3 * y) / Zp
            A[i * 2 + 1, 2] = 1.0 * (c2 * f + c3 * y) / Zp
            A[i * 2 + 1, 3] = -x * math.sin(omega) - (y * (
            x * math.cos(kappa) - y * math.sin(kappa)) / f - f * math.sin(kappa)) * math.cos(omega)
            # A[i*2+1, 3] = b1*(f+y**2/f) - b2*x*y/f + b3*x
            A[i * 2 + 1, 4] = -f * math.cos(kappa) - y * (x * math.sin(kappa) + y * math.cos(kappa)) / f
            A[i * 2 + 1, 5] = -x

            L[i * 2, 0] = x + f * Xp / Zp
            L[i * 2 + 1, 0] = y + f * Yp / Zp

        AT = A.T  # 转置矩阵
        ATA = AT * A  # 矩阵相乘
        ATAInv = ATA.I  # 求逆矩阵
        ATAAT = ATAInv * AT
        delta = ATAAT * L

        # 添加改正数
        Xs += delta[0]
        Ys += delta[1]
        Zs += delta[2]
        phi += delta[3]
        omega += delta[4]
        kappa += delta[5]

        num_ite += 1

        # 限差(看具体情况设置限差)
        if (math.fabs(delta[0]) < threshold) and (math.fabs(delta[1]) < threshold) and (math.fabs(delta[2]) < threshold) \
                and (math.fabs(delta[3]) < threshold2) and (math.fabs(delta[4]) < threshold2) \
                    and (math.fabs(delta[5]) < threshold2):
            break
        if num_ite > max_step:
            print("Error:overtime")
            break

    '''
    后续处理，各类残差、中误差
    '''

    print("迭代次数:%f\n" % num_ite)
    rotate[0, 0] = math.cos(phi) * math.cos(kappa) - math.sin(phi) * math.sin(omega) * math.sin(kappa)
    rotate[0, 1] = (-1.0) * math.cos(phi) * math.sin(kappa) - math.sin(phi) * math.sin(omega) * math.cos(kappa)
    rotate[0, 2] = (-1.0) * math.sin(phi) * math.cos(omega)
    rotate[1, 0] = math.cos(omega) * math.sin(kappa)
    rotate[1, 1] = math.cos(omega) * math.cos(kappa)
    rotate[1, 2] = (-1.0) * math.sin(omega)
    rotate[2, 0] = math.sin(phi) * math.cos(kappa) + math.cos(phi) * math.sin(omega) * math.sin(kappa)
    rotate[2, 1] = (-1.0) * math.sin(phi) * math.sin(kappa) + math.cos(phi) * math.sin(omega) * math.cos(kappa)
    rotate[2, 2] = math.cos(phi) * math.cos(omega)

    AX = A * delta
    # 残差
    v = AX - L

    vv = 0
    for it in v:
        vv += it[0, 0] * it[0, 0]
    m0 = math.sqrt(vv / (2 * len(image_list) - 6))

    print('---------------------')
    # 存残差的
    print_v = np.zeros((len(image_list), 2))
    for i in range(0, len(image_list)):
        print_v[i][0] = v[2 * i][0]
        print_v[i][1] = v[2 * i + 1][0]
    # print("%d   %f   %f"%(Pxy[i][0], v[2*i][0], v[2*i+1][0]))
    # 可以排个序来挑出较大残差，这里没有
    for i in range(0, len(image_list)):
        print("vx:%f  vy:%f\n" % (print_v[i][0], print_v[i][1]))
    print('---------------------')

    print("单位权中误差m0=%f\n" % m0)
    # print(ATAInv)
    for n in range(6):
        m[n] = m0 * math.sqrt(abs(ATAInv[n, n]))

    print("\n结果：\n")
    print("Xs=%f  m=%f" % (Xs, m[0]) + "  单位：m\n")
    print("Ys=%f  m=%f" % (Ys, m[1]) + "  单位：m\n")
    print("Zs=%f   m=%f" % (Zs, m[2]) + "  单位：m\n")
    print("omega=%f    m=%f" % (omega, m[4]) + "  单位：弧度\n")
    print("phi=%f  m=%f" % (phi, m[3]) + "  单位：弧度\n")
    print("kappa=%f    m=%f" % (kappa, m[5]) + "  单位：弧度\n")
    print("omega=%f    m=%f" % (math.degrees(omega), m[4]) + "  单位：度\n")
    print("phi=%f  m=%f" % (math.degrees(phi), m[3]) + "  单位：度\n")
    print("kappa=%f    m=%f" % (math.degrees(kappa), m[5]) + "  单位：度\n")
    print("\n旋转矩阵R为：\n")
    print(rotate)

    # 写文件
    with open('后方交会结果.txt', 'w') as fo:
        fo.writelines("\n像点残差：\n")
        for i in range(0, len(image_list)):
            fo.writelines("vx:%f  vy:%f\n" % (print_v[i][0], print_v[i][1]))
        fo.writelines("   \n")
        fo.writelines("单位权中误差m0=%f\n" % m0)
        fo.writelines("          \n")
        fo.writelines("Xs=%f  m=%f" % (Xs, m[0]) + "  单位：m\n")
        fo.writelines("Ys=%f  m=%f" % (Ys, m[1]) + "  单位：m\n")
        fo.writelines("Zs=%f   m=%f" % (Zs, m[2]) + "  单位：m\n")
        fo.writelines("omega=%f    m=%f" % (omega, m[4]) + "  单位：弧度\n")
        fo.writelines("phi=%f  m=%f" % (phi, m[3]) + "  单位：弧度\n")
        fo.writelines("kappa=%f    m=%f" % (kappa, m[5]) + "  单位：弧度\n")
        fo.writelines("omega=%f    m=%f" % (math.degrees(omega), m[4]) + "  单位：度\n")
        fo.writelines("phi=%f  m=%f" % (math.degrees(phi), m[3]) + "  单位：度\n")
        fo.writelines("kappa=%f    m=%f" % (math.degrees(kappa), m[5]) + "  单位：度\n")


def getImagePosition(x, y, z, p, h, b, pc, hc, bc, xc, yc, zc):

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
    rM = (Rz.dot(Rx)).dot(Ry)
    # p1 = (np.linalg.inv(rM)).dot(point)
    # 得到点的世界坐标
    p1 = rM.dot(point)
    p1[0][0] -= xc
    p1[1][0] -= yc
    p1[2][0] -= zc

    h = math.radians(hc)
    p = math.radians(pc)
    b = math.radians(bc)

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

    rM = (Rz.dot(Rx)).dot(Ry)

    p2 = rM.dot(p1)
    return p2[0][0], p2[1][0], p2[2][0]


if __name__ == '__main__':
    # 小正方体特征点索引
    indexs = [0, 1, 3, 2]
    cube_points = [(-100, 100, -100), (-100, -100, -100), (100, 100, -100), (100, -100, -100), (100, 100, 100), (100, -100, 100), (-100, 100, 100), (-100, -100, 100)]


    # f1 = 152.77
    # imageList = [[-109.515, 7.113], [-5.523, 95.813], [76.978, 48.100], [-51.693, -92.860], [22.200, -91.550], [95.177, -47.107]]
    # groudList = [[24308, 4582, 63],[26008, 5052, 57], [26722, 4055, 49], [24410, 3095, 52], [25292, 2708, 50], [26412, 2818, 47]]
    # 只有竖直摄影的时候可以将各个角度初值设为0，初始值尽量接近真实值，设定的值也很重要
    # spaceResection(image_list=imageList, ground_list=groudList, phi=0.002, omega=-0.01, kappa=-0.3, Xs=38520, Ys=26953, Zs=8929, f=f1)


    f2 = 153.24
    imageList = [[-86.15, -68.99], [-53.40, 82.21], [-14.78, -76.63], [10.46, 64.43]]
    groudList = [[36589.41, 25273.32, 2195.17], [37631.08, 31324.51, 728.69], [39100.97, 24934.98, 2386.50], [40426.54, 30319.81, 757.31]]
    # 竖直摄影
    spaceResection(image_list=imageList, ground_list=groudList, f=f2)

    # 仿真
    # imageList = []
    # groudList = []
    # for index in indexs:
    #     pos = getImagePosition(cube_points[index][0], cube_points[index][1], cube_points[index][2], 0, 0, 0, -20, -25, -100, 0, -1500, -1500)
    #     # 相机坐标系转换到图像坐标系
    #     x, y = fromCameraToImage(pos[0], pos[1], pos[2])
    #     imageList.append([x, y])
    #     groudList.append(cube_points[index])
    #spaceResection(image_list=imageList, ground_list=groudList, Xs=0.0, Ys=-1500, Zs=-1500, omega=math.radians(20), phi=math.radians(-25), kappa=math.radians(-90))


