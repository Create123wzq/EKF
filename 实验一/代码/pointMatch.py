"""
LMAP跟踪，点跟踪匹配寻找Landmark出现位置
区域匹配-->单应性矩阵-->变换跟踪点（可能不是特征点）
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob as gb
import os

MIN_MATCH_COUNT = 5  # RANSAC匹配条件
ratio = 0.65
IMG_PATH = r'info\s1-0000'
Lpath = IMG_PATH + r'\L1.tif'  # Windows系统不区分大小写

IMG = gb.glob(r'C:\Users\shi\Desktop\67pModel\1005\1k\*.tif')
trainImg = cv2.cvtColor(cv2.imread(Lpath), cv2.COLOR_BGR2RGB)   # 颜色通道
lmp = IMG_PATH + r'/lmp%s' % Lpath[14:-4]
if not os.path.exists(lmp):
    os.mkdir(lmp)

img_path = IMG.copy()
first = img_path[0]
del img_path[0]
# img_path.remove(r'C:\Users\shi\Desktop\67pModel\1005\1k\s1-0000.tif')

flag = True     # 判断cropImg是否存在，即是否有符合约束的匹配
i = 0
it = 0          # 整个数据集迭代次数2，考虑（近）圆形的拍摄轨迹
LH = np.empty((0, 3), np.str)   # 存储Landmark信息

# SIFT特征提取
img0 = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img0, None)

kp0 = kp1.copy()
des0 = des1.copy()
# print(cv2.KeyPoint_convert(kp1))
# img3 = cv2.drawKeypoints(cv2.cvtColor(trainImg, cv2.COLOR_RGB2BGR), kp1, np.array([]), color=(0, 255, 0))
# cv2.imshow('feature', img3)
# cv2.waitKey(0)

while flag or it < 2:  # it
    print('iteration:', it+1)
    if (not flag) and (it < 2):
        trainImg = cv2.cvtColor(cv2.imread(Lpath), cv2.COLOR_BGR2RGB)
        kp1 = kp0
        des1 = des0
        flag = True
    best = 0
    besti = 0
    cropImg = None
    tmp = None
    kp22 = []
    des22 = np.empty((0, 2))

    while i < len(img_path):  # 遍历图像
        print('[#----   %s %s Run in image %s   ---#]' % (IMG_PATH, Lpath[13:-4], img_path[i][-11:]))
        queryImg = cv2.cvtColor(cv2.imread(img_path[i]), cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)

        kp2, des2 = sift.detectAndCompute(img1, None)
        # FLANN匹配器
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        print('Num of good matches %d' % len(good))

        # RANSAC去噪
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # M为求得的单应性矩阵，mask返回列表来表示匹配成功的特征点
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            maskLen = matchesMask.count(1)

            print('RANSAC-[' + str(maskLen) + ']-' + Lpath[13:-4] + '_' + img_path[i][-11:])
            if maskLen > 5:    # 0.5*first_best特征点配对数阈值设定(视情况，这里直接用5点约束)
                if best < maskLen:
                    best = maskLen
                    h0, w0 = img0.shape
                    pts = np.float32([[0, 0], [0, h0 - 1], [w0 - 1, h0 - 1], [w0 - 1, 0]]).reshape(-1, 1, 2)
                    dst = np.int32(cv2.perspectiveTransform(pts, M))
                    center = np.float32([[h0 / 2, w0 / 2]]).reshape(-1, 1, 2)
                    center = np.int32(cv2.perspectiveTransform(center, M))
                    center_u = center[0][0][0]
                    center_v = center[0][0][1]

                    # x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
                    # x, y, w, h = cv2.boundingRect(dst)
                    # tmp = ['%s_IN_%s' % (Lpath[13:-4], img_path[i][-11:-4]), x, y, w, h]
                    # crop = queryImg.copy()
                    # cropImg = crop[y:y + h, x:x + w]
                    tmp = ['%s_IN_%s' % (Lpath[13:-4], img_path[i][-11:-4]), center_u, center_v]
                    crop = queryImg.copy()
                    cropImg = crop[center_v-h0//2:center_v+h0//2, center_u-w0//2:center_u + w0//2]
                    queryImg = cv2.polylines(queryImg, [dst], True, (0, 0, 255), 5, cv2.LINE_AA)
                    queryImg = cv2.circle(queryImg, (center_u, center_v), 20, (255, 255, 0), 5)
                    queryImg = cv2.rectangle(queryImg, (center_u - w0 // 2, center_v - h0 // 2),
                                             (center_u + w0 // 2, center_v + h0 // 2), (255, 255, 255), 4)
                    # 边界矩形框
                    # queryImg = cv2.rectangle(queryImg, (x, y), (x + w, y + h), (255, 255, 255), 4)
                    draw_params = dict(matchColor=(0, 255, 0),
                                       singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=2)
                    img2 = cv2.drawMatches(trainImg, kp1, queryImg, kp2, good, None, **draw_params)
                    plt.figure('%s in %s' % (Lpath[13:-4], img_path[i][-11:-4]))
                    plt.imshow(img2, 'gray'), plt.savefig(r'info\tmp\%s_in_%s.tif' % (Lpath[13:-4], img_path[i][-11:-4]))
                    # plt.show()

                    kp22 = []
                    des22 = []
                    for j in range(len(matchesMask)):
                        if matchesMask[j] == 1:
                            saveIdx = good[j].trainIdx
                            kp22.append(kp2[saveIdx])
                            des22.append(des2[saveIdx])
                    des22 = np.array(des22)

                    for j in range(len(kp22)):
                        # kp22[j].pt = (kp22[j].pt[0] - x, kp22[j].pt[1] - y)
                        kp22[j].pt = (kp22[j].pt[0] - (center_u-w0//2), kp22[j].pt[1] - (center_v-h0//2))
                    besti = i
            else:
                print('Not enough matches for RANSAC!')
        else:
            print("Not enough matches are found! - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
        i += 1

    i = 0
    kp1, des1 = kp22, des22
    if cropImg is None:
        it += 1
        flag = False
        continue
    else:
        cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
        cv2.imwrite(IMG_PATH + r'/lmp%s/%s_IN_%s.tif' % (Lpath[14:-4], Lpath[13:-4], img_path[besti][-11:-4]), cropImg)
        cropImg = cv2.cvtColor(cropImg, cv2.COLOR_RGB2BGR)
        trainImg = cropImg
        LH = np.append(LH, [tmp], axis=0)
        del img_path[besti]

LH.sort(axis=0)
# 每张图像中每个LMAP信息存储
np.savetxt(IMG_PATH + r'/lmp%s/%s.txt' % (Lpath[14:-4], Lpath[13:-4]), LH, fmt='%s')
