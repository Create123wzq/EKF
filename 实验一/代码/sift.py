import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

MIN_MATCH_COUNT = 5  # RANSAC匹配条件

def getSift(img_path1):
    '''''
    得到并查看sift特征
    '''
    # 读取图像
    img = cv2.imread(img_path1)
    # 转换为灰度图
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 创建sift的类
    sift = cv2.xfeatures2d.SIFT_create()
    # 在图像中找到关键点 也可以一步计算#kp, des = sift.detectAndCompute
    kp = sift.detect(gray,None)
    print(type(kp),type(kp[0]))
    # Keypoint数据类型分析 http://www.cnblogs.com/cj695/p/4041399.html
    print(kp[0].pt)
    print(type(kp[0].pt))
    # 计算每个点的sift
    des = sift.compute(gray,kp)
    print(type(des))
    # # des[0]为关键点的list，des[1]为特征向量的矩阵
    print(type(des[0]), type(des[1]))
    print(des[1].shape)
    # # 在灰度图中画出这些点
    img=cv2.drawKeypoints(img, kp, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # cv2.imwrite('sift_keypoints.jpg',img)
    plt.imshow(img)
    plt.show()

# getSift('/Users/apple/Desktop/test/img4/img0000.jpg')

def match(path1, path2, index, ratio=0.4):
    print('match %d start..' % index)
    sift = cv2.xfeatures2d.SIFT_create()

    img1 = cv2.imread(path1)
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子

    img2 = cv2.imread(path2)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # hmerge = np.hstack((gray1, gray2))  # 水平拼接
    # cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    # cv2.imshow("gray", hmerge)  # 拼接显示为gray
    # cv2.waitKey(0)

    # img3 = cv2.drawKeypoints(img1, kp1, None, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # img4 = cv2.drawKeypoints(img2, kp2, None, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    #
    # hmerge = np.hstack((img3, img4))  # 水平拼接
    # cv2.namedWindow('point', cv2.WINDOW_NORMAL)
    # cv2.imshow("point", hmerge)  # 拼接显示为gray
    # cv2.waitKey(0)

    # FLANN匹配器
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print('Num of matches: %d' % len(matches))
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    print('Num of good matches: %d' % len(good))

    # img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None)
    # cv2.namedWindow('FLANN', cv2.WINDOW_NORMAL)
    # cv2.imshow("FLANN", img5)
    # cv2.waitKey(0)

    if len(good) > MIN_MATCH_COUNT:
        ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 其中H为求得的单应性矩阵矩阵
        # status则返回一个列表来表征匹配成功的特征点
        # ptsA,ptsB为关键点
        # cv2.RANSAC
        # ransacReprojThreshold 则表示一对内群点所能容忍的最大投影误差
        # Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC method only）
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
        matchesMask = status.ravel().tolist()
        maskLen = matchesMask.count(1)
        if maskLen > 5: # 特征点配对数阈值设定(视情况，这里直接用5点约束)
            good2 = []
            for j in range(len(matchesMask)):
                if matchesMask[j] == 1:
                    good2.append(good[j])

            print('Num of good matches after RANCAC: %d' % len(good2))

            img6 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good2, None)
            # cv2.namedWindow('FLANN-RANCAC', cv2.WINDOW_NORMAL)
            # cv2.imshow("FLANN-RANCAC", img6)
            # cv2.waitKey(0)
            cv2.imwrite('img/match-' + str(index) + '.jpg', img6)
    else:
        raise Exception("Not enough matches for RANSAC!")

    #cv2.destroyAllWindows()
    return kp1, kp2, good2

# match('/Users/apple/Desktop/test/img4/img0000.jpg', '/Users/apple/Desktop/test/img4/img0011.jpg')

def getPositions(prefix, number, ratio=0.4):
    result = {}
    kp1Pre, kp2Pre, matches = match(prefix+"img0000.jpg", prefix+"img0001.jpg", 1, ratio)
    for m in matches:
        qId = m[0].queryIdx
        tId = m[0].trainIdx
        result[tId] = [kp1Pre[qId].pt, kp2Pre[tId].pt]
    print("number [%d] result length: %d" % (1, len(result)))
    # 相隔两帧进行匹配
    for i in range(1, number-1):
        path1 = prefix + "img" + "0" * (4 - len(str(i))) + str(i) + ".jpg"
        path2 = prefix + "img" + "0" * (4 - len(str(i+1))) + str(i+1) + ".jpg"
        kp1, kp2, matches = match(path1, path2, i+1, ratio)
        temp = {}
        for m in matches:
            qId = m[0].queryIdx
            tId = m[0].trainIdx
            # 如果这一次的qId(特征点序号)存在于上一次的结果中，对结果集进行更新
            if qId in result:
                pointList = result[qId]
                pointList.append(kp2[tId].pt)
                temp[tId] = pointList
        result = temp
        print("number [%d] result length: %d" % (i+1, len(result)))
        # if(len(result) == 4):
        #     return result, i+2
        # if(i+1 == 23):
        #     return result, i+2
        if(len(result) == 6):
            return result, i+2

    return result, number


# print(getPositions("/Users/apple/Desktop/test/img4/", 50, ratio=0.7))





