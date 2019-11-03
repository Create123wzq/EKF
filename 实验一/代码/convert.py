import cv2
import numpy as np

import math
# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(angles1) :
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0]*3.141592653589793/180.0
    theta[1] = angles1[1]*3.141592653589793/180.0
    theta[2] = angles1[2]*3.141592653589793/180.0
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_x, np.dot( R_y, R_z ))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    #print('dst:', R)
    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793
    rvecs = np.zeros((1, 1, 3), dtype=np.float64)
    rvecs,_ = cv2.Rodrigues(R, rvecs)
    #print()
    return R,rvecs,x,y,z

def rotationMatrixToEulerAngles(rvecs):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvecs, R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    #print('dst:', R)
    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793
    return x,y,z

def getVecFromQuaternions(q1, q2, q3, q4):
    R = np.array([[q1*q1-q2*q2-q3*q3+q4*q4, 2*(q1*q2+q3*q4), 2*(q1*q3-q2*q4)],
                  [2*(q1*q2-q3*q4), -q1*q1+q2*q2-q3*q3+q4*q4, 2*(q2*q3+q1*q4)],
                  [2*(q1*q3+q2*q4), 2*(q2*q3-q1*q4), -q1*q1-q2*q2+q3*q3+q4*q4]
                  ])

    return cv2.Rodrigues(R)[0]

if(__name__=='__main__'):
#     eulerAngles = np.zeros((3, 1), dtype=np.float64)
#     eulerAngles[0] = 30.0
#     eulerAngles[1] = 20.0
#     eulerAngles[2] = 10.0
#     R,rvecstmp,x,y,z = eulerAnglesToRotationMatrix(eulerAngles)
    print(getVecFromQuaternions(-1.372106619230289547e-03, 3.881782824369618967e-01, -9.214745608907018992e-01, 1.415528166458141987e-02))
#     print(rvecs)
#     print(rotationMatrixToEulerAngles(rvecs))