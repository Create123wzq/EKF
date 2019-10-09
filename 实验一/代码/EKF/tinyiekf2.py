'''
    Extended Kalman Filter in Python
'''
import numpy as np
from abc import ABCMeta, abstractmethod
import math


class IEKF2(object):
    __metaclass__ = ABCMeta

    def __init__(self, n, m, pval=0.1, qval=1e-4, rval=0.1, delta=0.0001):
        '''
        Creates a KF object with n states, m observables, and specified values for
        prediction noise covariance pval, process noise covariance qval, and
        measurement noise covariance rval.
        '''
        # No previous prediction noise covariance
        self.P_pre = None

        # Current state is zero, with diagonal noise covariance matrix
        self.x = np.zeros((n, 1),dtype=float)
        self.P_post = np.eye(n, dtype=float) * pval

        # Set up covariance matrices for process noise and measurement noise
        self.Q = np.eye(n, dtype=float) * qval
        self.R = np.eye(m, dtype=float) * rval

        # Identity matrix will be usefel later
        self.I = np.eye(n, dtype=float)

        # delta
        self.delta = delta

    def step(self, z):
        '''
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        '''
        # Predict ----------------------------------------------------
        self.F = self.getF(self.x)
        self.x = self.f(self.x)
        self.P_pre = np.dot(np.dot(self.F, self.P_post), self.F.T) + self.Q

        # Update -----------------------------------------------------
        x_temp = self.x.copy()
        x_pre = self.x.copy()
        self.H = self.getH(x_pre)
        G = np.dot(np.dot(self.P_pre, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P_pre), self.H.T) + self.R))


        x_post = x_pre + np.dot(G, (np.array(z) - self.h(x_pre)))
        num = math.sqrt(x_post[5][0]**2 + x_post[6][0]**2 + x_post[7][0]**2 + x_post[8][0]**2)

        x_post[5][0] /= num
        x_post[6][0] /= num
        x_post[7][0] /= num
        x_post[8][0] /= num

        count = 1
        while np.sqrt(np.sum(np.square(x_post-x_pre))) >= self.delta:
            print(count)
            x_pre = x_post
            self.H = self.getH(x_pre)

            G = np.dot(np.dot(self.P_pre, self.H.T),
                       np.linalg.inv(np.dot(np.dot(self.H, self.P_pre), self.H.T) + self.R))

            x_post = x_temp + np.dot(G, (np.array(z) - self.h(x_pre) - self.H.dot(x_temp - x_pre)))

            num = math.sqrt(x_post[5][0] ** 2 + x_post[6][0] ** 2 + x_post[7][0] ** 2 + x_post[8][0] ** 2)

            x_post[5][0] /= num
            x_post[6][0] /= num
            x_post[7][0] /= num
            x_post[8][0] /= num

            count += 1


        self.P_post = np.dot((self.I - np.dot(G, self.H)), self.P_pre)
        self.x = x_post.copy()
        # return self.x.asarray()
        return self.x

    @abstractmethod
    def f(self, x):
        '''
        Your implementing class should define this method for the state transition function f(x),
        returning a NumPy array of n elements.  Typically this is just the identity function np.copy(x).
        '''
        raise NotImplementedError()

    @abstractmethod
    def getF(self, x):
        '''
        Your implementing class should define this method for returning the n x n Jacobian matrix F of the
        state transition function as a NumPy array.  Typically this is just the identity matrix np.eye(n).
        '''
        raise NotImplementedError()

    @abstractmethod
    def h(self, x):
        '''
        Your implementing class should define this method for the observation function h(x), returning
        a NumPy array of m elements. For example, your function might include a component that
        turns barometric pressure into altitude in meters.
        '''
        raise NotImplementedError()

    @abstractmethod
    def getH(self, x):
        '''
        Your implementing class should define this method for returning the m x n Jacobian matirx H of the
        observation function as a NumPy array.
        '''
        raise NotImplementedError()