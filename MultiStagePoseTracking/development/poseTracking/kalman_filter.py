import copy

import numpy as np

class KMfilter():
    def __init__(self):
        self.step = 8./250.
        self.A = np.array([[1, self.step, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, self.step],[ 0, 0 ,0, 1]])
        self.B = 0
        self.H = np.array([[1, 0, 0, 0],[ 0, 0, 1, 0]])
        self.c_w = np.array([0.5]).reshape(1,1)

    # 1. State prediction
    # xhat[t|t − 1] = A xhat[t−1|t−1]
    def xhat(self, xprior):
        newxhat = self.A @ xprior
        # print("newxhat ", newxhat)
        return newxhat

    # 2. MSE Prediction:
    # M[t|t−1] = A M[t−1|t−1]A^T + BCqB^T
    def MSEhat(self, mseprior):
        msepost = self.A @ mseprior @ self.A.T + .05
        # print("msepost ", msepost)
        return msepost

    # 3. Kalman Gain Computation:
    # K[t] = M [t|t − 1]H^T [t] (Cw[t] + H[t] M [t|t − 1]H^T [t])^−1
    # assuming H = [ 1 0 0 0 ; 0 0 1 0 ] (revisit if necessary)
    # assume cW[t] = 0.5
    def KGC(self, mseprior):
        component = self.c_w + self.H @ mseprior @ self.H.T
        rand_noise = 0.00001 * np.random.rand( np.shape(component)[0], np.shape(component)[1])
        self.k = mseprior @ self.H.T @ (np.linalg.inv(component + rand_noise))
        # print("KGC ", self.k)
        return self.k

    # 4. State Estimation (= Correction):
    # xhat[t|t] = xhat[t|t − 1] + K[t] (z[t] − H[t] xhat[t|t − 1])
    def xhat_estimate(self, xprior, k, measurement):
        # print(xprior, k, measurement)
        xhat_estimate = xprior + k @ ( measurement - self.H @ xprior )
        # print("state estimate ", xhat_estimate)
        return xhat_estimate

    # 5. MSE Estimation:
    # M[t|t] = (1 − K[t]) H[t] M[t|t − 1]
    def MSE_estimate(self, mseprior, k):
        MSE_estimate = (1 - k ) @ self.H @ mseprior
        return MSE_estimate