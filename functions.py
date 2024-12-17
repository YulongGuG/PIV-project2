import numpy as np

def GetIntrinsic(fl, cp):
    K = np.array([[fl, 0 , cp[0]],
                  [0 , fl, cp[1]],
                  [0 , 0 ,    1 ]])
    return K

def GetExtrinsicZero():
    E = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]])
    return E

def GetCameraMatrix(fl, cp):
    K = GetIntrinsic(fl, cp)
    E = GetExtrinsicZero()
    P = K @ E
    return P

def GetPC(depth, P):
    PC = np.zeros((depth.size, 3))

    i, j = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
    pixels = np.stack((i.ravel(), j.ravel(), np.ones(depth.size)), axis=1)

    depth_flat = depth.ravel()
    PC = (P @ pixels.T).T * depth_flat[:, np.newaxis]
    return PC

def ICP(PC1, PC2):

    return R, T

