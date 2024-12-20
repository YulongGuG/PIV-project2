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

def GetPC(dpt, P):
    PC          = np.zeros((dpt.size, 3))
    i, j        = np.meshgrid(np.arange(dpt.shape[0]), np.arange(dpt.shape[1]), indexing='ij')
    pixels      = np.stack((i.ravel(), j.ravel(), np.ones(dpt.size)), axis=1)
    dpt_flat    = dpt.ravel()
    PC          = (P @ pixels.T).T * dpt_flat[:, np.newaxis]
    return PC

def RandomMatch(data, N, K):
    N       = data.shape[0]
    indices = np.random.choice(N, size=(K, N), replace=False)
    result  = data[indices]
    return result

def Transform(P, H):
    Q = H @ P
    return Q
    
def Inliers(P, Q, Th):
    D       = P - Q
    d       = np.linalg.norm(D, axis=1)
    inliers = (d < Th).astype(int)
    num     = np.sum(inliers)
    return num

# FROM P TO Q
# qi = R @ pi + T
def ICP(P, Q):
    MeanP   = np.mean(P, axis = 0)
    MeanQ   = np.mean(Q, axis = 0)
    Pl      = P - MeanP
    Ql      = Q - MeanQ
    W       = Ql.T @ Pl
    U, S, V = np.linalg.svd(W)
    R       = U @ V
    T       = MeanQ - R @ MeanP
    return R, T


def Homo(match):
    n = match.shape[0]
    A = np.zeros((2*n, 9))
    My1 = match[:, 0]
    Mx1 = match[:, 1]
    My2 = match[:, 2]
    Mx2 = match[:, 3]
    # [x1,    y1,      1,     0,     0,     0, -x2x1, -x2y1,   -x2]
    A[0::2, 0:3] = np.c_[Mx1, My1, np.ones(n)]
    A[0::2, 6:9] = -Mx2[:, np.newaxis] * np.c_[Mx1, My1, np.ones(n)]
    # [0,      0,      0,    x1,    y1,     1, -y2x1, -y2y1,   -y2]
    A[1::2, 3:6] = np.c_[Mx1, My1, np.ones(n)]
    A[1::2, 6:9] = -My2[:, np.newaxis] * np.c_[Mx1, My1, np.ones(n)]

    U, S, Vh = np.linalg.svd(A)
    X = Vh[-1]
    H = X.reshape(3,3)
    H = H / H[-1, -1]
    return H

# N  -> num of matches to run Homo
# K  -> num of iterations of RANSAC
# Th -> threshold to identifie inliers
def RANSAC(data, N = 4, K = 600, Th = 0.1):
    # ----- choise random matchs
    RanM        = RandomMatch(data, N, K)
    H           = [ICP(RanM[i, :, 0:3], RanM[i, :, 3:6]) for i in range(K)]
    Tdata       = [Transform(data[:, 0:3], H[i]) for i in range(K)]
    inliers_num = [Inliers(Tdata[i, :, :], data[:, 3:6], Th) for i in range(K)]
    index       = np.argmax(inliers_num)
    R           = R_M[index]
    T           = T_vec[index]
    return R, T



