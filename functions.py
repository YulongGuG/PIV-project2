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

def Transform(P, R, T):
    Q = R @ P + T
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

# N  -> num of matches to run ICP
# K  -> num of iterations of RANSAC
# Th -> threshold to identifie inliers
def RANSAC_with_ICP(data, N = 6, K = 600, Th = 0.1):
    # ----- choise random matchs
    RanM        = RandomMatch(data, N, K)
    # ----- runICP to get R and T
    res         = [ICP(RanM[i, :, 0:3], RanM[i, :, 3:6]) for i in range(K)]
    R_M, T_vec  = zip(*res)
    # ----- use R and T to obtain the num of inliers
    Tdata       = [Transform(data[:, 0:3], R_M[i, :, :], T_vec[i, :]) for i in range(K)]
    inliers_num = [Inliers(Tdata[i, :, :], data[:, 3:6], Th) for i in range(K)]
    # ----- compute the R and T with maximum inliers
    index       = np.argmax(inliers_num)
    R           = R_M[index]
    T           = T_vec[index]

    return R, T



