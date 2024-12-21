import numpy as np
import math
import itertools

def GetIntrinsic(fl, cp):
    K               = np.array([[fl, 0 , cp[0]],
                                [0 , fl, cp[1]],
                                [0 , 0 ,    1 ]])
    return K

def GetExtrinsicZero():
    E               = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]])
    return E

def GetCameraMatrix(fl, cp):
    K               = GetIntrinsic(fl, cp)
    E               = GetExtrinsicZero()
    P               = K @ E
    return P

def GetPC(dpt, P):
    PC              = np.zeros((dpt.size, 3))
    i, j            = np.meshgrid(np.arange(dpt.shape[0]), np.arange(dpt.shape[1]), indexing='ij')
    pixels          = np.stack((i.ravel(), j.ravel(), np.ones(dpt.size)), axis=1)
    dpt_flat        = dpt.ravel()
    PC              = (P @ pixels.T).T * dpt_flat[:, np.newaxis]
    return PC

# FROM P TO Q
# qi = R @ pi + T
def ICP(P, Q):
    MeanP           = np.mean(P, axis = 0)
    MeanQ           = np.mean(Q, axis = 0)
    Pl              = P - MeanP
    Ql              = Q - MeanQ
    W               = Ql.T @ Pl
    U, S, V         = np.linalg.svd(W)
    R               = U @ V
    T               = MeanQ - R @ MeanP
    return R, T

def KpMatch(D1, D2, th=0.75):
    matchs          = []
    for i, d1 in enumerate(D1):
        dist        = np.linalg.norm(D2 - d1, axis=1)
        sorted      = np.argsort(dist)
        best1       = sorted[0]
        best2       = sorted[1]
        if dist[best1] < th * dist[best2]:
            matchs.append((i, int(best1)))
    Res             = np.array(matchs)
    return Res

def zipKp(Kp1, Kp2, match):
    kps1            = Kp1[match[:, 0]]
    kps2            = Kp2[match[:, 1]]
    Res             = np.hstack((kps1, kps2))
    return Res

def RandomSel(M, N, K):
    AllComb         = list(itertools.combinations(range(M+1), N))
    
    SelComb         = np.random.choice(len(AllComb), size=K, replace=False)
    result          = [AllComb[i] for i in SelComb]
    return result

def Transform(P, H):
    ones            = np.ones((P.shape[0], 1))
    P_h             = np.hstack((P, ones))
    Q               = (H @ P_h.T).T
    Q               = Q / Q[:, -1].reshape(-1, 1) 
    return Q
    
def Inliers(P, Q, Th):
    D               = P - Q
    d               = np.linalg.norm(D, axis=1)
    inlier_mask     = (d < Th)
    inlier_indices  = np.where(inlier_mask)[0]
    return inlier_indices

def Homo(index, kp):
    #----------------------------------------------------
    # index -> nx1
    # kp    -> Kxn  [x11, y11, x21, y21]
    #                   ... ... ...
    #               [x1k, y1k, x2k, y2k]
    # [x2, y2, 1] = H @ [x1, y1, 1]
    #----------------------------------------------------
    n               = index.shape[0]
    A               = np.zeros((2*n, 9))
    My1             = kp[index, 0]
    Mx1             = kp[index, 1]
    My2             = kp[index, 2]
    Mx2             = kp[index, 3]
    # [x1,    y1,      1,     0,     0,     0, -x2x1, -x2y1,   -x2]
    A[0::2, 0:3]    = np.c_[Mx1, My1, np.ones(n)]
    A[0::2, 6:9]    = -Mx2[:, np.newaxis] * np.c_[Mx1, My1, np.ones(n)]
    # [0,      0,      0,    x1,    y1,     1, -y2x1, -y2y1,   -y2]
    A[1::2, 3:6]    = np.c_[Mx1, My1, np.ones(n)]
    A[1::2, 6:9]    = -My2[:, np.newaxis] * np.c_[Mx1, My1, np.ones(n)]
    U, S, Vh        = np.linalg.svd(A)
    X               = Vh[-1]
    H               = X.reshape(3,3)
    H               = H / H[-1, -1]
    return H

# N  -> num of matches to run Homo
# K  -> num of iterations of RANSAC
# Th -> threshold to identifie inliers
def RANSAC(Kps, N = 4, K = 600, Th = 5):
    M = Kps.shape[0]
    Comb            = math.comb(M, N)
    if Comb < K:
        K = Comb
    # ----- choise random matchs
    RanIndex        = RandomSel(M, N, K)
    H               = [Homo(RanIndex[i], Kps) for i in range(K)]
    Tdata           = [Transform(Kps[:, 0:2], H[i]) for i in range(K)]
    inliers         = [Inliers(Tdata[i, :], Kps[:, 2:4], Th) for i in range(K)]
    inliers_num     = [len(inliers[i]) for i in range(K)]
    index           = np.argmax(inliers_num)
    return inliers[index]



