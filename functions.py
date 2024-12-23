import numpy as np
import math
import itertools


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
    result          = [np.array(AllComb[i]) for i in SelComb]
    return np.array(result)

def Transform(P, H):
    ones            = np.ones((P.shape[0], 1))
    P_h             = np.hstack((P, ones))
    Q               = (H @ P_h.T).T
    Q               = Q / Q[:, -1].reshape(-1, 1) 
    return Q[:, 0:2]
    
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
    My1             = kp[index-1, 0]
    Mx1             = kp[index-1, 1]
    My2             = kp[index-1, 2]
    Mx2             = kp[index-1, 3]
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
def RANSAC(Kps, N = 4, K = 600, Th = 10):
    M = Kps.shape[0]
    if M < N:
        return np.array((0, 0))
    Comb            = math.comb(M, N)
    if Comb < K:
        K = Comb
    # ----- choise random matchs
    RanIndex        = RandomSel(M, N, K)
    H               = [Homo(RanIndex[i], Kps) for i in range(K)]
    Tdata           = [Transform(Kps[:, 0:2], H[i]) for i in range(K)]
    inliers         = [Inliers(Tdata[i][:], Kps[:, 2:4], Th) for i in range(K)]
    inliers_num     = [len(inliers[i]) for i in range(K)]
    index           = np.argmax(inliers_num)
    return inliers[index]



def Connected(matrix):
    n               = matrix.shape[0]
    undirM          = np.logical_or(matrix, matrix.T).astype(int)
    def dfs(node, vis):
        vis[node]   = True
        for neighbor in range(n):
            if undirM[node, neighbor] == 1 and not vis[neighbor]:
                dfs(neighbor, vis)
    vis             = [False] * n
    dfs(0, vis)
    return all(vis)

def GetPtC(depth_map, conf_map, color_map, fl, confidence_thresh=0.5):
    h, w            = depth_map.shape
    c_x, c_y        = (w / 2, h / 2)
    u, v            = np.meshgrid(np.arange(w), np.arange(h))
    Z               = depth_map
    X               = (u - c_x) * Z / fl
    Y               = (v - c_y) * Z / fl
    point_cloud     = np.stack((X, Y, Z), axis=-1)
    valid_mask      = (Z > 0) & (conf_map > confidence_thresh)
    valid_points    = point_cloud[valid_mask]
    valid_colors    = color_map[valid_mask]
    Final           = np.hstack((valid_points, valid_colors))
    return Final

def KpfromInlier(index_matrix, pair_indices, kp1, kp2):
    indices         = index_matrix.flatten()
    kp1_ind         = pair_indices[indices, 0]
    kp2_ind         = pair_indices[indices, 1]
    kp1_sel         = kp1[kp1_ind]
    kp2_sel         = kp2[kp2_ind]
    return kp1_sel, kp2_sel

def GetKp3d(Kps, depth, fl):
    Kps             = np.asarray(Kps)
    H, W            = depth.shape
    c_x, c_y        = W / 2, H / 2
    depth_values    = depth[Kps[:, 1].astype(int), Kps[:, 0].astype(int)]
    X               = (Kps[:, 0] - c_x) * depth_values / fl
    Y               = (Kps[:, 1] - c_y) * depth_values / fl
    Z               = depth_values
    keypoints_3d    = np.column_stack((X, Y, Z))
    return keypoints_3d

