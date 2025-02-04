import numpy as np
import math
import itertools
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, KDTree


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

def ICP2(point_cloud1, point_cloud2, max_iterations=50, tolerance=1e-5):
    """
    Perform Iterative Closest Point (ICP) algorithm.

    Parameters:
        point_cloud1 (numpy.ndarray): Source point cloud (Nx3).
        point_cloud2 (numpy.ndarray): Target point cloud (Mx3).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        R (numpy.ndarray): Rotation matrix (3x3).
        T (numpy.ndarray): Translation vector (3x1).
    """
    # Ensure inputs are numpy arrays
    point_cloud1 = np.asarray(point_cloud1)
    point_cloud2 = np.asarray(point_cloud2)

    # Initialize transformation
    R = np.eye(3)
    T = np.zeros((3, 1))

    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(point_cloud2)

    for iteration in range(max_iterations):
        # Apply current transformation to source point cloud
        transformed_pc1 = (R @ point_cloud1.T).T + T.T

        # Find the nearest neighbors in target point cloud
        distances, indices = tree.query(transformed_pc1)
        matched_pc2 = point_cloud2[indices]

        # Compute centroids of matched point clouds
        centroid_pc1 = np.mean(transformed_pc1, axis=0)
        centroid_pc2 = np.mean(matched_pc2, axis=0)

        # Center the point clouds
        centered_pc1 = transformed_pc1 - centroid_pc1
        centered_pc2 = matched_pc2 - centroid_pc2

        # Compute the covariance matrix
        H = centered_pc1.T @ centered_pc2

        # Perform Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)

        # Compute the optimal rotation
        R_new = Vt.T @ U.T

        # Handle special reflection case
        if np.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T

        # Compute the optimal translation
        T_new = centroid_pc2 - (R_new @ centroid_pc1.T).T

        # Update transformation
        R = R_new @ R
        T = (R_new @ T) + T_new.reshape(-1, 1)

        # Check for convergence
        if np.linalg.norm(T_new) < tolerance:
            break

    return R, T.reshape(-1)



def KpMatch(D1, D2, MatchNum=100):
    tree = cKDTree(D2)
    dist, idx = tree.query(D1, k=1)
    matchs = [(i, int(idx[i]), dist[i]) for i in range(len(D1))]
    M = np.array(matchs)
    sortedMatch = M[M[:, 2].argsort()]
    return sortedMatch[:MatchNum, :2].astype(int)

"""
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
"""

def zipKp(Kp1, Kp2, match):
    if match.shape[0] < 1:
        return np.zeros((0, 0)) 
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
def RANSAC(Kps, N = 4, K = 100, Th = 10):
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

def bfs_paths(connections, start, target):
    N                           = len(connections)
    visited                     = [False] * N
    parent                      = [-1] * N
    queue                       = deque([start])
    visited[start]              = True
    while queue:
        node = queue.popleft()
        if node == target:
            path                = []
            while node != -1:
                path.append(node)
                node            = parent[node]
            return path[::-1]
        for Neigh in range(N):
            if (connections[node][Neigh] == 1 or connections[Neigh][node] == 1) and not visited[Neigh]:
                visited[Neigh]  = True
                parent[Neigh]   = node
                queue.append(Neigh)
    return []

def PathToRef(connections, i_ref):
    N           = len(connections)
    all_paths   = []
    for i in range(N):
        path    = bfs_paths(connections, i, i_ref)
        all_paths.append(path)
    return all_paths

def InverseTransform(R_ij, T_ij):
    R_ji        = R_ij.T
    T_ji        = -R_ji @ T_ij
    return R_ji, T_ji

def TransformToRef(R, T, paths, i_ref):
    N                                       = len(R)
    ref_R                                   = [None] * N
    ref_T                                   = [None] * N
    ref_R[i_ref]                            = np.eye(3)
    ref_T[i_ref]                            = np.zeros(3)
    for i in range(N):
        if i == i_ref:
            continue
        path                                = paths[i]
        R_to_ref                            = np.eye(3)
        T_to_ref                            = np.zeros(3)
        for j in range(len(path) - 1):
            src, dst                        = path[j], path[j + 1]
            if R[src][dst] is None:
                R[src][dst], T[src][dst]    = InverseTransform(R[dst][src], T[dst][src])
            R_to_ref                        = R_to_ref @ R[src][dst]
            T_to_ref                        = R_to_ref @ T[src][dst] + T_to_ref
        ref_R[i]                            = R_to_ref
        ref_T[i]                            = T_to_ref
    return ref_R, ref_T

def MergePtc(point_clouds, rotations, translations):
    merged_cloud = []
    for pc, R, T in zip(point_clouds, rotations, translations):
        coords = pc[:, :3]
        colors = pc[:, 3:]
        transformed_coords = (R @ coords.T).T + T
        transformed_pc = np.hstack((transformed_coords, colors))
        merged_cloud.append(transformed_pc)
    merged_cloud = np.vstack(merged_cloud)
    merged_cloud = np.unique(merged_cloud, axis=0)
    return merged_cloud

def plotMatches(img1, img2, kp1, kp2,  i, j, matches, inliers=None,):
    '''plt.figure(figsize=(12, 6))
    plt.imshow(np.hstack((img1, img2)), cmap='gray')
    for m in matches:
        x1, y1 = kp1[m[0]]
        x2, y2 = kp2[m[1]]
        color = 'g' if inliers is not None and inliers[m[0]] else 'r'
        plt.plot([x1, x2 + img1.shape[1]], [y1, y2], color=color)
    plt.show()'''
    plt.figure(figsize=(12, 6))
    plt.imshow(np.hstack((img1, img2)), cmap='gray')
    for m in matches:
        x1, y1 = kp1[m[0]]
        x2, y2 = kp2[m[1]]
        color = 'g' if inliers is not None and m[0] in inliers else 'r'
        plt.plot([x1, x2 + img1.shape[1]], [y1, y2], color=color)
        

    plt.title(f'Index i = {i}, Index j = {j}')
    plt.show()

def plotMatches2(img1, img2, kp1, kp2, i, j, matches, inliers=None):
    '''绘制匹配的内点'''
    plt.figure(figsize=(12, 6))
    plt.imshow(np.hstack((img1, img2)), cmap='gray')

    # 如果有内点，绘制它们的匹配
    if inliers is not None:
        for m in range(inliers.shape[0]):
            # 获取匹配点对的坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = inliers[m, 0, 0], inliers[m, 0, 1], inliers[m, 0, 2], inliers[m, 0, 3]
            # 画线表示匹配
            plt.plot([x1, x2 + img1.shape[1]], [y1, y2], color='g')

    plt.title(f'Index i = {i}, Index j = {j}')
    plt.show()


def plot_image_with_keypoints(img, keypoints, title="Image with Keypoints"):
    # Display the image
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    
    # Overlay the keypoints
    plt.scatter(keypoints[:, 0], keypoints[:, 1], color='r', marker='.', s=50, label='Keypoints')
    
    # Add title and show the plot
    plt.title(title)
    plt.axis('off')  # Optional: turn off axis labels
    plt.legend()
    plt.show()

