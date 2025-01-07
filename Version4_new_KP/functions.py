import numpy as np
import math
import itertools
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import heapq
from math import inf
from scipy.spatial import KDTree

def icp_single(source, target, max_iterations=50, tolerance=1e-4):
    source_coords = source[:, :3]
    target_coords = target[:, :3]
    prev_error = float('inf')
    R = np.eye(3)
    t = np.zeros(3)
    for _ in range(max_iterations):
        tree = KDTree(target_coords)
        distances, indices = tree.query(source_coords)
        matched_target_coords = target_coords[indices]
        threshold = 2 * np.mean(distances)
        valid_indices = distances < threshold
        source_coords = source_coords[valid_indices]
        matched_target_coords = matched_target_coords[valid_indices]
        centroid_source = np.mean(source_coords, axis=0)
        centroid_target = np.mean(matched_target_coords, axis=0)
        H = (source_coords - centroid_source).T @ (matched_target_coords - centroid_target)
        U, _, Vt = np.linalg.svd(H)
        R_iter = U @ Vt
        if np.linalg.det(R_iter) < 0:
            Vt[-1, :] *= -1
            R_iter = U @ Vt
        t_iter = centroid_target - R_iter @ centroid_source
        source_coords = (R_iter @ source_coords.T).T + t_iter
        R = R_iter @ R
        t = R_iter @ t + t_iter
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    aligned_source = np.hstack(((R @ source[:, :3].T).T + t, source[:, 3:]))
    return R, t, aligned_source

def MergeICP(PtCs, i_ref, shortpath, max_iterations=50, tolerance=1e-5):
    ref_cloud                   = PtCs[i_ref]
    R_list                      = np.zeros((len(PtCs), 3, 3))
    T_list                      = np.zeros((len(PtCs), 3))
    R_list[i_ref, :, :]         = np.eye(3)
    T_list[i_ref, :]            = np.zeros(3)
    merged_cloud                = ref_cloud
    Sspath                      = sorted(shortpath, key=len)

    for i, p in enumerate(Sspath):
        th                      = len(p)
        if th < 2:
            continue
        source_cloud            = PtCs[p[0]]
        ref_cloud               = PtCs[p[1]]
        R, t, aligned_cloud     = icp_single(source_cloud, merged_cloud, max_iterations, tolerance)
        R_list[p[0], :, :]      = R
        T_list[p[0], :]         = t
        print(merged_cloud.shape)
        print(aligned_cloud.shape)
        merged_cloud            = np.vstack((merged_cloud, aligned_cloud))
    return merged_cloud, R_list, T_list
'''
def MergeICP(PtCs, i_ref, shortpath, max_iterations=50, tolerance=10):
    ref_cloud                   = PtCs[i_ref]
    R_list                      = np.zeros((len(PtCs), 3, 3))
    T_list                      = np.zeros((len(PtCs), 3))
    R_list[i_ref, :, :]         = np.eye(3)
    T_list[i_ref, :]            = np.zeros(3)
    merged_cloud                = ref_cloud
    Sspath                      = sorted(shortpath, key=len)

    for i, p in enumerate(Sspath):
        th                      = len(p)
        if th < 2:
            continue
        source_cloud            = PtCs[p[0]]
        ref_cloud               = PtCs[p[1]]
        R, t, aligned_cloud     = icp_single(source_cloud, ref_cloud, max_iterations, tolerance)
        R_list[p[0], :, :]      = R
        T_list[p[0], :]         = t
        if th > 2:
            for j in range(1, th-1):
                R               = R_list[p[j]]
                t               = T_list[p[j]]
                aligned_cloud   = np.hstack(((R @ aligned_cloud[:, :3].T).T + t, aligned_cloud[:, 3:]))
        print(merged_cloud.shape)
        print(aligned_cloud.shape)
        merged_cloud            = np.vstack((merged_cloud, aligned_cloud))
    return merged_cloud
'''

# FROM P TO Q
# qi = R @ pi + T
def KpICP(P, Q):
    MeanP           = np.mean(P, axis = 0)
    MeanQ           = np.mean(Q, axis = 0)
    Pl              = P - MeanP
    Ql              = Q - MeanQ
    W               = Ql.T @ Pl
    U, S, V         = np.linalg.svd(W)
    R               = U @ V
    T               = MeanQ - R @ MeanP
    return R, T
"""
def KpMatch(D1, D2, MatchNum=100):
    tree = cKDTree(D2)
    dist, idx = tree.query(D1, k=1)
    matchs = [(i, int(idx[i]), dist[i]) for i in range(len(D1))]
    M = np.array(matchs)
    sortedMatch = M[M[:, 2].argsort()]
    return sortedMatch[:MatchNum, :2]
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

def KpMatch_comb(D1, D2, absolute_threshold=150, th=0.75):
    matches = []
    for i, d1 in enumerate(D1):
        dist = np.linalg.norm(D2 - d1, axis=1)
        sorted_indices = np.argsort(dist)
        best1 = sorted_indices[0]
        best2 = sorted_indices[1]
        if dist[best1] < absolute_threshold and dist[best1] < th * dist[best2]:
            matches.append((i, int(best1)))

    Res = np.array(matches)
    return Res

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
'''
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
'''
def dijkstra_paths(connections, start, target):
    N = len(connections)
    distances = [inf] * N
    parent = [-1] * N
    distances[start] = 0
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, node = heapq.heappop(priority_queue)
        if node == target:
            path = []
            while node != -1:
                path.append(node)
                node = parent[node]
            return path[::-1]
        for neighbor in range(N):
            weight = connections[node][neighbor]
            if weight > 0:
                new_distance = current_distance + 1 / weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parent[neighbor] = node
                    heapq.heappush(priority_queue, (new_distance, neighbor))
    return []

def PathToRef(connections, i_ref):
    N = len(connections)
    all_paths = []
    for i in range(N):
        path = dijkstra_paths(connections, i, i_ref)
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

def MergeTransformation(RtoRef, TtoRef, RefineR, RefineT):
    N = len(RtoRef)
    FinalR = []
    FinalT = []
    
    for i in range(N):
        R_final = np.dot(RefineR[i], RtoRef[i])
        T_final = np.dot(RefineR[i], TtoRef[i]) + RefineT[i].reshape(3, 1)
        
        FinalR.append(R_final)
        FinalT.append(T_final)
    
    return FinalR, FinalT

def MergePtc(point_clouds, rotations, translations):
    PC_list = []
    for pc, R, T in zip(point_clouds, rotations, translations):
        coords = pc[:, :3]
        colors = pc[:, 3:]
        transformed_coords = (R @ coords.T).T + T
        transformed_pc = np.hstack((transformed_coords, colors))
        PC_list.append(transformed_pc)
    merged_cloud = np.vstack(PC_list)
    merged_cloud = np.unique(merged_cloud, axis=0)
    return merged_cloud, PC_list

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
    inl = inliers.astype(int)
    for m in matches:
        x1, y1 = kp1[m[0]]
        x2, y2 = kp2[m[1]]
        color = 'g' if inliers is not None and m[0] in inl else 'r'
        plt.plot([x1, x2 + img1.shape[1]], [y1, y2], color=color)
        

    plt.title(f'Index i = {i}, Index j = {j}')
    plt.show()

def plotMatches2(img1, img2, kp1, kp2, i, j, matches, inliers=None):
    plt.figure(figsize=(12, 6))
    plt.imshow(np.hstack((img1, img2)), cmap='gray')
    if inliers is not None:
        for m in range(inliers.shape[0]):
            x1, y1, x2, y2 = inliers[m, 0, 0], inliers[m, 0, 1], inliers[m, 0, 2], inliers[m, 0, 3]
            plt.plot([x1, x2 + img1.shape[1]], [y1, y2], color='g')
    plt.title(f'Index i = {i}, Index j = {j}')
    plt.show()


def plot_image_with_keypoints(img, keypoints, title="Image with Keypoints"):
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], color='r', marker='.', s=50, label='Keypoints')
    plt.title(title)
    plt.axis('off') 
    plt.legend()
    plt.show()

