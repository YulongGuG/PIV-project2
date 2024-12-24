import cv2
import numpy as np
from scipy.spatial.distance import cosine

def KpMatch_comb(D1, D2, absolute_threshold=150, th=0.75):
    matches = []
    for i, d1 in enumerate(D1):
        dist = np.linalg.norm(D2 - d1, axis=1)
        sorted_indices = np.argsort(dist)
        best1 = sorted_indices[0]
        best2 = sorted_indices[1]
        
        # 同时满足绝对值阈值和相对值阈值
        if dist[best1] < absolute_threshold and dist[best1] < th * dist[best2]:
            matches.append((i, int(best1)))

    Res = np.array(matches)
    return Res

# 特征点匹配函数，使用 BFMatcher
def KpMatch(desc1, desc2, th=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # 使用 L2 距离
    matches = bf.knnMatch(desc1, desc2, k=2)  # 计算最近邻和次近邻
    good_matches = []
    for m, n in matches:
        if m.distance < th * n.distance:  # Lowe's ratio test
            good_matches.append([m.queryIdx, m.trainIdx])
    return np.array(good_matches)

def KpMatch_FLANN(desc1, desc2, th=0.75):
    index_params = dict(algorithm=1, trees=5)  # 使用 KDTree 算法
    search_params = dict(checks=50)  # 指定递归深度（搜索次数）
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < th * n.distance:  # Lowe's ratio test
            good_matches.append([m.queryIdx, m.trainIdx])
    return np.array(good_matches)

def KpMatch_Symmetric(desc1, desc2, th=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_12 = bf.knnMatch(desc1, desc2, k=2)  # desc1 到 desc2 的匹配
    matches_21 = bf.knnMatch(desc2, desc1, k=2)  # desc2 到 desc1 的匹配
    
    good_matches = []
    for (m, n), (m2, n2) in zip(matches_12, matches_21):
        if (m.distance < th * n.distance and m2.distance < th * n2.distance and
                m.queryIdx == m2.trainIdx and m.trainIdx == m2.queryIdx):  # 对称性检查
            good_matches.append([m.queryIdx, m.trainIdx])
    return np.array(good_matches)

def KpMatch_Ratio(desc1, desc2, th=0.75, ratio=0.6):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < th * n.distance and m.distance < ratio * np.mean([m.distance, n.distance]):
            good_matches.append([m.queryIdx, m.trainIdx])
    return np.array(good_matches)

def KpMatch_Hist(desc1, desc2, th=0.3):
    good_matches = []
    for i, d1 in enumerate(desc1):
        distances = [cosine(d1, d2) for d2 in desc2]
        min_idx = np.argmin(distances)
        if distances[min_idx] < th:  # 使用余弦相似度阈值
            good_matches.append([i, min_idx])
    return np.array(good_matches)




# 合并关键点和描述符的函数
def zipKp(kps1, kps2, matches):
    if len(matches) == 0:
        return np.zeros((0, 4))  # 空矩阵返回
    kp1_coords = np.array([kps1[m[0]] for m in matches])
    kp2_coords = np.array([kps2[m[1]] for m in matches])
    return np.hstack((kp1_coords, kp2_coords))

# RANSAC 函数
def RANSAC(matches_kps, Th=10):
    if matches_kps.shape[0] < 4:  # RANSAC 需要至少 4 个点
        return np.zeros((0, 1))

    # 提取匹配点对
    src_pts = matches_kps[:, :2]
    dst_pts = matches_kps[:, 2:4]

    # 使用 RANSAC 计算 Homography
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, Th)

    # 筛选内点索引
    inlier_indices = np.where(mask.ravel() == 1)[0]
    return inlier_indices.reshape(-1, 1)

def RANSAC_F(matches_kps, Th=1.0):
    if matches_kps.shape[0] < 8:  # 至少需要 8 个点计算 Fundamental Matrix
        return np.zeros((0, 1))

    # 提取匹配点对
    src_pts = matches_kps[:, :2]
    dst_pts = matches_kps[:, 2:4]

    # 使用 RANSAC 计算 Fundamental Matrix
    _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, Th)

    # 筛选内点索引
    inlier_indices = np.where(mask.ravel() == 1)[0]
    return inlier_indices.reshape(-1, 1)

def RANSAC_E(matches_kps, K, Th=1.0):
    if matches_kps.shape[0] < 5:  # 至少需要 5 个点计算 Essential Matrix
        return np.zeros((0, 1))

    # 提取匹配点对
    src_pts = matches_kps[:, :2]
    dst_pts = matches_kps[:, 2:4]

    # 使用相机内参矩阵 K 计算本质矩阵
    _, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.99, threshold=Th)

    # 筛选内点索引
    inlier_indices = np.where(mask.ravel() == 1)[0]
    return inlier_indices.reshape(-1, 1)

def RANSAC_A(matches_kps, Th=3.0):
    if matches_kps.shape[0] < 3:  # 至少需要 3 个点计算 Affine Transformation
        return np.zeros((0, 1))

    # 提取匹配点对
    src_pts = matches_kps[:, :2]
    dst_pts = matches_kps[:, 2:4]

    # 使用 RANSAC 计算仿射变换
    _, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=Th)

    # 筛选内点索引
    inlier_indices = np.where(mask.ravel() == 1)[0]
    return inlier_indices.reshape(-1, 1)


