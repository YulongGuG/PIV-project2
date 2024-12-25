import numpy as np
from scipy.io import loadmat, savemat
import functions as f
import functions_cv as fcv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

#   X -> High
#   Y -> Width
plt.ioff()
MATLABData      = loadmat('cams_info_no_extr.mat')
MATLABWrldData  = loadmat('wrld_info.mat')
MATLABKeyPt     = loadmat('kp.mat')

# Read camera info
cams_info       = MATLABData.get('cams_info')
RGBs            = np.array([cams_info[i, 0]['rgb'][0, 0] for i in range(cams_info.shape[0])])
depths          = np.array([cams_info[i, 0]['depth'][0, 0] for i in range(cams_info.shape[0])])
confs           = np.array([cams_info[i, 0]['conf'][0, 0] for i in range(cams_info.shape[0])])
fls             = np.array([cams_info[i, 0]['focal_lenght'][0, 0] for i in range(cams_info.shape[0])])

N               = RGBs.shape[0]
H               = RGBs.shape[1]
W               = RGBs.shape[2]

# Read keypoint
Kps             = []
Desc            = []
for i in range(N):
    key         = 'Feature_img' + str(i+1) + '_00000'
    KpStruct    = MATLABKeyPt.get(key)
    Keypoint    = KpStruct[0, 0]['kp']
    Descriptor  = KpStruct[0, 0]['desc']
    Kps.append(Keypoint)
    Desc.append(Descriptor)
    
for i in range(1, N):
    f.plot_image_with_keypoints(RGBs[i-1], Kps[i])
"""
IMGsmatch is a list of list where each element is a matrix of N * 2, N is the number of matched 
keypoints between img i and img j. Matched only with descriptors. The 1st column correspond to 
index in Kps and Desc of img i, and 2nd column correspond to index in Kps and Desc of img j
"""
IMGsmatch               = [
    [np.zeros((0, 0)) if i == j else f.KpMatch(Desc[i], Desc[j]) for j in range(N)]
    for i in range(N)
]

KpsComb                 = [
    [np.zeros((0, 0)) if i == j else f.zipKp(Kps[i], Kps[j], IMGsmatch[i][j]) for j in range(N)]
    for i in range(N)
]
DescComb                = [
    [np.zeros((0, 0)) if i == j else f.zipKp(Desc[i], Desc[j], IMGsmatch[i][j]) for j in range(N)]
    for i in range(N)
]

"""
InlierMatch is a list of list where each element is a matrix of N * 1, N is the number of 
credible matches. Each element of matrix correspond to the index to the corresponded matches 
list at IMGsmatch.
"""
'''InlierMatch             = [
    [np.zeros((0, 0)) if i == j else fcv.RANSAC(KpsComb[i][j], Th=5) for j in range(N)]
    for i in range(N)
]'''

InlierMatch = []
for i in range(N): 
    for j in range(N):
        if i == j:
            continue

        # 计算单应性矩阵和掩码
        H, mask = cv2.findHomography(KpsComb[i][j][:, :2], KpsComb[i][j][:, 2:4], cv2.RANSAC, 5.0)

        # 使用 mask 选择内点匹配
        inlier_matches = [KpsComb[i][j][k] for k, m in enumerate(KpsComb[i][j]) if mask[k]]
        
        # 确保 inliers 是一个正确的 N x 4 数组 (x1, y1, x2, y2)
        inliers = KpsComb[i][j][mask]  # 获取内点匹配对
        f.plotMatches2(RGBs[i], RGBs[j], Kps[i], Kps[j], i, j, IMGsmatch[i][j], inliers)

for i in range(N):
    for j in range(N):
        f.plotMatches(RGBs[i],RGBs[j], Kps[i], Kps[j], i, j, IMGsmatch[i][j], InlierMatch[i][j])

#exit()

inliers_thresh          = 10
Connections             = [
    [0 if inlierlist.shape[0] < inliers_thresh else 1 for inlierlist in corres]
    for corres in InlierMatch
]

print(Connections)
if not f.Connected(np.array(Connections)):
    print('Not all connected')
    #exit()
else:
    print('All connected')

Kp3dComb = []
RComb = []
TComb = []
for i in range(N):
    a = []
    r = []
    t = []
    for j in range(N):
        if Connections[i][j] == 0:
            a.append(np.zeros((0, 0)))
            r.append(None)
            t.append(None)
        else:
            Kp2d1, Kp2d2    = f.KpfromInlier(InlierMatch[i][j], IMGsmatch[i][j], Kps[i], Kps[j])
            Kp3d1           = f.GetKp3d(Kp2d1, depths[i], fls[i][0,0])
            Kp3d2           = f.GetKp3d(Kp2d2, depths[j], fls[j][0,0])
            R, T            = f.ICP(Kp3d1, Kp3d2) # from Kp3d1 to Kp3d2; from i to j
            Kp3d            = np.hstack((Kp3d1, Kp3d2))
            a.append(Kp3d)
            r.append(R)
            t.append(T)
    Kp3dComb.append(a)
    RComb.append(r)
    TComb.append(t)

PtC = []
for i in range(N):
    pointcloud              = f.GetPtC(depths[i], confs[i], RGBs[i], fls[i][0,0])

    PtC.append(pointcloud)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PtC[:, 0], PtC[:, 1], PtC[:, 2], c=PtC[:, 3:6], marker='o', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    """




i_ref   = 3
PtC_ref = PtC[i_ref]
Shortest_path = f.PathToRef(Connections, i_ref)

RtoRef, TtoRef = f.TransformToRef(RComb, TComb, Shortest_path, i_ref)

MergedPtC = f.MergePtc(PtC, RtoRef, TtoRef)

savemat("MergedPT.mat", {'pc': MergedPtC[:, 0:3], 'color': MergedPtC[:, 3:6]})

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(MergedPtC[:, 0], MergedPtC[:, 1], MergedPtC[:, 2], c=MergedPtC[:, 3:6], marker='.', s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()'''


#for i in range(1, N):



