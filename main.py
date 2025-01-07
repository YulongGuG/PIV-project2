import numpy as np
from scipy.io import loadmat, savemat
import functions as f
# import functions_cv as fcv
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

'''
for i in range(1, N):
    f.plot_image_with_keypoints(RGBs[i-1], Kps[i])
'''

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
'''
InlierMatch             = [
    [np.zeros((0, 0)) if i == j else f.RANSAC(KpsComb[i][j], Th=5) for j in range(N)]
    for i in range(N)
]
'''

InlierMatch = []
Connections = []
for i in range(N): 
    ab = []
    cd = []
    for j in range(N):
        if i == j:
            ab.append(np.zeros(0))
            cd.append(0)
            continue
        src_points = KpsComb[i][j][:, 0:2]
        dst_points = KpsComb[i][j][:, 2:4]

        if len(src_points) < 4 or len(dst_points) < 4:
            # print(f"Not enough points to calculate homography for KpsComb[{i}][{j}]")
            H, mask = None, None
            ab.append(np.zeros(0))
            cd.append(0)
            continue
        else:
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 20.0)               

        inlier_matches = [KpsComb[i][j][k] for k, m in enumerate(KpsComb[i][j]) if mask[k]]
        inliers = KpsComb[i][j][mask.flatten() == 1]
        print(inliers.shape[0])
        ab.append(inliers)
        cd.append(inliers.shape[0])
    InlierMatch.append(ab)
    Connections.append(cd)
Connections = np.array(Connections)

print(np.mean(Connections))
"""
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        f.plotMatches(RGBs[i],RGBs[j], Kps[i], Kps[j], i, j, IMGsmatch[i][j], InlierMatch[i][j])
"""
#exit()

inliers_thresh          = np.mean(Connections)
Connected               = [
    [0 if inlierlist.shape[0] < inliers_thresh else 1 for inlierlist in corres]
    for corres in InlierMatch
]

print(Connections)
print(np.array(Connected))
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
            r.append(np.eye(3))
            t.append(np.zeros(3))
        else:
            #Kp2d1, Kp2d2    = f.KpfromInlier(InlierMatch[i][j], IMGsmatch[i][j], Kps[i], Kps[j])
            Kp2d1, Kp2d2    = InlierMatch[i][j][:,0:2], InlierMatch[i][j][:,2:4]
            Kp3d1           = f.GetKp3d(Kp2d1, depths[i], fls[i][0,0])
            Kp3d2           = f.GetKp3d(Kp2d2, depths[j], fls[j][0,0])
            R, T            = f.KpICP(Kp3d1, Kp3d2) # from Kp3d1 to Kp3d2; from i to j
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
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=pointcloud[:, 3:6], marker='o', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    '''

i_ref   = 3
PtC_ref = PtC[i_ref]
Shortest_path = f.PathToRef(Connections, i_ref)

RtoRef, TtoRef = f.TransformToRef(RComb, TComb, Shortest_path, i_ref)

MergedPtC, PtC_list = f.MergePtc(PtC, RtoRef, TtoRef)

MergedPtC, RefineR, RefineT = f.MergeICP(PtC_list, i_ref, Shortest_path, 70, 1e-2)

FinalR, FinalT = f.MergeTransformation(RtoRef, TtoRef, RefineR, RefineT)

# MergedPtC, PtC_list = f.MergePtc(PtC_list, R_list, T_list)

transforms = {}

for i in range(N):
    R = FinalR[i]
    T = FinalT[i]
    transforms[f'pointcloud_{i+1}'] = {'R': R, 'T': T}

savemat("output.mat", {'pc': MergedPtC[:, 0:3], 'color': MergedPtC[:, 3:6]})
savemat('transforms.mat', {'transforms': transforms})

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(MergedPtC[:, 0], MergedPtC[:, 1], MergedPtC[:, 2], c=MergedPtC[:, 3:6], marker='.', s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()'''


#for i in range(1, N):



