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

sift            = cv2.SIFT_create()
GrayImg         = []
Kps             = []
Desc            = []
for i in range(N):
    gray_img    = cv2.cvtColor(RGBs[i], cv2.COLOR_RGB2GRAY)
    gray_img    = (gray_img * 255).astype(np.uint8)
    kp, desc    = sift.detectAndCompute(gray_img, None)
    GrayImg.append(gray_img)
    Kps.append(kp)
    Desc.append(desc)

Kps_matrix_list = [np.array([kp.pt for kp in kps]) for kps in Kps]
Desc_matrix_list = [desc for desc in Desc]

Inliers = []
bf                      = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
for i in range(N):
    a = []
    for j in range(N):
        matches         = bf.match(Desc[i], Desc[j])
        matches         = sorted(matches, key=lambda x: x.distance)
        good_matches    = matches[:100]
        pts1            = np.float32([Kps[i][m.queryIdx].pt for m in good_matches])
        pts2            = np.float32([Kps[j][m.trainIdx].pt for m in good_matches])
        #H, mask         = cv2.findHomography(pts1, pts2, cv2.RANSAC, 1.0)
        H, mask         = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 1.0)
        inlier_matches  = [m for i, m in enumerate(good_matches) if mask[i]]
        '''img_matches     = cv2.drawMatches(GrayImg[i], Kps[i], GrayImg[j], Kps[j], inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
        plt.imshow(img_matches_rgb)
        plt.axis('off')  # Hide axis
        plt.show()'''
        matches_matrix = np.array([
            [
                Kps[i][m.queryIdx].pt[0],
                Kps[i][m.queryIdx].pt[1],
                Kps[j][m.trainIdx].pt[0],
                Kps[j][m.trainIdx].pt[1],
                *Desc[i][m.queryIdx],      # 源描述符
                *Desc[j][m.trainIdx]       # 目标描述符
            ]
            for m in inlier_matches
        ])
        a.append(matches_matrix)
    Inliers.append(a)

inliers_thresh          = 15
Connections             = [
    [0 if matches_matrix.shape[0] < inliers_thresh else 1 for matches_matrix in corres]
    for corres in Inliers
]

print(Connections)
if not f.Connected(np.array(Connections)):
    print('Not all connected')
    #exit()
else:
    print('All connected')

PtC = []
for i in range(N):
    pointcloud              = f.GetPtC(depths[i], confs[i], RGBs[i], fls[i][0,0])

    PtC.append(pointcloud)

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
            Kp2d1, Kp2d2    = Inliers[i][j][:,0:2], Inliers[i][j][:,2:4]
            Kp3d1           = f.GetKp3d(Kp2d1, depths[i], fls[i][0,0])
            Kp3d2           = f.GetKp3d(Kp2d2, depths[j], fls[j][0,0])
            Kps3d1          = f.GetKp3d(Kps_matrix_list[i], depths[i], fls[i][0,0])
            Kps3d2          = f.GetKp3d(Kps_matrix_list[j], depths[j], fls[j][0,0])
            #R, T            = f.ICP2(Kps3d1, Desc_matrix_list[i], Kps3d2, Desc_matrix_list[j])
            #R, T            = f.ICP2(Kp3d1, Inliers[i][j][:,4:132], Kp3d2, Inliers[i][j][:,132:260])
            R, T            = f.ICP2(Kp3d1, Kp3d2)
            #R, T            = f.ICP2(PtC[i][:,:3], PtC[j][:,:3])
            Kp3d            = np.hstack((Kp3d1, Kp3d2))
            a.append(Kp3d)
            r.append(R)
            t.append(T)
    Kp3dComb.append(a)
    RComb.append(r)
    TComb.append(t)


    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PtC[:, 0], PtC[:, 1], PtC[:, 2], c=PtC[:, 3:6], marker='o', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    """




i_ref   = 1
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



