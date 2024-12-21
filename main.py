import numpy as np
from scipy.io import loadmat, savemat
import functions as f

#   X -> High
#   Y -> Width

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

IMGsmatch       = [
    [0 if i == j else f.KpMatch(Desc[i], Desc[j]) for j in range(N)]
    for i in range(N)
]

KpsComb         = [
    [0 if i == j else f.zipKp(Kps[i], Kps[j], IMGsmatch[i][j] if len(IMGsmatch[i][j]) > 3 else []) for j in range(N)]
    for i in range(N)
]

DescComb         = [
    [0 if i == j else f.zipKp(Desc[i], Desc[j], IMGsmatch[i][j] if len(IMGsmatch[i][j]) > 3 else []) for j in range(N)]
    for i in range(N)
]

InlierMatch      = [
    [0 if i == j else f.RANSAC(KpsComb[i][j]) for j in range(N)]
    for i in range(N)
]

# Compute camera matrix from camera info
PMs             = []
PtCs            = []
E               = f.GetExtrinsicZero()
central         = np.array([H/2, W/2])
for i in range(N):
    I           = f.GetIntrinsic(fls[i], central)
    P           = I @ E
    PMs.append(P)

# Compute point cloud from depth and camera info
    PtC         = np.zeros((depths.size, 3))

    i, j        = np.meshgrid(np.arange(depths.shape[0]), np.arange(depths.shape[1]), indexing='ij')
    pixels      = np.stack((i.ravel(), j.ravel(), np.ones(depths.size)), axis=1)

    depth_flat  = depths.ravel()
    PtC         = (PMs @ pixels.T).T * depth_flat[:, np.newaxis]


# Compute the color of point cloud from rgb

# Match keypoints

# RANSAC with ICP