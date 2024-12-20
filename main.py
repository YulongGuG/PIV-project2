import numpy as np
from scipy.io import loadmat, savemat

#   X -> High
#   Y -> Width

MATLABData      = loadmat('cams_info_no_extr.mat')
MATLABWrldData  = loadmat('wrld_info.mat')
MATLABKeyPt     = loadmat('kp.mat')

# Read camera info
cams_info        = MATLABData.get('cams_info')
# Read rgd data
RGBs = np.array([cams_info[i, 0]['rgb'][0, 0] for i in range(cams_info.shape[0])])
# RGBs = cams_info[1,0]['rgb'][0,0]
# Read depth data
depths = np.array([cams_info[i, 0]['rgb'][0, 0] for i in range(cams_info.shape[0])])


# Compute camera matrix from camera info

# Compute point cloud from depth and camera info

# Compute the color of point cloud from rgb

# Match keypoints

# RANSAC with ICP