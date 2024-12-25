import numpy as np
from scipy.io import loadmat

# 加载 .mat 文件
file_path = 'cams_info_no_extr.mat'
mat_data = loadmat(file_path)

# 打印所有变量名称
print("Variables in the .mat file:")
print(mat_data.keys())

# 提取 cams_info 变量
cams_info = mat_data['cams_info']

# 打印 cams_info 的基本信息
print("Type of cams_info:", type(cams_info))
print("Shape of cams_info:", cams_info.shape)

#print(cams_info)

# 遍历每个元素，查看具体内容
for i, item in enumerate(cams_info):
    print(f"Element {i}: {item}")