load('MergedPT.mat')
color255 = color * 255;
ptc = pointCloud(pc, 'Color', color);
pcshow(ptc); 