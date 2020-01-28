from skimage.segmentation import slic,mark_boundaries
from skimage import io
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import os

train_dir = os.path.join("D://自学//深度學習+Python//深度图像//代码//Superpixel//输入")
train_image_names = os.listdir(train_dir)
#print(train_image_names)
n=len(train_image_names) #计算图片数量
#img = io.imread("%d.png"%(i))  #读取图像
image = io.imread(train_dir +"/"+ train_image_names[0])

seg_num = 25 # 设置超像素块数量
segments = slic(image, n_segments=seg_num, compactness=50) # 切割函数
out2=mark_boundaries(image,segments) # 画边界
fig = plt.figure('Superpixels')
for i in range(seg_num):
    #print("segments:\n", segments)
    #print("np.unique(segments):", np.unique(segments))
    mask = np.where(segments==i,1,0) # 遮罩
    #mask = np.array([mask,mask,mask])
    mask = np.expand_dims(mask, axis=2) # 扩维(1)
    mask = np.concatenate((mask, mask, mask), axis=-1) # 扩维(2)
    #print(mask.shape)
    seg = np.multiply(image,mask)
    #print(image.shape)
    #print(seg.shape)
    ax = fig.add_subplot(5, 5, i+1)
    ax.imshow(seg)
plt.show()