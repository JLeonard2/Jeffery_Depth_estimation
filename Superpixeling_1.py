from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
#import PIL.Image as Image
import os

train_dir = os.path.join("D://自学//深度學習+Python//深度图像//代码//Superpixel//输入")
train_image_names = os.listdir(train_dir)
#print(train_image_names)
n=len(train_image_names) #计算图片数量

for i in range(6):

    #img = io.imread("%d.png"%(i))  #读取图像
    img = io.imread(train_dir +"/"+ train_image_names[i])
    segments1 = slic(img, n_segments=25, compactness=50)
    out2=mark_boundaries(img,segments1)
    #plt.imshow(out2)
    


    fig, ax = plt.subplots(1,1)
    #im = out2[:, :, (2, 1, 0)]
    ax.imshow(out2)
    # 去除图像周围的白边
    height, width, channels = out2.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width/100.0/1.0, height/100.0/1.0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) #将图像放置在画布左下角并1：1铺满画布
    #plt.margins(1,1)
    #dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
    plt.savefig('./深度图像/代码/Superpixel/输出/%d.png'%(i),dpi=300)
    #transPNG('%d.png'%(i+10),'%d.png'%(i+10))
