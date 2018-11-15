import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mnist import MNIST
from PIL import Image

n = 100
m = 3 #生成的数据维度
loc = [1,2,3]#均值
scale = [5,5,2]#方差
k = 2   #降维后的维度

#计算协方差矩阵
def Covariance(X):
    return X.T.dot(X)/X.shape[0]

#计算均值
def Average(X):
    return np.sum(X,axis = 0)/X.shape[0]

#寻找前k个主要成分（需要进行中心化后的数据）
def PCA(X,k):
    covariance = Covariance(X)
    eigenvalue,eigenvector = np.linalg.eig(covariance)
    indexSort = np.argsort(-eigenvalue)
    return np.mat(eigenvector[indexSort[:k]]).T

#将降维后的数据重构到原来的空间中
def backSquare(reduce,average,topKeigenvector):
    return reduce.dot(topKeigenvector.T)+average


#生成3维空间中的数据点
def birthX(num,scale,m,loc):
    X = []
    for i in range(m):
        if(len(X) == 0):
            X = np.random.normal(loc[i], scale[i], num)
            continue
        X = np.vstack((X, np.random.normal(loc[i], scale[i], num)))
    return X.T


#将array转化为图片
def arrayImage(img):
    #图采用unit8格式
    img = Image.fromarray(img.astype(np.uint8))
    return img

#将image中的多张图片合成一张大图展示
#img:图片集合   row:每行图的数量    col:每列图的数量    width:每张
def combine(image,row,col,width,height,type):
    Img = Image.new(type,(col*width,row*height))
    for i in range(len(image)):
        photo = arrayImage(np.array(image[i]).reshape(width,height))
        Img.paste(photo,((i%col)*width,int(i/col)*height))
    return Img

#测试手写数字
def figureWritten():
    #获取测试数据
    image,label = MNIST(path="figure",return_type="numpy").load_testing()
    img = []
    for i in range(2000):
        #筛选数字8的图片
        if  (label[i]==8):
            img.append(image[i])
            #仅选出100张
            if  len(img)>=100:
                break
    img = np.array(img)
    #将100张图片合成为一张大图进行展示
    Img = combine(img,10,10,28,28,'L')
    Img.show()
    average = Average(img)
    img = img - average
    topKeigenvector = PCA(img,1)
    reduce = img*(topKeigenvector)
    back = backSquare(reduce,average,topKeigenvector)
    combine(back,10,10,28,28,'L').show()


#根据三个点计算投影平面各属性值：ax+by+cz+d=0
def get_panel(p1,p2,p3):
    a = ( (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]) )
    b = ( (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]) )
    c = ( (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]) )
    d = ( 0-(a*p1[0]+b*p1[1]+c*p1[2]) )

    return a,b,c,d


#根据生成的三维数据画图
def draw(X,reduce,topKeigenvector,average,back):
    X=X.T
    back = (back).T
    reduce = reduce.T
    max = int(np.max(X))
    min = -max
    fig = plt.figure()
    ax = Axes3D(fig)
    topKeigenvector = (np.vstack((topKeigenvector.T,[0,0,0]))*20).tolist()
    #计算各参数
    a,b,c,d = get_panel(topKeigenvector[0],topKeigenvector[1],topKeigenvector[2])
    topKeigenvector = np.mat(topKeigenvector)
    x = np.arange(min, max, 1)
    y = np.arange(min, max, 1)
    x,y = np.meshgrid(x,y)
    z = -(a*x+b*y+d)/c
    ax.plot_surface(x,y,z,rstride=1, cstride=1) #,cmap=plt.get_cmap('rainbow'))
    ax.set_xlim3d(min,max)
    ax.set_ylim3d(min,max)
    ax.set_zlim3d(min,max)
    ax.scatter(X[0],X[1],X[2],color = 'r')
    ax.scatter(back[0],back[1],back[2],color = 'y')
    plt.show()
    
X = birthX(n,scale,m,loc)
average = Average(X)
X = X-average
topKeigenvector = PCA(X,k)
reduce = X.dot(topKeigenvector)
back = backSquare(reduce,average,topKeigenvector)
figureWritten()
draw(X,reduce,topKeigenvector,average,back)