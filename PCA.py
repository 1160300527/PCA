import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mnist import MNIST
from PIL import Image

n = 100
m = 3 #生成的数据维度
loc = [1,2,3]
scale = [5,5,2]
k = 2   #降维后的维度


def Covariance(X):
    return X.T.dot(X)/X.shape[0]

def Average(X):
    return np.sum(X,axis = 0)/X.shape[0]

def PCA(X,k):
    X = X-Average(X)
    covariance = Covariance(X)
    eigenvalue,eigenvector = np.linalg.eig(covariance)
    indexSort = np.argsort(-eigenvalue)
    return np.mat(eigenvector[indexSort[:k]]).T



def birthX(num,scale,m,loc):
    X = []
    for i in range(m):
        if(len(X) == 0):
            X = np.random.normal(loc[i], scale[i], num)
            continue
        X = np.vstack((X, np.random.normal(loc[i], scale[i], num)))
    return X.T


def arrayImage(img):
    img = Image.fromarray(img.astype(np.uint8))
    img.show()
    return img


def figureWritten():
    image,label = MNIST(path="figure",return_type="numpy").load_testing()
    img = []
    for i in range(2000):
        if  (label[i]==8):
            img.append(image[i])
            if  len(img)>=100:
                break
    img = arrayImage(np.array(img))
    

#根据三个点计算投影平面
def get_panel(p1,p2,p3):
    a = ( (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]) )
 
    b = ( (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]) )
 
    c = ( (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]) )
 
    d = ( 0-(a*p1[0]+b*p1[1]+c*p1[2]) )

    return a,b,c,d


def draw(X,reduce,topKeigenvector):
    X=X.T
    reduce = reduce.T
    topKeigenvector = (np.vstack((topKeigenvector.T,[0,0,0]))*20).tolist()
    a,b,c,d = get_panel(topKeigenvector[0],topKeigenvector[1],topKeigenvector[2])
    topKeigenvector = np.mat(topKeigenvector)
    max = int(np.max(X))
    min = -max
    x = np.arange(min, max, 1)
    y = np.arange(min, max, 1)
    x,y = np.meshgrid(x,y)
    z = -(a*x+b*y+d)/c
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim3d(min,max)
    ax.set_ylim3d(min,max)
    ax.set_zlim3d(min,max)
    ax.scatter(X[0],X[1],X[2],color = 'r')
    ax.plot_surface(x,y,z,rstride=1, cstride=1)
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlim3d(min,max)
    ax.set_ylim3d(min,max)
    ax.set_zlim3d(min,max)
    ax.scatter(reduce[0],reduce[1],0)
    plt.show()
    
X = birthX(n,scale,m,loc)
topKeigenvector = PCA(X,k)
reduce = X.dot(topKeigenvector)
figureWritten()
draw(X,reduce,topKeigenvector)