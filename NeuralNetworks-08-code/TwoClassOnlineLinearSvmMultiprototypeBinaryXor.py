import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

margin = 1
learning_rate = 0.025
weight_decay = 0.01
C = 2
K = 2 # number of clusters
clusterN = 10
maxEpoch = 4

X1_1=0.2 * np.random.rand(clusterN,3)   # class '0'
X1_1 += [0,0,0]
X1_2=0.2 * np.random.rand(clusterN,3)   # class '0'
X1_2 += [1,1,0]
X2_1=0.2 * np.random.rand(clusterN,3)   # class '+1'
X2_1 += [1,0,0]
X2_2=0.2 * np.random.rand(clusterN,3)   # class '+1'
X2_2 += [0,1,0]
X = np.vstack ([X1_1,X1_2,X2_1,X2_2])
X[:,2] = 1 #bias
Y=np.ones (C * 2 * clusterN, dtype = int)
Y[0:K*clusterN] = 0
XTrain = X
yTrain = Y

W= np.random.rand(C*K,3)

minX = min(X[:,0])
maxX = max(X[:,0])
minY = min(X[:,1])
maxY = max(X[:,1])

(N, dim) = XTrain.shape

fig = plt.figure()
metadata = dict(title='TwoClassOnlineLinearSvmMultiprototypeBinaryXor', artist='Matplotlib',
                comment='Movie support!')
pillowWriter = animation.writers['pillow']
moviewriter = pillowWriter(fps=2, metadata=metadata)
moviewriter.setup(fig=fig, outfile='TwoClassOnlineLinearSvmMultiprototypeBinaryXor.gif', dpi=150)


for i in tqdm(range(maxEpoch)):
    randperm = np.random.permutation(N)
    for j in randperm:
        x = XTrain[j, :]
        t = yTrain[j]
        z = W @ x
        zz = z.copy()
        zz[t * K: (t + 1) * K] = -np.inf
        q = np.argmax(zz)    # index of max from other classes

        zz = z.copy()
        zz[0:t * K] = -np.inf
        zz[(t + 1) * K:] = -np.inf
        p = np.argmax(zz)   # index of max from this class

        if (z[p] - z[q] < margin or (i==maxEpoch-1 and j == randperm[-1])):
            plt.cla()
            xcntr = list(np.arange (-3, 3, 0.01) )
            ycntr = list(np.arange (-3, 3, 0.01))
            X, Y = np.meshgrid(xcntr, ycntr)
            ZZ = np.zeros([C*K, len(ycntr), len (xcntr)])
            for ii in range (C):
                for jj in range(K):
                    idx = ii * K + jj
                    ZZ[idx,:,:] = W[idx, 0] * X + W[idx, 1] * Y + W[idx, 2]
            Z = np.argmax(ZZ, axis = 0) // K
            cs = plt.contourf (X, Y, Z, levels = [-0.5, 0.5, 1.5], colors = ['royalblue', 'salmon'])
            plt.contour(X, Y, Z, levels=[0,1])

            plt.plot(X1_1[:, 0], X1_1[:, 1], 'b.')
            plt.plot(X1_2[:, 0], X1_2[:, 1], 'b.')
            plt.plot(X2_1[:, 0], X2_1[:, 1], 'r.')
            plt.plot(X2_2[:, 0], X2_2[:, 1], 'r.')

            if t == 0:
                drwstr = 'b*'
            else:
                drwstr = 'r*'

            plt.plot(x[0], x[1], drwstr, markersize=20)

            plt.axis([-3, 3, -3, 3])

            plt.show(block = False)
            moviewriter.grab_frame()

        W *= (1 - learning_rate * weight_decay)
        if (z[p] - z[q] < margin):
            #print (z[p] - z[q])
            print (np.linalg.norm(x), np.linalg.norm(W[p,:]))
            W[p,:]  += learning_rate * x
            W[q, :] -= learning_rate * x

moviewriter.finish()


