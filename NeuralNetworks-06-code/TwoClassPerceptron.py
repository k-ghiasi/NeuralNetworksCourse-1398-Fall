import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

halfN = 10
X1=np.random.rand(halfN,3)   # class '-1'
X2=np.random.rand(halfN,3)   # class '+1'

X1[:,1] /= 2
X1[:,2] = 1 # bias

X2[:,1] += X2[:,0] + 1
X2[:,2] = 1 # bias
X = np.vstack ([X1,X2])

Y=np.ones (2 * halfN)
Y[0:halfN] *= -1

XTrain = X
yTrain = Y

w= np.array([10, 0.1, 0.0], dtype = 'float')

minX = min(X[:,0])
maxX = max(X[:,0])
minY = min(X[:,1])
maxY = max(X[:,1])

maxEpoch = 40
(N, dim) = XTrain.shape

fig = plt.figure()
metadata = dict(title='TwoClassPerceptron', artist='Matplotlib',
                comment='Movie support!')
pillowWriter = animation.writers['pillow']
moviewriter = pillowWriter(fps=2, metadata=metadata)
moviewriter.setup(fig=fig, outfile='TwoClassPerceptron.gif', dpi=150)


for i in tqdm(range(maxEpoch)):
    randperm = np.random.permutation(N)
    for j in randperm:
        x = XTrain[j, :]
        t = yTrain[j]
        z = w @ x

        y = 1 if z >= 0 else -1
        if (y != t or (i==maxEpoch-1 and j == randperm[-1])):
            plt.cla()
            xcntr = list(np.arange (0, 1, 0.01) )
            ycntr = list(np.arange (0, 2, 0.01))
            X, Y = np.meshgrid(xcntr, ycntr)
            Z     = w[0] * X + w[1] * Y + w[2]
            cs = plt.contourf (X, Y, Z, levels = 0, colors = ['royalblue', 'salmon'])

            plt.plot(X1[:, 0], X1[:, 1], 'b.')
            plt.plot(X2[:, 0], X2[:, 1], 'r.')

            p1 = np.array([0, (-w[0] * 0 - w[2]) / w[1]])
            p2 = np.array([1, (-w[0] * 1 - w[2]) / w[1]])
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k', linewidth=3)

            if t == -1:
                drwstr = 'b*'
            else:
                drwstr = 'r*'

            plt.plot(x[0], x[1], drwstr, markersize=20)

            plt.axis([0, 1, 0, 2])

            plt.show(block = False)
            moviewriter.grab_frame()

        if (y != t):
            w += t * (0.1 * x)

moviewriter.finish()


