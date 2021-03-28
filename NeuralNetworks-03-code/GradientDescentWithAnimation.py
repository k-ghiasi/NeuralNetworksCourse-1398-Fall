import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import numpy as np
import math

mpl.rcParams['figure.dpi']= 300

x0 = 0
y0 = 0
a = 1
b = 1/10
s = 1/a ** 2
t = 1/b ** 2
theta = np.pi / 4
R = np.array([[math.cos(theta),  math.sin(theta)],
              [-math.sin(theta), math.cos(theta)]])
S = np.array([[s, 0],
              [0, t]])
M = np.matmul(np.matmul(np.transpose(R),S), R)


#-------------------

T = 100
x = np.zeros([T])
y = np.zeros([T])

x[0] = -2
y[0] = 3

alpha = 0.005

for i in range (1,T):
    x[i] = x[i-1] - alpha * (2*M[0,0]*x[i-1] + (M[0,1]+M[1,0])*y[i-1])
    y[i] = y[i-1] - alpha * (2*M[1,1]*y[i-1] + (M[0,1]+M[1,0])*x[i-1])

fig = plt.figure()     
ax = plt.subplot(111, aspect='equal')
for scale in np.arange (10):
    width = scale * a
    height = scale * b
    e = Ellipse((x0, y0), width, height, theta * 180 / np.pi
           , facecolor = 'none', edgecolor = 'r')
    ax.add_artist(e)

plt.xlim(-4,4)
plt.ylim(-4,4)

metadata = dict(title='GradientDescent', artist='Matplotlib',
                comment='Movie support!')
pillowWriter = animation.writers['pillow']
moviewriter = pillowWriter(fps=10, metadata=metadata)
moviewriter.setup(fig=fig, outfile='GradientDescent.gif', dpi=300)

for i in range (1,T):
    plt.plot(x[i-1:i],y[i-1:i], 'k.')
    plt.show(block=False)
    moviewriter.grab_frame()

moviewriter.finish()

print(x[T-1])
print(y[T-1])

