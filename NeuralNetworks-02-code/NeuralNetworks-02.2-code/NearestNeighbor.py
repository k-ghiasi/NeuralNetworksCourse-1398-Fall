"""

Author: Sayed Kamaledin Ghiasi-Shirazi
Year:   2019

"""

import scipy.io as sio
import numpy as np
from tqdm import tqdm

def NearestNeighbor (XTrain, yTrain, XTest, yTest):
	score = 0
	#for n in tqdm(range(XTest.shape[0])):
	for n in range(XTest.shape[0]):
		x = XTest[n,:]
		x = np.array(x,dtype=float, ndmin = 2)
		xrepeated = np.repeat(x, XTrain.shape[0], axis = 0)
		d = XTrain - xrepeated
		d **= 2
		d = np.sum(d, axis = 1)
		idx = np.argmin(d)
		predicted_label = yTrain[idx]
		if ( yTest[n] == predicted_label):
			score = score + 1
			
	acc = score / XTest.shape[0]
	return acc
