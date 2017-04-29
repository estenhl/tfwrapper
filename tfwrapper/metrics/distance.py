import numpy as np 

def euclidean(p1, p2):
	return np.sqrt(np.sum((p1-p2)**2))