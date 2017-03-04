import numpy as np

def normalize(array):
	return (array - array.mean()) / array.std()

