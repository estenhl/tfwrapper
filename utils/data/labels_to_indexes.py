import numpy as np

def labels_to_indexes(y):
	labels = []
	indices = []

	for label in y:
		if label not in labels:
			labels.append(label)
		indices.append(labels.index(label))

	return np.asarray(indices), labels