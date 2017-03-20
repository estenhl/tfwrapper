import numpy as np

from tfwrapper.utils.data import batch_data

def test_batch_data():
	data = np.ones(77)
	batches = batch_data(data, 10)

	assert 8 == len(batches)
	assert 7 == len(batches[-1])