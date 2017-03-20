def batch_data(data, batch_size):
	batches = []

	for i in range(0, int(len(data) / batch_size) + 1):
		start = (i * batch_size)
		end = min((i + 1) * batch_size, len(data))
		if start < end:
			batches.append(data[start:end])

	return batches