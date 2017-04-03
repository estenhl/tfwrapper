import os

def parse_features(src, delimiter=','):
	all_features = {}

	if os.path.isfile(src):
		with open(src, 'r') as f:
			for line in f.readlines():
				tokens = line.split(delimiter)

				name = tokens[0]
				features = [float(x) for x in tokens[1:]]

				all_features[name] = features

	return all_features

def write_features(all_features, output_file, delimiter=','):
	with open(output_file, 'w') as f:
		for name, features in all_features.items():
			f.write(name + delimiter + delimiter.join([str(x) for x in features]) + '\n')