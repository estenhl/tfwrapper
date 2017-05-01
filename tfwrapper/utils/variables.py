import tensorflow as tf

def get_variable_by_name(name, append_tensor_id=True):
	if append_tensor_id:
		name += ':0'

	return [v for v in tf.global_variables() if v.name == name][0]
