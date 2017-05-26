import tensorflow as tf

from tfwrapper import logger

def get_variable_by_name(name, append_tensor_id=True):
    if append_tensor_id:
        name += ':0'

    variables = [v for v in tf.global_variables() if v.name == name]

    if len(variables) == 0:
        logger.warning('Looking up non-existing variable with name %s' % name)
        return None
    elif len(variables) > 1:
        logger.warning('Found several variables named %s, returning the first' % name)
        
    return variables[0]
