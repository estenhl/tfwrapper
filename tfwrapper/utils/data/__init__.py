import json
import tensorflow as tf

from tfwrapper import logger

from tfwrapper.utils.exceptions import InvalidArgumentException, log_and_raise, raise_exception

from .tfrecord import parse_tfrecord


def get_all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in get_all_subclasses(s)]


def get_subclass_by_name(baseclass, name):
    for subclass in get_all_subclasses(baseclass):
        if subclass.__name__ == name:
            return subclass

    raise_exception('%s has no subclass named %s. (Valid is %s)' % (baseclass.__name__, name, str([s.__name__ for s in baseclass.__subclasses__()])), InvalidArgumentException)


def ensure_serializable(data):
    for key in data:
        try:
            json.dumps(data[key])
        except TypeError:
            logger.warning('Unable to serialize parameter with keyword \'%s\' and type %s. Force-casting to str' % (key, repr(type(data[key]))))
            data[key] = str(data[key])

    return data


def create_tfrecord_feature(element):
    if type(element) == int:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[element]))
    elif type(element) == bytes:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[element]))
    elif type(element) == float:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[element]))
    elif type(element) is list:
        if len(element) == 0 or type(element[0]) == int:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=element))
        elif type(element[0]) == bytes:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=element))
        elif type(element[0]) == float:
            return tf.train.Feature(float_list=tf.train.FloatList(value=element))

    if type(element) is list:
        raise log_and_raise(InvalidArgumentException, 'Invalid tfrecord datatype [%s]. (Valid is [\'int\', \'bytes\', \'float\'] and lists containing these types)' % type(element[0]))
    else:
        raise log_and_raise(InvalidArgumentException, 'Invalid tfrecord datatype %s. (Valid is [\'int\', \'bytes\', \'float\'] and lists containing these types)' % type(element))
