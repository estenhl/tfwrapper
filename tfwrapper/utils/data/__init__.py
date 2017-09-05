import json

from tfwrapper import logger
from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.exceptions import InvalidArgumentException


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