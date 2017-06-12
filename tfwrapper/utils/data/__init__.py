from tfwrapper.utils.exceptions import raise_exception

def get_all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in get_all_subclasses(s)]

def get_subclass_with_name(baseclass, name):
    for subclass in get_all_subclasses(baseclass):
        if subclass.__name__ == name:
            return subclass

    raise_exception('%s has no subclass named %s. (Valid is %s)' % (baseclass.__name__, classname, str([s.name for s in cls.__subclasses__()])))
