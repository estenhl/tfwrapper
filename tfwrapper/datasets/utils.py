import os
import pickle

from tfwrapper import config

curr_path = config.DATASETS

def setup_structure(name, create_data_folder=True):
    root_path = os.path.join(curr_path, name)
    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    data_path = os.path.join(root_path, 'data')
    if create_data_folder and not os.path.isdir(data_path):
        os.mkdir(data_path)

    labels_file = os.path.join(root_path, 'labels.txt')

    return root_path, data_path, labels_file

def recursive_delete(path, skip=[]):
    ret_val = True
    if os.path.isdir(path):
        for filename in os.listdir(path):
            ret_val = recursive_delete(os.path.join(path, filename), skip=skip) and ret_val


        if ret_val:
            os.rmdir(path)

        return ret_val
    elif os.path.isfile(path) and path not in skip:
        os.remove(path)

        return True

    return False

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict