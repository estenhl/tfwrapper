import os
import pytest
import numpy as np

from tfwrapper.models import TransferLearningModel
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.models.nets import SingleLayerNeuralNet

from fixtures import tf
from utils import curr_path, remove_dir

def test_save_nonserializable_param(tf):
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'save-non-serializable')

    try:
        os.mkdir(folder)
        feature_model = FrozenInceptionV3()
        prediction_model = SingleLayerNeuralNet([2048], 1, 2)
        model = TransferLearningModel(feature_model, prediction_model)

        with tf.Session(graph=prediction_model.graph) as sess:
            sess.run(tf.global_variables_initializer())
            prediction_model._checkpoint_variables(sess)
            print('Initializing: %s' % [str(v.name) for v in tf.global_variables()])

        model.save(path, labels=np.asarray(['label1']))

    finally:
        remove_dir(folder)
