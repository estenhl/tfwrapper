import tensorflow as tf
import cv2
import os
from tfwrapper import twimage
from tfwrapper.nets.pretrained.pretrained_model import PretrainedModel
from tfwrapper.utils.download import google_drive
from tfwrapper import config

INCEPTION_PB_PATH = os.path.join(config.MODELS,'inception_v4.pb')

FEATURE_LAYER = "InceptionV4/Logits/PreLogitsFlatten/Reshape:0"
SUBFEATURES_LAYER = "InceptionV4/InceptionV4/Mixed_7d/concat:0"
PREDICTIONS = "InceptionV4/Logits/Predictions:0"

DOWNLOAD_ID = '0B1b2bIlebXOqN3JWdHRZc05xdzQ'

class Inception_v4(PretrainedModel):
    FEATURE_LAYER = "InceptionV4/Logits/PreLogitsFlatten/Reshape:0"

    def __init__(self, graph_file=INCEPTION_PB_PATH):
        #self.download_if_necessary()

        with tf.gfile.FastGFile(graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

            PretrainedModel.__init__(self, tf.get_default_graph())
            tf.reset_default_graph()

    def get_feature(self, img, sess=None, layer=FEATURE_LAYER):
        if not sess:
            sess = tf.Session(graph=self.graph)

        tensor = sess.graph.get_tensor_by_name(layer)
        try:
            feature = sess.run(tensor, {'input:0': img})

            return feature[0]

        except Exception as e:
            print(e)
            print('Unable to get feature')

            return None

    def get_feature_from_file(self, image_file, sess=None, layer=FEATURE_LAYER):
        if not sess:
            sess = tf.Session(graph=self.graph)

        tensor = sess.graph.get_tensor_by_name(layer)
        try:
            # image_data = tf.gfile.FastGFile(image_file, 'rb').read()
            img = twimage.imread(image_file)
            feature = sess.run(tensor, {'input:0': img})

            return feature

        except Exception as e:
            print(e)
            print('Unable to get feature for ' + str(image_file))

            return None

    def download_if_necessary(self, path=INCEPTION_PB_PATH):
        if not os.path.isfile(path):
            print("Downloading Inception_v4.pb")
            google_drive.download_file_from_google_drive(DOWNLOAD_ID, path)
            print("Completed downloading Inception_v4.pb")