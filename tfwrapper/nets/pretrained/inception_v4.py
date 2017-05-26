import tensorflow as tf
import cv2
import os

from tfwrapper import config
from tfwrapper import logger
from tfwrapper import twimage
from tfwrapper import TFSession
from tfwrapper.utils.download import google_drive

from .pretrained_model import PretrainedModel

INCEPTION_PB_PATH = os.path.join(config.MODELS,'inception_v4.pb')

SUBFEATURES_LAYER = "InceptionV4/InceptionV4/Mixed_7d/concat:0"
PREDICTIONS = "InceptionV4/Logits/Predictions:0"

DOWNLOAD_ID = '0B1b2bIlebXOqN3JWdHRZc05xdzQ'

class InceptionV4(PretrainedModel):
    DEFAULT_INPUT_LAYER = 'input:0'
    DEFAULT_FEATURES_LAYER = 'InceptionV4/Logits/PreLogitsFlatten/Reshape:0'

    def __init__(self, graph_file=INCEPTION_PB_PATH):
        self.download_if_necessary(graph_file)

        with tf.gfile.FastGFile(graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

            super().__init__(tf.get_default_graph())

    def run_op(self, to_layer, from_layer, data, sess=None):
        with TFSession(sess, self.graph) as sess:
            logger.debug('Extracting features from layer ' + from_layer + ' to ' + to_layer)
            to_tensor = self.graph.get_tensor_by_name(to_layer)   

            try:  
                feature = sess.run(to_tensor,{from_layer: data})

                return feature[0]
            except Exception as e:
                logger.warning('Unable to extract feature')
                logger.warning(str(e))
                raise e

    def extract_features(self, img,  layer=DEFAULT_FEATURES_LAYER, sess=None):
        with TFSession(sess, self.graph) as sess:
            return self.run_op(layer, self.DEFAULT_INPUT_LAYER, img ,sess=sess)

    def extract_features_from_file(self, image_file, layer=DEFAULT_FEATURES_LAYER, sess=None):
        with TFSession(sess, self.graph) as sess:
            try:
                img = twimage.imread(image_file)
                feature = self.run_op(layer, self.DEFAULT_INPUT_LAYER, img, sess=sess)

                return feature

            except Exception as e:
                logger.warning('Unable to read image %s' % str(image_file))
                logger.warning(str(e))

                return None

    def download_if_necessary(self, path=INCEPTION_PB_PATH):
        if not os.path.isfile(path):
            logger.info('Downloading Inception_v4.pb')
            google_drive.download_file_from_google_drive(DOWNLOAD_ID, path)
            logger.info('Completed downloading Inception_v4.pb')