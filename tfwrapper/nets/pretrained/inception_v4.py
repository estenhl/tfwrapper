import tensorflow as tf
import cv2
import os
from tfwrapper import twimage
from tfwrapper.nets.pretrained.pretrained_model import PretrainedModel

INCEPTION_PB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','inception_v4.pb')

FEATURE_LAYER = "InceptionV4/Logits/PreLogitsFlatten/Reshape:0"
SUBFEATURES_LAYER = "InceptionV4/InceptionV4/Mixed_7d/concat:0"
PREDICTIONS = "InceptionV4/Logits/Predictions:0"


class Inception_v4(PretrainedModel):
    def __init__(self, graph_file):
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