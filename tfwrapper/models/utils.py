import os
import shutil
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.layers import initializer
from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.exceptions import IllegalStateException


# https://stackoverflow.com/questions/42858785/connect-input-and-output-tensors-of-two-different-graphs-tensorflow
def combine_graphs(graph1, graph2, *, graph1_out, graph2_in, graph1_name='graph1', graph2_name='graph2'):
    if tf.get_default_session() is not None:
        raise_exception('Unable to combine models when a session is set as default', IllegalStateException)

    graph2_def = graph2.as_graph_def()
    graph1_def = graph1.as_graph_def()

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            graph1out, = tf.import_graph_def(graph1_def, name=graph1_name, return_elements=[graph1_out])
            tf.import_graph_def(graph2_def, name=graph2_name, input_map={graph2_in: graph1out})

            assign_ops = []
            for op in graph.get_operations():
                if op.name.endswith('Assign'):
                    assign_ops.append(graph.get_tensor_by_name(op.name + ':0'))

            init_op = initializer(assign_ops, name='init')

    return graph, init_op


def save_serving(export_path, in_tensor, out_tensor, sess, over_write=False):

    if os.path.isdir(export_path):
        logger.info('Export path: ' + export_path + ' exists already.')
        if over_write:
            logger.info('Over write set, removing old model.')
            shutil.rmtree(export_path)
        else:
            logger.info('Use kwarg over_write to overwrite model.')
            return

    logger.info('Exporting model to ' + str(export_path))

    builder = saved_model_builder.SavedModelBuilder(export_path)

    classification_inputs = utils.build_tensor_info(in_tensor)
    classification_outputs_scores = utils.build_tensor_info(out_tensor)

    classification_signature = signature_def_utils.build_signature_def(
        inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
        outputs={
            signature_constants.CLASSIFY_OUTPUT_SCORES:
                classification_outputs_scores
        },
        method_name=signature_constants.CLASSIFY_METHOD_NAME)

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature
        })

    builder.save()
