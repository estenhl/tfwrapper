import os
import shutil

from tfwrapper import logger

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils


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
