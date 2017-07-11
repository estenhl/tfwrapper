import os
import tensorflow as tf
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.models.frozen import FrozenInceptionV4
from tfwrapper.models.frozen import FrozenResNet152
from tfwrapper import config

base_path = config.SERVING_MODELS

incv3_path = os.path.join(base_path, 'incv3_2048')
incv3_version = '1'

incv4_path = os.path.join(base_path, 'incv4_1536')
incv4_version = '1'

res152_path = os.path.join(base_path, 'res152_2048')
res152_version = '1'

with tf.Session() as sess:
    model = FrozenInceptionV3(sess=sess)
    model.save_serving(os.path.join(incv3_path, incv3_version), sess, over_write=True)

with tf.Session() as sess:
    model = FrozenInceptionV4(sess=sess)
    model.save_serving(os.path.join(incv4_path, incv4_version), sess, over_write=True)

with tf.Session() as sess:
    model = FrozenResNet152(sess=sess)
    model.save_serving(os.path.join(res152_path, res152_version), sess, over_write=True)



