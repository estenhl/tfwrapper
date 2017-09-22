import cv2
import numpy as np
import tensorflow as tf

def parse_tfrecord(path):
    iterator = tf.python_io.tf_record_iterator(path)

    features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
    }

    with tf.Session() as sess:
        for entry in iterator:
            obj = tf.parse_single_example(entry, features=features)
            img = sess.run(obj['image/encoded'])
            path = sess.run(obj['image/filename'])
            print(path)
            height = sess.run(obj['image/height'])
            width = sess.run(obj['image/width'])

            ymins = sess.run(obj['image/object/bbox/ymin'])
            xmins = sess.run(obj['image/object/bbox/xmin'])
            ymaxs = sess.run(obj['image/object/bbox/ymax'])
            xmaxs = sess.run(obj['image/object/bbox/xmax'])


            img = np.fromstring(img, dtype=np.uint8)
            img = cv2.imdecode(img, 1)

            for i in range(len(ymins.values)):
                ymin = int(ymins.values[i] * height)
                xmin = int(xmins.values[i] * width)
                ymax = int(ymaxs.values[i] * height)
                xmax = int(xmaxs.values[i] * width)
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            #img = np.reshape(img, (height, width, 3))
            cv2.imshow('Hei', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

import sys
if __name__ == '__main__':
    parse_tfrecord(sys.argv[1])