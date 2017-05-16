import tensorflow as tf


def resize(img_size, method=tf.image.ResizeMethod.BILINEAR, name='reshape'):
    def create_layer(x):
        size = tf.Variable(img_size, trainable=False, name=name + '/shape')

        return tf.image.resize_images(x, img_size, method=method, name=name)

    return create_layer


def normalize_image(name='image_normalization'):
    return lambda x: tf.map_fn(lambda img: tf.image.per_image_standardization(img), x, name=name)


def normalize_batch(name='batch_normalization'):
    raise NotImplementedError('Batch normalization is not implemented')


def flip_up_down(seed=None):
    return lambda x: tf.map_fn(lambda img: tf.image.random_flip_up_down(img, seed=seed), x, name=name)


def flip_left_right(seed=None):
    return lambda x: tf.map_fn(lambda img: tf.image.random_flip_left_right(img, seed=seed), x, name=name)


def brightness(max_delta, seed=None):
    return lambda x: tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta, seed=seed), x, name=name)


def contrast(lower, upper, seed=None):
    return lambda x: tf.map_fn(lambda img: tf.image.random_contrast(img, lower, upper, seed=seed), x, name=name)


def hue(max_delta=0.5, seed=None):
    return lambda x: tf.map_fn(lambda img: tf.image.random_hue(img, max_delta, seed=seed), x, name=name)


def saturation(lower, upper, seed=None):
    return lambda x: tf.map_fn(lambda img: tf.image.random_saturation(img, lower, upper, seed=seed), x, name=name)
