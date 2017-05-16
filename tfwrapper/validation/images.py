import os

from tfwrapper import ImageLoader
from tfwrapper import ImageDataset
from tfwrapper import ImagePreprocessor
from tfwrapper.nets import ShallowCNN

from .validation import Validator
from .validation import kfold_validation

# Imagesizes are on the form (x, y)
def kfold_validation_imagesize(dataset, image_sizes, cls=ShallowCNN, k=10, epochs=10, bw=False):
    validator = Validator([str(s) for s in image_sizes])
    for image_size in image_sizes:
        preprocessor = ImagePreprocessor()
        preprocessor.resize_to = image_size

        if bw:
            preprocessor.bw = True
        
        dataset.loader = ImageLoader(preprocessor)
        create_model = lambda x, sess: cls([image_size[1], image_size[0], 3], 2, name='Val%dx%d_%d' % (image_size[0], image_size[1], x), sess=sess)
        
        validator = kfold_validation(dataset, create_model, k=k, epochs=epochs, validator=validator)

    return validator

def kfold_validation_bw(dataset, k=10, epochs=10):
    validator = Validator(['colours', 'bw'])
    for val in [False, True]:
        preprocessor = ImagePreprocessor()
        preprocessor.bw = val
        dataset.loader = ImageLoader(preprocessor)
        create_model = lambda x, sess: ShallowCNN([image_size[0], image_size[1], 3], 2, name='Val%dx%d_%d' % (image_size[0], image_size[1], x), sess=sess)
            
        validator = kfold_validation(dataset, create_model, k=k, epochs=epochs, validator=validator)

    return validator
