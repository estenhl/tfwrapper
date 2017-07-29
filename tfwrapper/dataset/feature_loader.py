import os
import numpy as np
import pandas as pd

from tfwrapper.logger import logger
from tfwrapper.tfsession import TFSession
from tfwrapper.dataset import ImageLoader
from tfwrapper.dataset import ImagePreprocessor
from tfwrapper.utils.files import parse_features
from tfwrapper.utils.files import write_features


class FeatureLoader(ImageLoader):
    sess = None

    def __init__(self, model, layer=None, cache=None, preprocessor=None, sess=None):
        if preprocessor is None:
            preprocessor = ImagePreprocessor()
        
        super().__init__(preprocessor=preprocessor)
        self.model = model
        self.layer = layer

        self.cache = None
        self.features = pd.DataFrame(columns=['filename', 'label', 'features'])
        if cache:
            self.cache = cache
            self.features = parse_features(cache)

        self.sess = sess

    def load(self, img, name=None, label=None):
        if label is not None and type(label) is np.ndarray:
            label = np.argmax(label)

        names = self.preprocessor.get_names(img, name)

        features = []
        records = []

        with TFSession(self.sess, self.model.graph) as sess:
            for i in range(len(names)):
                if self.cache is not None and names[i] in self.features['filename'].values:
                    logger.debug('Skipping %s' % names[i])
                    vector = self.features[self.features['filename'] == names[i]]['features']
                    vector = np.asarray(vector)[0]
                    features.append(vector)
                else:
                    logger.debug('Extracting features for %s' % names[i])

                    imgs, names = super().load(img, name, label=label)

                    if self.layer is None:
                        vector = self.model.extract_bottleneck_features(imgs[i], sess=sess)
                    else:
                        vector = self.model.extract_features(imgs[i], layer=self.layer, sess=sess)
                    features.append(vector)
                    record = {'filename': names[i], 'features': vector}

                    if label is not None:
                        record['label'] = label

                    records.append(record)
                    
                    if self.cache is not None:
                        self.features = self.features.append(record, ignore_index=True)

            if self.cache and len(records) > 0:
                write_features(self.cache, records, append=os.path.isfile(self.cache))

        return features, names