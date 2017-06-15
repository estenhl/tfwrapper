import numpy as np
from abc import ABC, abstractmethod
from collections import Counter

from tfwrapper import logger
from tfwrapper.utils.exceptions import raise_exception


class DatasetGeneratorBase(ABC):
    def __init__(self, dataset, batch_size, normalize=False, shuffle=True, infinite=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.normalize = normalize
        self.shuffle = shuffle
        self.infinite = infinite
        self.cursor = 0
        
        if self.shuffle:
            # Pre-shuffle the dataset (makes splitting an unused generator more healthy)
            self.dataset = self.dataset.shuffled()

    def get_base_length(self):
        """
        Gets the base generator length. This may differ from len() in generators upsampling or downsampling the dataset.
        :return: The length of the underlying dataset this generator is based on.
        """
        return self.__len__()

    @abstractmethod
    def _next_batch(self):
        raise NotImplementedError('DatasetGeneratorBase is a generic class')

    def _start(self):
        self.cursor = 0
        if self.shuffle:
            self.dataset = self.dataset.shuffle()

    def __iter__(self):
        self._start()
        return self

    def __next__(self):
        if self.cursor >= len(self):
            if not self.infinite:
                raise StopIteration
            self._start()

        batch_X, batch_y = self._next_batch()

        if self.normalize:
            # TODO (08.06.17): change this to dataset.normalize_array once the circular dependency issue is solved
            batch_X = (batch_X - batch_X.mean()) / batch_X.std()

        return batch_X, batch_y

    def __len__(self):
        """
        :return: The length of the generator output
        """
        return len(self.dataset)

    def __getitem__(self, val):
        return self.__class__(self.dataset[val], self.batch_size, normalize=self.normalize, shuffle=self.shuffle,
                              infinite=self.infinite)


class DatasetGenerator(DatasetGeneratorBase):
    def __init__(self, dataset, batch_size, normalize=False, shuffle=True, infinite=False):
        super().__init__(dataset, batch_size, normalize=normalize, shuffle=shuffle, infinite=infinite)

    def _next_batch(self):
        dataset, self.cursor = self.dataset.next_batch(self.cursor, self.batch_size)
        return dataset.X, dataset.y


class DatasetSamplingGenerator(DatasetGeneratorBase):
    """
    Samples classes of the dataset so that the number of samples for each class is equal per epoch.
    In comparison to balancing the dataset this keeps the effective learning rate equal for each label.
    At the cost of weighing each particular sample differently.
    upsampling_factor scales where between the largest and smallest class the target number of samples is set. Ie:
     1.0   - upsampling to the largest class by randomly picked repeat samples
     0.5   - up/downsampling to a sample number between the largest and smallest labels.
     0.0   - downsampling to the smallest class by randomly picking within the too large labels.
    Note that this downsampling differs from pre-balancing the dataset as the specific samples will vary per epoch.
    """
    def __init__(self, dataset, batch_size, normalize=False, shuffle=True, infinite=False, upsampling_factor=1.0):
        super().__init__(dataset, batch_size, normalize=normalize, shuffle=shuffle, infinite=infinite)
        if upsampling_factor < 0.:
            logger.warning('Upsampling_factor must be positive. Defaulting to 1')
            upsampling_factor = 1.

        # Get the dataset labels on a common format so that we can look at their frequency
        if len(dataset.y.shape) > 1 and dataset.y.shape[-1] > 1:
            # Dataset is one-hot.
            self.y = np.asarray([np.argmax(dataset.y_) for dataset.y_ in dataset.y])
        else:
            self.y = dataset.y

        counts = Counter(self.y)
        self.n_labels = len(counts)
        n_largest_label = max([counts[x] for x in counts])
        n_smallest_label = min([counts[x] for x in counts])
        self.upsampling_factor = upsampling_factor
        self.per_label_size = int((n_largest_label - n_smallest_label) * self.upsampling_factor + n_smallest_label)
        # Get indexes for each label
        self.label_indexes = [np.where(self.y == index)[0] for index in counts.keys()]
        self.indexes = np.zeros((self.__len__()), dtype=np.int64)
        self.initialized = False
        if not self.shuffle:
            # Populate the indexes only once. Same sequence every loop.
            self._populate_indexes()

    def get_base_length(self):
        return len(self.dataset)

    def _populate_indexes(self):
        # Populate indexes from our classes until we reach the desired per_label_size
        fill_ptr = 0
        for label in range(self.n_labels):
            label_picks_left = self.per_label_size
            while label_picks_left > 0:
                # Pick without reuse to ensure all samples are present (if possible)
                picks = np.random.choice(self.label_indexes[label], min(len(self.label_indexes[label]),
                                                                        label_picks_left), replace=False)
                self.indexes[fill_ptr:fill_ptr + len(picks)] = picks
                fill_ptr += len(picks)
                label_picks_left -= len(picks)
        # Shuffle index as while indexes within each class are randomly ordered, the classes themselves are in order.
        np.random.shuffle(self.indexes)
        self.initialized = True

    def _start(self):
        self.cursor = 0
        if self.shuffle or not self.initialized:
            self._populate_indexes()

    def _next_batch(self):
        batch_indexes = self.indexes[self.cursor:min(self.cursor + self.batch_size, len(self.indexes))]
        batch_X = self.dataset.X[batch_indexes]
        batch_y = self.dataset.y[batch_indexes]
        self.cursor += len(batch_X)
        return batch_X, batch_y

    def __len__(self):
        return self.per_label_size * self.n_labels

    def __getitem__(self, val):
        # Note: this fetches from the original dataset, not the upsampled one. This is intentional to prevent duplicates
        # in supposedly independent folds.
        return self.__class__(self.dataset[val], self.batch_size, normalize=self.normalize, shuffle=self.shuffle,
                              infinite=self.infinite, upsampling_factor=self.upsampling_factor)


class GeneratorWrapper(DatasetGeneratorBase):
    def __init__(self, dataset, batch_size=1, normalize=False, shuffle=True, infinite=False):
        self.generator = dataset

        self.normalize = normalize
        self.shuffle = shuffle
        self.infinite = infinite

        if self.normalize:
            logger.warning('Unable to normalize when wrapping a built-in python generator')
        if self.shuffle:
            logger.warning('Unable to shuffle when wrapping a built-in python generator')
        if self.infinite:
            logger.warning('Unable to set infinite when wrapping a built-in python generator')

    def get_base_length(self):
        raise_exception('Unable to get length when wrapping a built-in python generator', TypeError)

    def _next_batch(self):
        raise_exception('Unable to get next batch when wrapping a built-in python generator', TypeError)

    def __iter__(self):
        return self.generator

    def __len__(self):
        raise_exception('Unable to get length when wrapping a built-in python generator', TypeError)


    def __getitem__(self, val):
        raise_exception('Unable to get item when wrapping a built-in python generator', TypeError)
