class DatasetGenerator():
    def __init__(self, dataset, batch_size, normalize=False, shuffle=False, infinite=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.normalize = normalize
        self.shuffle = shuffle
        self.infinite = infinite
        self.cursor = 0

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self):
            if not self.infinite:
                raise StopIteration

            self.cursor = 0

        dataset, self.cursor = self.dataset.next_batch(self.cursor, self.batch_size)

        if self.normalize:
            dataset = dataset.normalize()

        if self.shuffle:
            dataset = dataset.shuffle()

        return dataset.X, dataset.y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, val):
        return self.__class__(self.dataset[val], self.batch_size, normalize=self.normalize, shuffle=self.shuffle,
                              infinite=self.infinite)
