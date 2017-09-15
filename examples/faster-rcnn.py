from tfwrapper.datasets import FDDB

dataset = FDDB(size=10)
dataset = dataset.translated_labels()
dataset.visualize(num=3)