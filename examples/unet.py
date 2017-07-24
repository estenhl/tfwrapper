from tfwrapper.datasets import VOC2012
from tfwrapper.models.nets import UNet

dataset = VOC2012(size=500)
dataset = dataset.resized(max_size=(388, 388))
dataset = dataset.squarepadded()
dataset = dataset.framed_X((184, 184))

print('Translating labels')
dataset = dataset.translated_labels()

print('Onehot encoding')
dataset = dataset.onehot_encoded()

net = UNet(dataset.num_classes, name='ExampleUNet')
net.batch_size = 1
net.train(dataset.X, dataset.y, epochs=10)