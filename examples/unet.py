from tfwrapper.datasets import VOC2012
from tfwrapper.models.nets import UNet

dataset = VOC2012(size=2000)
dataset = dataset.resized(max_size=(392, 392))
dataset = dataset.squarepadded()
dataset = dataset.framed_X((180, 180))

print('Translating labels')
dataset = dataset.translated_labels()
print(dataset.y.shape)

print('Onehot encoding')
dataset = dataset.onehot_encoded()
print(dataset.y.shape)

print('Labels: ' + str(dataset.labels))

print(dataset.X.shape)
print(dataset.y.shape)
print(dataset.labels)
print(dataset.num_classes)

net = UNet(dataset.num_classes, name='ExampleUNet')
net.train(dataset.X, dataset.y, epochs=100)