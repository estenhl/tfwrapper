from tfwrapper.datasets import VOC2012

dataset = VOC2012(size=5)
dataset.visualize(2)

print('Resizing to (572, 572)')
dataset = dataset.resized(max_size=(392, 392))
dataset.visualize(2)

print('Padding to squares')
dataset = dataset.squarepadded()
dataset.visualize(2)

print('Adding border')
dataset = dataset.framed_X((180, 180))
dataset.visualize(2)