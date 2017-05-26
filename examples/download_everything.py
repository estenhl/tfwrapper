from tfwrapper.datasets import wine
from tfwrapper.datasets import mnist
from tfwrapper.datasets import cifar10
from tfwrapper.datasets import cifar100
from tfwrapper.datasets import flowers
from tfwrapper.datasets import cats_and_dogs
from tfwrapper.nets.pretrained import InceptionV3
from tfwrapper.nets.pretrained import InceptionV4

cats_and_dogs(size=1)
mnist(size=1)
cifar10(size=1)
cifar100(size=1)
wine(size=1)
flowers(size=1)

InceptionV3()
InceptionV4()