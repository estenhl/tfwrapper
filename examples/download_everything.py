from tfwrapper.datasets import wine
from tfwrapper.datasets import mnist
from tfwrapper.datasets import cifar10
from tfwrapper.datasets import cifar100
from tfwrapper.datasets import flowers
from tfwrapper.datasets import cats_and_dogs
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.models.frozen import FrozenInceptionV4

cats_and_dogs(size=1)
mnist(size=1)
cifar10(size=1)
cifar100(size=1)
wine(size=1)
flowers(size=1)

FrozenInceptionV3()
FrozenInceptionV4()