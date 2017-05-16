# [TODO: Model]()
Proposed abstract top-level class of the model hierarchy.
##### Proposed interface:
	__init__(name: str)
	predict(X: np.ndarray, y: np.ndarray, epochs: int)
	validate(X: np.ndarray, y: np.ndarray, epochs: int)
## [SupervisedModel](https://github.com/epigramai/tfwrapper/blob/master/tfwrapper/supervisedmodel.py)
Abstract parent for trainable models
##### Interface:
	__init__(tons of shit at the moment)
	train(X: np.ndarray = None, y: np.ndarray = None, generator: generator = None, epochs: int)
	""" Requires either X and y or a generator """
	predict(X: np.ndarray, y: np.ndarray, epochs: int)
	validate(X: np.ndarray, y: np.ndarray, epochs: int)
	save(path: str)
	load(path: str)
### [TODO: Regression]()
#### [TODO: Linear Regression]()
### [Neural Nets](https://github.com/epigramai/tfwrapper/blob/master/tfwrapper/nets/neural_net.py)
#### [Single Layer Neural Net](https://github.com/epigramai/tfwrapper/blob/master/tfwrapper/nets/single_layer_neural_net.py)
#### [Dual Layer Neural Net](https://github.com/epigramai/tfwrapper/blob/master/tfwrapper/nets/dual_layer_neural_net.py)
### [CNNs](https://github.com/epigramai/tfwrapper/blob/master/tfwrapper/nets/neural_net.py)
#### [ShallowCNN](https://github.com/epigramai/tfwrapper/blob/master/tfwrapper/nets/shallow_cnn.py)
Example CNN
#### [SqueezeNet](https://github.com/epigramai/tfwrapper/blob/master/tfwrapper/nets/squeezenet.py)
[https://arxiv.org/abs/1602.07360](https://arxiv.org/abs/1602.07360)
#### [VGG16](https://github.com/epigramai/tfwrapper/blob/master/tfwrapper/nets/vgg16.py)
[https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
#### [TODO: QuickNet]()
[https://arxiv.org/abs/1701.02291](https://arxiv.org/abs/1701.02291) 
#### [TODO: DenseNet161]()
[https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
## [TODO: FrozenModel]()
Proposed concrete parent class for frozen (.pb) models
##### Proposed interface:
	__init__(path: str, name: str)
	predict(X: np.ndarray, y: np.ndarray, epochs: int)
	validate(X: np.ndarray, y: np.ndarray, epochs: int)
### [TODO: FrozenInceptionV3]()
### [TODO: FrozenInceptionV4]()
## [TODO: EnsembleModel]()
Proposed concrete parent class for combining models
##### Proposed interface:
	__init__(models: [Model], name: str)
	predict(X: np.ndarray, y: np.ndarray, epochs: int, method: ['accumulate', 'votes', 'majority'])
	validate(X: np.ndarray, y: np.ndarray, epochs: int, method: ['accumulate', 'votes', 'majority'])

