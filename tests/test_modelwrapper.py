import pytest

from tfwrapper.models import ModelWrapper, RegressionModelWrapper, ClassificationModelWrapper
from tfwrapper.utils.exceptions import InvalidArgumentException

from mock import MockBaseModel, MockRegressionModel, MockClassificationModel
from utils import softmax_wrapper


def test_from_instance_regression():
    model = MockRegressionModel()
    wrapper = ModelWrapper.from_instance(model)

    assert isinstance(wrapper, RegressionModelWrapper), 'Wrapping a regression model does not yield a RegressionModelWrapper'


def test_from_instance_classification():
    model = MockClassificationModel()
    wrapper = ModelWrapper.from_instance(model)

    assert isinstance(wrapper, ClassificationModelWrapper), 'Wrapping a classification model does not yield a ClassificationModelWrapper'


def test_from_instance_invalid():
    exception = False
    try:
        ModelWrapper.from_instance(MockBaseModel([1, 2], 2, [softmax_wrapper()]))
    except InvalidArgumentException:
        exception = True

    assert exception, 'Wrapping an invalid type (BaseModel) does not raise an exception'


def test_graph():
    model = MockRegressionModel()
    wrapper = RegressionModelWrapper(model)
    model._graph = 'Test'

    assert model.graph == wrapper.graph


def test_variables():
    model = MockRegressionModel()
    wrapper = RegressionModelWrapper(model)
    model._variables = 'Test'

    assert model.variables == wrapper.variables