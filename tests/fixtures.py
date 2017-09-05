from pytest import fixture

@fixture
def tf():
    import tensorflow
    tensorflow.reset_default_graph()
    return tensorflow
