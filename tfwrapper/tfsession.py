import tensorflow as tf

class TFSession():
    def __init__(self, session=None, graph=None, init=False, variables={}):
        self.is_local_session = session is None
        self.session = session
        
        if session:
            self.graph = session.graph
            if init:
                self.session.run(tf.global_variables_initializer())
        elif graph:
            self.graph = graph
        else:
            self.graph = tf.Graph()

        if self.is_local_session:
            self.graph.as_default()
            self.session = tf.Session(graph=graph)

            if init:
                self.session.run(tf.global_variables_initializer())
            if len(variables) > 0:
                for name in variables:
                    variable = [v for v in tf.global_variables() if v.name == name][0]
                    self.session.run(variable.assign(variables[name]))

    def __enter__(self):
        return self.session

    def __exit__(self, type, value, traceback):
        if self.is_local_session:
            self.session.close()