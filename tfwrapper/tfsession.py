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
            self.context_mgr = self.graph.as_default()
            self.session = tf.Session(graph=self.graph)

            if init:
                self.session.run(tf.global_variables_initializer())
            if len(variables) > 0:
                for name in variables:
                    tensor = variables[name]['tensor']
                    value = variables[name]['value']
                    self.session.run(tensor.assign(value))

    def __enter__(self):
        if self.is_local_session:
            self.context_mgr.__enter__()
            self.session.__enter__()
        return self.session

    def __exit__(self, type, value, traceback):
        if self.is_local_session:
            self.session.close()
            self.context_mgr.__exit__(type, value, traceback)