import tensorflow as tf

assign_placeholder = tf.placeholder(tf.float32, shape=[1])

class TFSession():
    def __init__(self, session=None, graph=None, init=False, variables=None):
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

        self.init = init
        self.variables = variables

    def __enter__(self):
        global assign_op

        if self.is_local_session:
            self.context_mgr.__enter__()
            self.session.__enter__()

            if self.init:
                self.session.run(tf.global_variables_initializer())

            if self.variables is not None:
                for name in self.variables:
                    tensor = self.variables[name]['tensor']
                    value = self.variables[name]['value']
                    tensor_name = tensor.name

                    if ':' in tensor_name:
                        tensor_name = tensor_name.split(':')[0]
                    
                    # Creates a single assign op to avoid exploding the graph with assign nodes
                    if not 'assign_op' in self.variables[name]:
                        placeholder = tf.placeholder(dtype=tensor.dtype, shape=tensor.shape, name=tensor_name + '_assign_placeholder')
                        assign_op = tensor.assign(placeholder)
                        self.variables[name]['placeholder'] = placeholder
                        self.variables[name]['assign_op'] = assign_op

                    self.session.run(self.variables[name]['assign_op'], feed_dict={self.variables[name]['placeholder']: value})

        return self.session

    def __exit__(self, type, value, traceback):
        if self.is_local_session:
            self.session.__exit__(type, value, traceback)
            self.context_mgr.__exit__(type, value, traceback)