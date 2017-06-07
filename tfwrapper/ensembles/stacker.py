class Stacker():
    def __init__(self, prediction_models, decision_model, name='Stacker'):
        self.prediction_models = prediction_models
        self.decision_model = decision_model
        self.name = name

    def train(self, dataset, *, epochs, sess=None):
        for model in self.prediction_models:
            model.train(dataset, epochs=epochs, sess=sess)

        predictions = self.prediction_models[0].predict(dataset)
        for model in self.prediction_models[1:]:
            predictions = np.concatenate([predictions, model.predict(dataset, sess=sess)], axis=1)

        print(predictions.shape)