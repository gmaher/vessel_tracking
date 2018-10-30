from vessel_tracking.ml.models import linear

def get_model(config):



class AbstractModel(object):
    def __init__(self, config):
        self.config = config

        self.build()

    def build(self):
        raise RuntimeError("Abstract not implemented")

    def fit(self, X, Y):
        raise RuntimeError("Abstract not implemented")

    def predict(self, X):
        raise RuntimeError("Abstract not implemented")

    def predict_prob(self, X):
        raise RuntimeError("Abstract not implemented")    
