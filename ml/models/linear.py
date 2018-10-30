from vessel_tracking.ml.models.model import AbstractModel

from sklearn.linear_model import SGDClassifier

class LinearModel(AbstractModel):
    def build(self):
        self.model = SGDClassifier
