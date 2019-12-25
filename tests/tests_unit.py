import abc
import numerox as nx
import pandas as pd 
import numpy as np
import pytest
import joblib
import models


DUMMY_DF = pd.DataFrame({
    "x": np.random.random(5),
    "y": np.random.random(5)
})


class Model:
    @abc.abstractmethod
    def fit(self, X, y):

        features = X is not None
        targets = y is not None

        return features, targets 

    @abc.abstractmethod
    def predict(self, X):

        features = X is not None

        return features 

def test_model():
    results = Model().fit(DUMMY_DF.x, DUMMY_DF.y)
    assert results[0] == True 
    assert results[1] == True
    assert Model().predict(DUMMY_DF.x) == True 

    
class DummyModel(nx.Model):
    def __init__(self, verbose=False):
        self.params = None
        self.verbose = verbose
        self.model = Model()

    def fit(self, X, y):
        return self.model.fit(X, y)

    def fit_predict(self, dfit, dpre, tournament):
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)
        
        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)
    

def test_dummy():
    assert 1 == 1

@pytest.fixture(params=[
    models.LSTMModel,
    models.XGBoostModel,
    models.LinearModel,
    models.KerasModel,
    models.BidirectionalLSTMModel,
    models.FunctionalLSTMModel,
    DummyModel
])
def model(request):
    return request.param()


def test_lstm_model(model):
    assert type(model) == nx.Model
