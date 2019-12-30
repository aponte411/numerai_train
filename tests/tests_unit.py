from abc import abstractmethod
import numerox as nx
import pandas as pd 
import numpy as np
import pytest
import joblib
import models


DUMMY_DF = pd.DataFrame({
    "x": np.random.random(1000),
    "x_test": np.random.random(1000),
    "y": np.random.random(1000)
})


class Model:
    @abstractmethod
    def fit(self, X, y):

        features = X is not None
        targets = y is not None

        return features, targets 

    @abstractmethod
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
        super(DummyModel).__init__()
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


@pytest.fixture(params=[
    models.LSTMModel,
    models.XGBoostModel,
    models.CatBoostRegressorModel,
    models.LightGBMRegressorModel,
    models.LinearModel,
    models.BidirectionalLSTMModel,
    models.FunctionalLSTMModel,
    DummyModel
])
def model(request):
    """
    Model fixture to test multiple models
    in sequence. 

    Pass into test functions.
    """

    return request.param

def test_model_subclass(model):
    assert issubclass(model, nx.Model) == True

def test_model_name(model):
    assert model().name is not None

def test_model_fit_method(model):
    # model().fit(DUMMY_DF.x, DUMMY_DF.y)
    pass 

def test_model_predict_method():
    # DUMMY_DF['y_hat'] = model().predict(DUMMY.x_test)
    # assert DUMMY_DF['yhat'] is not None
    pass 

def test_model_submission():
    pass 


