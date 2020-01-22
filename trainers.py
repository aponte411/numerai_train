from abc import abstractmethod
import numerox as nx
import numerapi
import os
import models
from typing import Any, Tuple, List, Dict
import click
import utils
from metrics import correlations

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import (Pipeline, make_pipeline, FeatureUnion,
                              make_union)
from sklearn.feature_selection import SelectFromModel

LOGGER = utils.get_logger(__name__)

CURRENT_TOURNAMENT = 'kazutsugi'


class Trainer:
    """Base class for handling training/inference"""
    def __init__(self, data: nx.data.Data, tournament: str, name: str):
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def get_data(self) -> None:
        return self.data

    def get_name(self):
        return self.name

    def get_model(self):
        return self.model

    @abstractmethod
    def load_model_locally(self):
        pass

    @abstractmethod
    def load_model_from_s3(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def save_model_locally(self):
        pass

    @abstractmethod
    def save_model_to_s3(self):
        pass

    def make_predictions_and_prepare_submission(self,
                                                tournament: str,
                                                submit: bool = False
                                                ) -> nx.Prediction:
        """
        Make predictions using the .predict() method
        and save to CSV undet tmp folder.

        Requires environmental variables for PUBLIC_ID and
        SECRET_KEY.
        """

        public_id = os.environ["NUMERAI_PUBLIC_ID"]
        secret_key = os.environ["NUMERAI_SECRET_KEY"]

        LOGGER.info(f"Making predictions...")
        prediction: nx.Prediction = self.model.predict(self.data['tournament'],
                                                       self.tournament)
        prediction_filename: str = f'/tmp/{self.name}_prediction_{self.tournament}.csv'
        try:
            LOGGER.info(f"Saving predictions to CSV: {prediction_filename}")
            prediction.to_csv(prediction_filename)
        except Exception as e:
            LOGGER.error(f'Failed to save predictions with {e}')
            raise e

        if submit:
            try:
                submission_id = nx.upload(filename=prediction_filename,
                                          tournament=self.tournament,
                                          public_id=public_id,
                                          secret_key=secret_key,
                                          block=False,
                                          n_tries=3)
                LOGGER.info(
                    f'Predictions submitted. Submission id: {submission_id}')
            except Exception as e:
                LOGGER.error(f'Failure to upload predictions with {e}')
                raise e

        return prediction


class XGBoostTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='xgboost'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.XGBoostModel()
        self.model.load(saved_model_name)

    def load_from_s3(self, saved_model_name: str):
        self.model = models.XGBoostModel()
        self.model.load_from_s3(filename=saved_model_name,
                                key=saved_model_name)
        self.model = self.model.load(saved_model_name)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict):
        LOGGER.info(
            f"Building XGBoostModel from scratch for {self.tournament}")
        self.model = models.XGBoostModel(max_depth=params["max_depth"],
                                         learning_rate=params["learning_rate"],
                                         l2=params["l2"],
                                         n_estimators=params["n_estimators"])
        LOGGER.info(f"Training XGBoost model for {self.tournament}")
        eval_set = [(self.data['validation'].x,
                     self.data['validation'].y[self.tournament])]
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(saved_model_name)

    def save_to_s3(self, saved_model_name: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=saved_model_name, key=saved_model_name)


class LightGBMTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='lightgbm'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.LightGBMRegressorModel()
        self.model.load(saved_model_name)

    def load_from_s3(self, saved_model_name: str):
        self.model = models.LightGBMRegressorModel()
        self.model.load_from_s3(filename=saved_model_name,
                                key=saved_model_name)
        self.model = self.model.load(saved_model_name)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict):
        LOGGER.info(
            f"Building LightGBMModel from scratch for {self.tournament}")
        self.model = models.LightGBMModel(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            reg_lambda=params["reg_lambda"])
        LOGGER.info(f"Training LightGBMModel for {self.tournament}")
        eval_set = [(self.data['validation'].x,
                     self.data['validation'].y[self.tournament])]
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(saved_model_name)

    def save_to_s3(self, saved_model_name: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=saved_model_name, key=saved_model_name)


class CatBoostTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='catboost'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.CatBoostModel()
        self.model.load(saved_model_name)

    def load_from_s3(self, saved_model_name: str):
        self.model = models.CatBoostModel()
        self.model.load_from_s3(filename=saved_model_name,
                                key=saved_model_name)
        self.model = self.model.load(saved_model_name)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict):
        LOGGER.info(
            f"Building CatBoostModel from scratch for {self.tournament}")
        self.model = models.CatBoostModel(
            depth=params['depth'],
            learning_rate=params['learning_rate'],
            l2=params['l2'],
            iterations=params['iterations'])
        LOGGER.info(f"Training CatBoostModel for {self.tournament}")
        eval_set = [(self.data['validation'].x,
                     self.data['validation'].y[self.tournament])]
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(saved_model_name)

    def save_to_s3(self, saved_model_name: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=saved_model_name, key=saved_model_name)


class LSTMTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='lstm', gpu=None):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.gpu = gpu
        self.model = None

    def load_model_locally(self, saved_model_name: str):

        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.LSTMModel()
        self.model.load(saved_model_name)

    def load_from_s3(self, saved_model_name: str):

        self.model = models.LSTMModel()
        self.model.load_from_s3(filename=saved_model_name,
                                key=saved_model_name)
        self.model = self.model.load(saved_model_name)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict):

        LOGGER.info(f"Building LSTMModel from scratch for {self.tournament}")
        if self.gpu is not None:
            LOGGER.info(f"Building model with {self.gpu} GPU's")
        self.model = models.LSTMModel(time_steps=params['time_steps'],
                                      gpu=self.gpu)
        LOGGER.info(f"Training LSTM model for {self.tournament}")
        eval_set = (self.data['validation'].x,
                    self.data['validation'].y[self.tournament])
        for i in range(params['epochs']):
            self.model.fit(dfit=self.data['train'],
                           tournament=self.tournament,
                           eval_set=eval_set,
                           epochs=1,
                           batch_size=params['batch_size'])

    def save_model_locally(self, saved_model_name: str):

        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(saved_model_name)

    def save_to_s3(self, saved_model_name: str):

        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=saved_model_name, key=saved_model_name)


class FunctionalLSTMTrainer(Trainer):
    pass


class BidirectionalLSTMTrainer(Trainer):
    pass


class LinearTrainer(Trainer):
    pass


class VotingRegressorTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='voting_regressor'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.VotingRegressorModel()
        self.model.load(saved_model_name)

    def load_from_s3(self, saved_model_name: str):
        self.model = models.VotingRegressorModel()
        self.model.load_from_s3(filename=saved_model_name,
                                key=saved_model_name)
        self.model = self.model.load(saved_model_name)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict):
        LOGGER.info(
            f"Building VotingRegressorModel from scratch for {self.tournament}"
        )
        self.model = models.VotingRegressorModel()
        LOGGER.info(f"Training VotingRegressorModel for {self.tournament}")
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(saved_model_name)

    def save_to_s3(self, saved_model_name: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=saved_model_name, key=saved_model_name)


class StackingRegressorTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='stacking_regressor'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.StackingRegressorModel()
        self.model.load(saved_model_name)

    def load_from_s3(self, saved_model_name: str):
        self.model = models.StackingRegressorModel()
        self.model.load_from_s3(filename=saved_model_name,
                                key=saved_model_name)
        self.model = self.model.load(saved_model_name)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict):
        LOGGER.info(
            f"Building StackingRegressorModel from scratch for {self.tournament}"
        )
        self.model = models.StackingRegressorModel()
        LOGGER.info(f"Training StackingRegressorModel for {self.tournament}")
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, saved_model_name: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(saved_model_name)

    def save_to_s3(self, saved_model_name: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=saved_model_name, key=saved_model_name)


def train_and_save_functional_lstm_model(tournament: str,
                                         data: nx.data.Data,
                                         load_model: bool,
                                         save_model: bool,
                                         params: Dict = None) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'functional_lstm_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.FunctionalLSTMModel()
        model.load_from_s3(filename=saved_model_name, key=saved_model_name)
        model = model.load(saved_model_name)
        LOGGER.info(f"Trained model loaded from s3")
    else:
        LOGGER.infp(
            f"Building FunctionalLSTMModel from scratch for {tournament}")
        model = models.FunctionalLSTMModel(time_steps=1)
        LOGGER.info(f"Training Functional LSTM model for {tournament}")
        eval_set = (data['validation'].x, data['validation'].y[tournament])
        model.fit(dfit=data['train'],
                  tournament=tournament,
                  eval_set=eval_set,
                  epochs=params['epochs'],
                  batch_size=params['batch_size'])
    if save_model:
        LOGGER.info(f"Saving model for {tournament}")
        model.save(saved_model_name)
        LOGGER.info(f"Saving model for {tournament} to s3 bucket")
        model.save_to_s3(filename=saved_model_name, key=saved_model_name)

    return model


def train_and_save_bidirectional_lstm_model(tournament: str,
                                            data: nx.data.Data,
                                            load_model: bool,
                                            save_model: bool,
                                            params: Dict = None) -> nx.Model:
    """Trains Bidirectional LSTM model and saves weights"""

    saved_model_name = f'bidirectional_lstm_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.BidirectionalLSTMModel()
        model.load_from_s3(filename=saved_model_name, key=saved_model_name)
        model = model.load(saved_model_name)
        LOGGER.info(f"Trained model loaded from s3")
    else:
        LOGGER.info(
            f"Building BidirectionalLSTMModel from scratch for {tournament}")
        model = models.BidirectionalLSTMModel(time_steps=1)
        LOGGER.info(f"Training BidirectionalLSTM model for {tournament}")
        eval_set = (data['validation'].x, data['validation'].y[tournament])
        model.fit(dfit=data['train'],
                  tournament=tournament,
                  eval_set=eval_set,
                  epochs=params['epochs'],
                  batch_size=params['batch_size'])
    if save_model:
        LOGGER.info(f"Saving model for {tournament} locally")
        model.save(saved_model_name)
        LOGGER.info(f"Saving model for {tournament} to s3 bucket")
        model.save_to_s3(filename=saved_model_name, key=saved_model_name)

    return model


def train_linear_model() -> None:
    """Train model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(tournaments)
    for tournament_name in tournaments:
        model = models.LinearModel()
        try:
            LOGGER.info(f"fitting model for {tournament_name}")
            model.fit(data['train'], tournament_name)
        except Exception as e:
            LOGGER.error(f"Training failed with {e}")
            raise e

        LOGGER.info(f"saving model for {tournament_name}")
        model.save(f'model_trained_{tournament_name}')