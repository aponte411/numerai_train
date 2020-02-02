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

    def load_model_locally(self, key: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.XGBoostModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str):
        self.model = models.XGBoostModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict):
        LOGGER.info(
            f"Building XGBoostModel from scratch for {self.tournament}")
        if params["tree_method"] == 'gpu_hist':
            LOGGER.info(f"Training XGBoost with GPU's")
        self.model = models.XGBoostModel(max_depth=params["max_depth"],
                                         learning_rate=params["learning_rate"],
                                         l2=params["l2"],
                                         n_estimators=params["n_estimators"],
                                         tree_method=params["tree_method"])
        LOGGER.info(f"Training XGBoost model for {self.tournament}")
        eval_set = [(self.data['validation'].x,
                     self.data['validation'].y[self.tournament])]
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, key: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class LightGBMTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='lightgbm'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, key: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.LightGBMModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str):
        self.model = models.LightGBMModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
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

    def save_model_locally(self, key: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class CatBoostTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='catboost'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, key: str) -> None:
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.CatBoostModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str) -> None:
        self.model = models.CatBoostModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict) -> None:
        LOGGER.info(
            f"Building CatBoostModel from scratch for {self.tournament}")
        self.model = models.CatBoostModel(
            depth=params['depth'],
            learning_rate=params['learning_rate'],
            l2=params['l2'],
            iterations=params['iterations'],
            task_type=params['task_type'])
        LOGGER.info(f"Training CatBoostModel for {self.tournament}")
        eval_set = [(self.data['validation'].x,
                     self.data['validation'].y[self.tournament])]
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, key: str) -> None:
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class LSTMTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='lstm', gpu=None):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.gpu = gpu
        self.model = None

    def load_model_locally(self, key: str) -> None:
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.LSTMModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str) -> None:
        self.model = models.LSTMModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict) -> None:
        LOGGER.info(f"Building LSTMModel from scratch for {self.tournament}")
        if self.gpu is not None:
            LOGGER.info(f"Building model with {self.gpu} GPU's")
        self.model = models.LSTMModel(time_steps=params['time_steps'],
                                      gpu=self.gpu)
        LOGGER.info(f"Training LSTM model for {self.tournament}")
        eval_set = (self.data['validation'].x,
                    self.data['validation'].y[self.tournament])
        for i in range(params['training_epochs']):
            self.model.fit(dfit=self.data['train'],
                        tournament=self.tournament,
                        eval_set=eval_set,
                        epochs=1)

    def save_model_locally(self, key: str) -> None:
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str) -> None:
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class FunctionalLSTMTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='funclstm', gpu=None):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.gpu = gpu
        self.model = None

    def load_model_locally(self, key: str) -> None:
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.FunctionalLSTMModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str) -> None:
        self.model = models.FunctionalLSTMModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict) -> None:
        LOGGER.info(
            f"Building FunctionalLSTMModel from scratch for {self.tournament}")
        if self.gpu is not None:
            LOGGER.info(f"Building model with {self.gpu} GPU's")
        self.model = models.FunctionalLSTMModel(
            time_steps=params['time_steps'], gpu=self.gpu)
        LOGGER.info(f"Training FunctionalLSTM model for {self.tournament}")
        eval_set = (self.data['validation'].x,
                    self.data['validation'].y[self.tournament])
        for i in range(params['epochs']):
            self.model.fit(dfit=self.data['train'],
                           tournament=self.tournament,
                           eval_set=eval_set,
                           epochs=1,
                           batch_size=params['batch_size'])

    def save_model_locally(self, key: str) -> None:
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str) -> None:
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class BidirectionalLSTMTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='bilstm', gpu=None):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.gpu = gpu
        self.model = None

    def load_model_locally(self, key: str) -> None:
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.BidirectionalLSTMModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str) -> None:
        self.model = models.BidirectionalLSTMModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict) -> None:
        LOGGER.info(
            f"Building BidirectionalLSTMModel from scratch for {self.tournament}"
        )
        if self.gpu is not None:
            LOGGER.info(f"Building model with {self.gpu} GPU's")
        self.model = models.BidirectionalLSTMModel(
            time_steps=params['time_steps'], gpu=self.gpu)
        LOGGER.info(f"Training BidirectionalLSTM model for {self.tournament}")
        eval_set = (self.data['validation'].x,
                    self.data['validation'].y[self.tournament])
        for i in range(params['epochs']):
            self.model.fit(dfit=self.data['train'],
                           tournament=self.tournament,
                           eval_set=eval_set,
                           epochs=1,
                           batch_size=params['batch_size'])

    def save_model_locally(self, key: str) -> None:
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str) -> None:
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class LinearTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='linear'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, key: str) -> None:
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.LinearModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str) -> None:
        self.model = models.LinearModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict) -> None:
        LOGGER.info(f"Building LinearModel from scratch for {self.tournament}")
        self.model = models.LinearModel()
        LOGGER.info(f"Training LinearModel for {self.tournament}")
        eval_set = [(self.data['validation'].x,
                     self.data['validation'].y[self.tournament])]
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, key: str) -> None:
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class VotingRegressorTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='voting_regressor'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, key: str) -> None:
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.VotingRegressorModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str) -> None:
        self.model = models.VotingRegressorModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict) -> None:
        LOGGER.info(
            f"Building VotingRegressorModel from scratch for {self.tournament}"
        )
        self.model = models.VotingRegressorModel()
        LOGGER.info(f"Training VotingRegressorModel for {self.tournament}")
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, key: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str) -> None:
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class StackingRegressorTrainer(Trainer):
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, data=None, tournament=None, name='stacking_regressor'):
        super().__init__(data=data, tournament=tournament, name=name)
        self.data = data
        self.tournament = tournament
        self.name = name
        self.model = None

    def load_model_locally(self, key: str) -> None:
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.StackingRegressorModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str) -> None:
        self.model = models.StackingRegressorModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, params: Dict) -> None:
        LOGGER.info(
            f"Building StackingRegressorModel from scratch for {self.tournament}"
        )
        self.model = models.StackingRegressorModel()
        LOGGER.info(f"Training StackingRegressorModel for {self.tournament}")
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

    def save_model_locally(self, key: str) -> None:
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str) -> None:
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)