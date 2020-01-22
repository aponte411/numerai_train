import numerox as nx
import numpy as np
import pandas as pd
import joblib
from typing import Any, Tuple, List
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, VotingRegressor,
                              StackingRegressor)
from xgboost import XGBRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
from keras import layers, Sequential
from keras import metrics
from keras.utils import multi_gpu_model
from keras.callbacks import (EarlyStopping, ModelCheckpoint, TensorBoard)
from keras.preprocessing.sequence import TimeseriesGenerator
from keras_functional_lstm import LstmModel
from utils import get_logger, S3Client
from metrics import (correlation_coefficient_loss, get_spearman_rankcor,
                     correlations)

LOGGER = get_logger(__name__)


class LinearModel(nx.Model):
    """Linear Regression"""
    def __init__(self):
        self.params = None
        self.model = LinearRegression()

    def fit(self, dfit, tournament):
        self.model.fit(dfit.x, dfit.y[tournament])

    def fit_predict(self, dfit, dpre, tournament):
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename) -> None:
        """Serialize model"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""

        return joblib.load(filename)


class XGBoostModel(nx.Model):
    """XGBoost Regressor"""
    def __init__(self,
                 max_depth: int = 7,
                 learning_rate: float = 0.001777765,
                 l2: float = 0.1111119,
                 n_estimators: int = 2019,
                 colsample_bytree: float = 0.019087,
                 tree_method: str = 'auto'):
        self.params = None
        self.model = XGBRegressor(max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  reg_lambda=l2,
                                  n_estimators=n_estimators,
                                  n_jobs=-1,
                                  tree_method=tree_method,
                                  colsample_bytree=colsample_bytree,
                                  verbosity=3)

    def fit(self,
            dfit: nx.data.Data,
            tournament: str,
            eval_set=None,
            eval_metric=None) -> None:
        # might be necessary for custom eval_metric function (callable object)
        # xgbtrain = xgb.DMatrix(dfit.x, label=dfit.y[tournament])

        self.model.fit(X=dfit.x,
                       y=dfit.y[tournament],
                       eval_set=eval_set,
                       eval_metric=eval_metric,
                       early_stopping_rounds=50)

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """

        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict(data_predict.x)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        try:
            prediction = prediction.merge_arrays(data_predict.ids, yhat,
                                                 self.name, tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament) -> Tuple:
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save_to_s3(self, filename: str, key: str) -> None:
        """Save model to s3 bucket"""

        s3 = S3Client()
        s3.upload_file(filename=filename, key=key)

    def load_from_s3(self, filename: str, key: str) -> None:
        """Download model from s3 bucket"""

        s3 = S3Client()
        s3.download_file(filename=filename, key=key)

    def save(self, filename) -> None:
        """Serialize model locally"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""

        return joblib.load(filename)


class LSTMModel(nx.Model):
    """
    Stacked LSTM network built with
    Keras Sequential class. The LSTMModel
    object inherits functionality from the
    nx.Model class.

    Reference: numerox/numerox/examples/model.rst
    """
    def __init__(self, time_steps: int = 1, gpu=None):
        self.params: Any = None
        self.time_steps: int = time_steps
        self.gpu = gpu
        self.model: Sequential = self._lstm_model()
        self.callbacks: List = None

    def _lstm_model(self) -> Sequential:
        """Returns LSTM Sequential model"""

        model = Sequential()
        model.add(
            layers.LSTM(units=225,
                        activation='relu',
                        input_shape=(self.time_steps, 310),
                        return_sequences=True))
        model.add(
            layers.LSTM(units=200,
                        kernel_initializer='glorot_normal',
                        activation='relu',
                        return_sequences=False))
        model.add(
            layers.Dense(units=150,
                         kernel_initializer='glorot_normal',
                         activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(
            layers.Dense(units=50,
                         kernel_initializer='glorot_normal',
                         activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(units=1))

        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model_checkpoint = ModelCheckpoint('best_lstm_model.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)
        logdir = f'LSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        tensorboard_callback = TensorBoard(log_dir=logdir)
        self.callbacks = [early_stop, model_checkpoint, tensorboard_callback]

        if self.gpu >= 2:
            try:
                model = multi_gpu_model(model, gpus=self.gpu, cpu_relocation=True)
                LOGGER.info(f"Training model with {self.gpu} gpus")
            except Exception as e:
                LOGGER.info(f"Failed to train model with GPUS due to {e}, reverting to CPU")
                raise e

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=[metrics.mae, correlation_coefficient_loss])

        return model

    def fit(self,
            dfit: nx.data.Data,
            tournament: str,
            eval_set: Tuple = None,
            epochs: int = 1,
            batch_size: int = 30) -> None:
        """Trains LSTM model using TimeSeriesGenerator"""

        train_generator = TimeseriesGenerator(data=dfit.x,
                                              targets=dfit.y[tournament],
                                              length=self.time_steps,
                                              sampling_rate=1,
                                              batch_size=batch_size)
        eval_generator = TimeseriesGenerator(data=eval_set[0],
                                             targets=eval_set[1],
                                             length=self.time_steps,
                                             sampling_rate=1,
                                             batch_size=batch_size)

        LOGGER.info('Training started...')
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=epochs,
                                 verbose=2,
                                 validation_data=eval_generator,
                                 callbacks=self.callbacks)
        # only necessary for online/batch learning
        self.model.reset_states()

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """

        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        test_generator = TimeseriesGenerator(
            data=data_predict.x,
            targets=data_predict.y[tournament],
            length=self.time_steps,
            sampling_rate=1,
            batch_size=1)
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict_generator(test_generator)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e
        # still needs to be fixed due to indexing error
        try:
            prediction = prediction.merge_arrays(ids=data_predict.ids,
                                                 y=yhat,
                                                 name=self.name,
                                                 tournament=tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit: nx.data.Data, dpre: nx.data.Data,
                    tournament: str) -> Tuple:
        """fit is done separately in .fit()"""

        test_generator = TimeseriesGenerator(data=dpre.x,
                                             targets=dpre.y[tournament],
                                             length=self.time_steps,
                                             sampling_rate=1,
                                             batch_size=1)
        yhat = self.model.predict_generator(test_generator)

        return dpre.ids, yhat

    def save_to_s3(self, filename: str, key: str) -> None:
        """Save model to s3 bucket"""

        s3 = S3Client()
        s3.upload_file(filename=filename, key=key)
    
    def load_from_s3(self, filename: str, key: str) -> None:
        """Download model from s3 bucket"""

        s3 = S3Client()
        s3.download_file(filename=filename, key=key)

    def save(self, filename) -> None:
        """Serialize model"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""

        return joblib.load(filename)


class FunctionalLSTMModel(nx.Model):
    """Functional implementation of Keras LSTM Model"""
    def __init__(self, time_steps: int = 1):
        self.params = None
        self.time_steps = time_steps
        self.model = self._functional_lstm_model()
        self.callbacks = None

    def _functional_lstm_model(self) -> LstmModel:
        """Returns Functional LSTM model"""

        model = LstmModel(timesteps=1)

        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model_checkpoint = ModelCheckpoint('best_functional_lstm_model.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)
        logdir = f'FLSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        tensorboard_callback = TensorBoard(log_dir=logdir)
        self.callbacks = [early_stop, model_checkpoint, tensorboard_callback]

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def fit(self,
            dfit: nx.data.Data,
            tournament: str,
            eval_set: Tuple = None,
            epochs: int = 1,
            batch_size: int = 30) -> None:
        """Trains Functional LSTM model using TimeSeriesGenerator"""

        train_generator = TimeseriesGenerator(data=dfit.x,
                                              targets=dfit.y[tournament],
                                              length=self.time_steps,
                                              sampling_rate=1,
                                              batch_size=batch_size)
        eval_generator = TimeseriesGenerator(data=eval_set[0],
                                             targets=eval_set[1],
                                             length=self.time_steps,
                                             sampling_rate=1,
                                             batch_size=batch_size)

        LOGGER.info('Training started...')
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=epochs,
                                 verbose=2,
                                 validation_data=eval_generator,
                                 callbacks=self.callbacks)

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """

        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        test_generator = TimeseriesGenerator(
            data=data_predict.x,
            targets=data_predict.y[tournament],
            length=self.time_steps,
            sampling_rate=1,
            batch_size=1)
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict_generator(test_generator)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        try:
            prediction = prediction.merge_arrays(data_predict.ids, yhat,
                                                 self.name, tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament) -> Tuple:

        # fit is done separately in `.fit()`
        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat
    
    def save_to_s3(self, filename: str, key: str) -> None:
        """Save model to s3 bucket"""

        s3 = S3Client()
        s3.upload_file(filename=filename, key=key)

    def load_from_s3(self, filename: str, key: str) -> None:
        """Download model from s3 bucket"""

        s3 = S3Client()
        s3.download_file(filename=filename, key=key)

    def save(self, filename) -> None:
        """Serialize model"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""

        return joblib.load(filename)


class BidirectionalLSTMModel(nx.Model):
    """Keras Bidirectional LSTM model"""
    def __init__(self, time_steps: int = 1):
        self.params = None
        self.time_steps: int = time_steps
        self.model: Sequential = self._bidirectional_lstm_model()
        self.callbacks: List = None

    def _bidirectional_lstm_model(self) -> Sequential:
        """Returns Bidirectional LSTM model"""

        model = Sequential()
        model.add(
            layers.Bidirectional(
                layers.LSTM(
                    units=200,
                    activation='relu',
                    input_shape=(self.time_steps, 310),
                    return_sequences=
                    False  # change back to True when adding another LSTM layer
                )))
        # model.add(layers.Bidirectional(layers.LSTM(
        #     units=200,
        #     kernel_initializer='glorot_normal',
        #     activation='relu',
        #     return_sequences=False
        #     )))
        model.add(
            layers.Dense(units=100,
                         kernel_initializer='glorot_normal',
                         activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(
            layers.Dense(units=50,
                         kernel_initializer='glorot_normal',
                         activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(units=1))

        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model_checkpoint = ModelCheckpoint('best_bidirectional_lstm_model.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)
        logdir = f'BILSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        tensorboard_callback = TensorBoard(log_dir=logdir)
        self.callbacks = [early_stop, model_checkpoint, tensorboard_callback]

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def fit(self,
            dfit: nx.data.Data,
            tournament: str,
            eval_set: Tuple = None,
            epochs: int = 1,
            batch_size: int = 30) -> None:
        """Trains Bidirectional LSTM model using TimeSeriesGenerator"""

        train_generator = TimeseriesGenerator(data=dfit.x,
                                              targets=dfit.y[tournament],
                                              length=self.time_steps,
                                              sampling_rate=1,
                                              batch_size=batch_size)
        eval_generator = TimeseriesGenerator(data=eval_set[0],
                                             targets=eval_set[1],
                                             length=self.time_steps,
                                             sampling_rate=1,
                                             batch_size=batch_size)

        LOGGER.info('Training started...')
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=epochs,
                                 verbose=2,
                                 validation_data=eval_generator,
                                 callbacks=self.callbacks)

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to .fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """

        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        test_generator = TimeseriesGenerator(
            data=data_predict.x,
            targets=data_predict.y[tournament],
            length=self.time_steps,
            sampling_rate=1,
            batch_size=1)
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict_generator(test_generator)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        try:
            prediction = prediction.merge_arrays(data_predict.ids, yhat,
                                                 self.name, tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament) -> Tuple:
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat
    
    def save_to_s3(self, filename: str, key: str) -> None:
        """Save model to s3 bucket"""

        s3 = S3Client()
        s3.upload_file(filename=filename, key=key)

    def load_from_s3(self, filename: str, key: str) -> None:
        """Download model from s3 bucket"""

        s3 = S3Client()
        s3.download_file(filename=filename, key=key)

    def save(self, filename):
        """Serialize model"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        """Load trained model"""

        return joblib.load(filename)


class LightGBMModel(nx.Model):
    """LGBMRegressor Model"""
    def __init__(self,
                 n_estimators: int = 2019,
                 learning_rate: float = 0.00111325,
                 reg_lambda: float = 0.111561):
        self.params = None
        self.model = LGBMRegressor(n_estimators=n_estimators,
                                   learning_rate=learning_rate,
                                   reg_lambda=reg_lambda,
                                   silent=False,
                                   random_state=511,
                                   early_stopping_rounds=25,
                                   n_jobs=-1)

    def fit(self,
            dfit: nx.data.Data,
            tournament: str,
            eval_set=None,
            eval_metric=None) -> None:

        self.model.fit(X=dfit.x,
                       y=dfit.y[tournament],
                       eval_set=eval_set,
                       early_stopping_rounds=50)

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """

        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict(data_predict.x)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        try:
            prediction = prediction.merge_arrays(data_predict.ids, yhat,
                                                 self.name, tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament) -> Tuple:
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save_to_s3(self, filename: str, key: str) -> None:
        """Save model to s3 bucket"""

        s3 = S3Client()
        s3.upload_file(filename=filename, key=key)
    
    def load_from_s3(self, filename: str, key: str) -> None:
        """Download model from s3 bucket"""

        s3 = S3Client()
        s3.download_file(filename=filename, key=key)

    def save(self, filename) -> None:
        """Serialize model"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""

        return joblib.load(filename)


class CatBoostModel(nx.Model):
    """CatBoostRegressor Model"""
    def __init__(self,
                 depth: int = 8,
                 learning_rate: float = 0.01234,
                 l2: float = 0.01,
                 iterations: int = 2019,
                 task_type: str = 'CPU'):
        self.params = None
        self.model = CatBoostRegressor(loss_function='RMSE',
                                       depth=depth,
                                       learning_rate=learning_rate,
                                       l2_leaf_reg=l2,
                                       iterations=iterations,
                                       task_type=task_type,
                                       random_seed=511,
                                       od_type='Iter',
                                       od_wait=20)

    def fit(self, dfit: nx.data.Data, tournament: str, eval_set=None) -> None:

        self.model.fit(X=dfit.x, y=dfit.y[tournament], eval_set=eval_set)

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """

        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict(data_predict.x)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        try:
            prediction = prediction.merge_arrays(data_predict.ids, yhat,
                                                 self.name, tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament) -> Tuple:
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save_to_s3(self, filename: str, key: str) -> None:
        """Save model to s3 bucket"""

        s3 = S3Client()
        s3.upload_file(filename=filename, key=key)
    
    def load_from_s3(self, filename: str, key: str) -> None:
        """Download model from s3 bucket"""

        s3 = S3Client()
        s3.download_file(filename=filename, key=key)

    def save(self, filename) -> None:
        """Serialize model"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""

        return joblib.load(filename)


class VotingRegressorModel(nx.Model):
    """
    VotingRegressorModel that uses
    XGBRegressor, RandomForestRegressor, LightGBMRegressor,
    CatBoostRegressor, LinearRegression, ExtraTreesRegressor, 
    and GradientBoostingRegressor.
    """
    def __init__(self):
        self.params = None
        self.model = VotingRegressor([
            ('XGBoost',
             XGBRegressor(max_depth=5,
                          learning_rate=0.001,
                          l2=0.01,
                          n_estimators=1000)),
            ('RandomForest', RandomForestRegressor()),
            ('LightGBM',
             LGBMRegressor(n_estimators=100,
                           learning_rate=0.01,
                           reg_lambda=0.1)),
            ('CatBoost',
             CatBoostRegressor(depth=4,
                               learning_rate=0.0001,
                               l2_leaf_reg=0.2,
                               iterations=300)),
            ('LinearRegression', LinearRegression()),
            ('ExtraTrees', ExtraTreesRegressor()),
            ('GradientBoosting', GradientBoostingRegressor())
        ])

    def fit(self,
            dfit: nx.data.Data,
            tournament: str,
            eval_set=None,
            eval_metric=None) -> None:

        self.model.fit(X=dfit.x, y=dfit.y[tournament])

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """

        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict(data_predict.x)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        try:
            prediction = prediction.merge_arrays(data_predict.ids, yhat,
                                                 self.name, tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament) -> Tuple:
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename) -> None:
        """Serialize model"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""

        return joblib.load(filename)


class StackingRegressorModel(nx.Model):
    """
    StackingRegressorModel that uses XGBRegressor, 
    RandomForestRegressor, LightGBMRegressor,
    CatBoostRegressor, ExtraTreesRegressor,
    and GradientBoostingRegressor.
    """
    def __init__(self):
        self.params = None
        self.model = StackingRegressor(estimators=[
            ('XGBoost',
             XGBRegressor(max_depth=7,
                          learning_rate=0.001,
                          l2=0.01,
                          n_estimators=500)),
            ('LightGBM',
             LGBMRegressor(n_estimators=300,
                           learning_rate=0.001,
                           reg_lambda=0.001)),
            ('CatBoost',
             CatBoostRegressor(depth=4,
                               learning_rate=0.0001,
                               l2_leaf_reg=0.2,
                               iterations=500)),
            ('ExtraTrees', ExtraTreesRegressor()),
            ('GradientBoosting', GradientBoostingRegressor())
        ],
                                       final_estimator=RandomForestRegressor())

    def fit(self,
            dfit: nx.data.Data,
            tournament: str,
            eval_set=None,
            eval_metric=None) -> None:

        self.model.fit(X=dfit.x, y=dfit.y[tournament])

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """

        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict(data_predict.x)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        try:
            prediction = prediction.merge_arrays(data_predict.ids, yhat,
                                                 self.name, tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament) -> Tuple:
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename) -> None:
        """Serialize model"""

        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""

        return joblib.load(filename)