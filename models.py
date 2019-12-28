import numerox as nx
import numpy as np
import pandas as pd
import joblib
from typing import Any, Tuple, List
from datetime import datetime
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from keras import layers, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import TimeseriesGenerator
from keras_functional_lstm import LstmModel
from utils import get_logger

LOGGER = get_logger(__name__)


class LinearModel(nx.Model):
    """Linear Regression"""

    def __init__(self, verbose=False):
        self.params = None
        self.verbose = verbose
        self.model = LinearRegression()

    def fit(self, dfit, tournament):
        self.model.fit(dfit.x, dfit.y[tournament])

    def fit_predict(self, dfit, dpre, tournament):
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)
        
        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


class XGBoostModel(nx.Model):
    """XGBoost Regressor"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.params = None
        self.model = XGBRegressor(max_depth=7,
                                  learning_rate=0.019182738141,
                                  n_estimators=1982,
                                  n_jobs=-1,
                                  colsample_bytree=0.106543,
                                  verbosity=3)

    def fit(self, dfit: nx.data.Data, tournament: str, eval_set=None):
        self.model.fit(
            X=dfit.x, 
            y=dfit.y[tournament],
            eval_set=eval_set, 
            early_stopping_rounds=50
        )

    def predict(self, dpre: nx.data.Data, tournament: str) -> Any:
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
            LOGGER.info('Inference complete...now preparing predictions for submission')
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e 

        try:
            prediction = prediction.merge_arrays(
                data_predict.ids, 
                yhat, 
                self.name, 
                tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament):
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


class KerasModel(nx.Model):
    """Keras Regressor"""
        
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.params = None
        self.model = self._keras_model()
        self.callbacks = None

    def _keras_model(self):
        """Returns Keras Sequential model"""

        model = Sequential()
        model.add(layers.Dense(
            units=225, 
            kernel_initializer='glorot_normal', 
            activation='relu', 
            input_dim=310
            ))
        model.add(layers.Dense(
            units=200, 
            kernel_initializer='glorot_normal', 
            activation='relu'
            ))
        model.add(layers.Dense(
            units=150, 
            kernel_initializer='glorot_normal', 
            activation='relu'
            ))
        model.add(layers.Dense(
            units=50, 
            kernel_initializer='glorot_normal',
            activation='relu'
            ))
        model.add(layers.Dense(units=1))
        
        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model_checkpoint = ModelCheckpoint(
            'best_keras_model.h5', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True
            )
        self.callbacks = [early_stop, model_checkpoint]

        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy']
            )
        
        return model 

    def fit(
        self, 
        dfit: nx.data.Data, 
        tournament: str, 
        eval_set: Tuple = None) -> Any:
        self.model.fit(
            dfit.x, 
            dfit.y[tournament],
            epochs=100, 
            batch_size=120,
            verbose=2,
            validation_data=eval_set,
            callbacks=self.callbacks
        )

    def fit_predict(self, dfit, dpre, tournament):

        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


class LSTMModel(nx.Model):
    """
    Stacked LSTM network buil with
    Keras Sequential class. The LSTMModel
    object inherits functionality from the
    nx.Model class.

    Reference: numerox/numerox/examples/model.rst
    """
        
    def __init__(self, time_steps: int = 1):
        self.params: Any = None
        self.time_steps: int = time_steps
        self.model: Sequential = self._lstm_model()
        self.callbacks: List = None

    def _lstm_model(self):
        """Returns LSTM Sequential model"""

        model = Sequential()
        model.add(layers.LSTM(
            units=225, 
            activation='relu', 
            input_shape=(self.time_steps, 310),
            return_sequences=True
            ))
        model.add(layers.LSTM(
            units=200, 
            kernel_initializer='glorot_normal', 
            activation='relu', 
            return_sequences=False
            ))
        model.add(layers.Dense(
            units=150, 
            kernel_initializer='glorot_normal', 
            activation='relu'
            ))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(
            units=50, 
            kernel_initializer='glorot_normal', 
            activation='relu'
            ))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(units=1))
        
        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model_checkpoint = ModelCheckpoint(
            'best_lstm_model.h5', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True
            )
        logdir = f'LSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        tensorboard_callback = TensorBoard(log_dir=logdir)
        self.callbacks = [
            early_stop, 
            model_checkpoint,
            tensorboard_callback
            ]

        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy']
            )
        
        return model 

    def prepare_model_data(self, train, eval, test) -> Any:
        """
        self.train_generator = dfit
        self.eval_generator = eval_set
        self.test_generator = dpre
        """
        pass

    def fit(
        self, 
        dfit: nx.data.Data, 
        tournament: str, 
        eval_set: Tuple = None,
        epochs: int = 50,
        batch_size: int = 30) -> Any:
        """Trains LSTM model using TimeSeriesGenerator"""

        train_generator = TimeseriesGenerator(
            data=dfit.x, 
            targets=dfit.y[tournament],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=batch_size)

        eval_generator = TimeseriesGenerator(
            data=eval_set[0], 
            targets=eval_set[1],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=batch_size)

        LOGGER.info('Training started...')           
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            verbose=2,
            validation_data=eval_generator,
            callbacks=self.callbacks
        )

    def predict(self, dpre: nx.data.Data, tournament: str) -> Any:
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
            batch_size=1
            )

        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict_generator(test_generator)
            LOGGER.info('Inference complete...now preparing predictions for submission')
            return yhat
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e 

        # try:
        #     prediction = prediction.merge_arrays(
        #         data_predict.ids, 
        #         yhat, 
        #         self.name, 
        #         tournament)
        #     return prediction
        # except Exception as e:
        #     LOGGER.error(f'Failure to prepare predictions with {e}')
        #     raise e

    def fit_predict(
        self, 
        dfit: nx.data.Data, 
        dpre: nx.data.Data, 
        tournament: str) -> Tuple[np.array]:
        """fit is done separately in .fit()"""

        test_generator = TimeseriesGenerator(
            data=dpre.x,
            targets=dpre.y[tournament],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=1)
        yhat = self.model.predict_generator(test_generator)

        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


class FunctionalLSTMModel(nx.Model):
    """Functional implementation of Keras LSTM Model"""
        
    def __init__(self, time_steps: int = 1):
        self.params = None
        self.time_steps = time_steps
        self.model = self._functional_lstm_model()
        self.callbacks = None

    def _functional_lstm_model(self):
        """Returns Functional LSTM model"""

        model = LstmModel(timesteps=1)
    
        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model_checkpoint = ModelCheckpoint(
            'best_functional_lstm_model.h5', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True
            )
        logdir = f'FLSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        tensorboard_callback = TensorBoard(log_dir=logdir)
        self.callbacks = [
            early_stop, 
            model_checkpoint,
            tensorboard_callback
            ]

        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy']
            )
        
        return model 

    def fit(
        self, 
        dfit: nx.data.Data, 
        tournament: str, 
        eval_set: Tuple = None,
        epochs: int = 50,
        batch_size: int = 30) -> Any:
        """Trains Functional LSTM model using TimeSeriesGenerator"""

        train_generator = TimeseriesGenerator(
            data=dfit.x, 
            targets=dfit.y[tournament],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=batch_size)

        eval_generator = TimeseriesGenerator(
            data=eval_set[0], 
            targets=eval_set[1],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=batch_size)

        LOGGER.info('Training started...')           
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            verbose=2,
            validation_data=eval_generator,
            callbacks=self.callbacks
        )


    def predict(self, dpre: nx.data.Data, tournament: str) -> Any:
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
            batch_size=1
            )

        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict_generator(test_generator)
            LOGGER.info('Inference complete...now preparing predictions for submission')
            return yhat
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e 

        # try:
        #     prediction = prediction.merge_arrays(
        #         data_predict.ids, 
        #         yhat, 
        #         self.name, 
        #         tournament)
        #     return prediction
        # except Exception as e:
        #     LOGGER.error(f'Failure to prepare predictions with {e}')
        #     raise e

    def fit_predict(self, dfit, dpre, tournament):

        # fit is done separately in `.fit()`
        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


class BidirectionalLSTMModel(nx.Model):
    """Keras Bidirectional LSTM model"""
        
    def __init__(self, time_steps: int = 1):
        self.params = None
        self.time_steps = time_steps
        self.model = self._bidirectional_lstm_model()
        self.callbacks = None

    def _bidirectional_lstm_model(self):
        """Returns Bidirectional LSTM model"""

        model = Sequential()
        model.add(layers.Bidirectional(layers.LSTM(
            units=225, 
            activation='relu', 
            input_shape=(self.time_steps, 310),
            return_sequences=True
            )))
        model.add(layers.Bidirectional(layers.LSTM(
            units=200, 
            kernel_initializer='glorot_normal', 
            activation='relu', 
            return_sequences=False
            )))
        model.add(layers.Dense(
            units=150, 
            kernel_initializer='glorot_normal', 
            activation='relu'
            ))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(
            units=50, 
            kernel_initializer='glorot_normal', 
            activation='relu'
            ))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(units=1))
        
        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model_checkpoint = ModelCheckpoint(
            'best_bidirectional_lstm_model.h5', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True
            )
        logdir = f'BILSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        tensorboard_callback = TensorBoard(log_dir=logdir)
        self.callbacks = [
            early_stop, 
            model_checkpoint,
            tensorboard_callback
            ]

        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy']
            )
        
        return model 

    def fit(
        self, 
        dfit: nx.data.Data, 
        tournament: str, 
        eval_set: Tuple = None,
        epochs: int = 1,
        batch_size: int = 30) -> Any:
        """Trains Bidirectional LSTM model using TimeSeriesGenerator"""

        train_generator = TimeseriesGenerator(
            data=dfit.x, 
            targets=dfit.y[tournament],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=batch_size)

        eval_generator = TimeseriesGenerator(
            data=eval_set[0], 
            targets=eval_set[1],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=batch_size)
        LOGGER.info('Training started...')           
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            verbose=2,
            validation_data=eval_generator,
            callbacks=self.callbacks
        )


    def predict(self, dpre: nx.data.Data, tournament: str) -> Any:
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
            batch_size=1
            )

        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict_generator(test_generator)
            LOGGER.info('Inference complete...now preparing predictions for submission')
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e 

        try:
            prediction = prediction.merge_arrays(
                data_predict.ids, 
                yhat, 
                self.name, 
                tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament):

        # fit is done separately in `.fit()`
        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


class LightGBMRegressorModel(nx.Model):
    """LGBMRegressor Model"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.params = None
        self.model = LGBMRegressor(
            n_estimators=2037,
            silent=False,
            random_state=511,
            early_stopping_rounds=50
        )

    def fit(self, dfit: nx.data.Data, tournament: str, eval_set=None):
        self.model.fit(
            X=dfit.x, 
            y=dfit.y[tournament],
            eval_set=eval_set, 
            early_stopping_rounds=50
        )

    def predict(self, dpre: nx.data.Data, tournament: str) -> Any:
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
            LOGGER.info('Inference complete...now preparing predictions for submission')
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e 

        try:
            prediction = prediction.merge_arrays(
                data_predict.ids, 
                yhat, 
                self.name, 
                tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament):
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


class CatBoostRegressorModel(nx.Model):
    """CatBoostRegressor Model"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.params = None
        self.model = CatBoostRegressor(
            loss_function='RMSE', 
            depth=7, 
            learning_rate=0.43217778, 
            iterations=1003,
            random_seed=511,
            od_type='Iter',
            od_wait=20
            )

    def fit(self, dfit: nx.data.Data, tournament: str, eval_set=None):
        self.model.fit(
            X=dfit.x, 
            y=dfit.y[tournament],
            eval_set=eval_set, 
            early_stopping_rounds=50
        )

    def predict(self, dpre: nx.data.Data, tournament: str) -> Any:
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
            LOGGER.info('Inference complete...now preparing predictions for submission')
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e 

        try:
            prediction = prediction.merge_arrays(
                data_predict.ids, 
                yhat, 
                self.name, 
                tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament):
        # fit is done separately in `.fit()`

        yhat = self.model.predict(dpre.x)

        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)