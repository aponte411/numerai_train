import numerox as nx
import joblib
from typing import Any
from datetime import datetime

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from keras import layers, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import TimeseriesGenerator

from keras_functional_lstm import LstmModel


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

    def fit(self, dfit: nx.data.Data, tournament: str, eval_set: Tuple = None):
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
    """Keras LSTM Regressor"""
        
    def __init__(self, time_steps: int = 1):
        self.params = None
        self.time_steps = time_steps
        self.model = self._lstm_model()
        self.callbacks = None
        self.train_generator = None 
        self.eval_generator = None
        self.logdir = f'LSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    def _lstm_model(self):
        """Returns LSTM model"""

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
        tensorboard_callback = TensorBoard(log_dir=self.logdir)
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
        epochs: int = 50):
        """Trains LSTM model using TimeSeriesGenerator"""

        self.train_generator = TimeseriesGenerator(
            data=dfit.x, 
            targets=dfit.y[tournament],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=15)

        self.eval_generator = TimeseriesGenerator(
            data=eval_set[0], 
            targets=eval_set[1],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=15)
                    
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=epochs,
            verbose=2,
            validation_data=self.eval_generator,
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


class FunctionalLSTMModel(nx.Model):
    """Functional implementation of Keras LSTM Model"""
        
    def __init__(self, time_steps: int = 1):
        self.params = None
        self.time_steps = time_steps
        self.model = self._functional_lstm_model()
        self.callbacks = None
        self.train_generator = None 
        self.eval_generator = None
        self.logdir = f'FLSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'

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
        tensorboard_callback = TensorBoard(log_dir=self.logdir)
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
        epochs: int = 50):
        """Trains Functional LSTM model using TimeSeriesGenerator"""

        self.train_generator = TimeseriesGenerator(
            data=dfit.x, 
            targets=dfit.y[tournament],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=15)

        self.eval_generator = TimeseriesGenerator(
            data=eval_set[0], 
            targets=eval_set[1],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=15)
                    
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=epochs,
            verbose=2,
            validation_data=self.eval_generator,
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


class BidirectionalLSTMModel(nx.Model):
    """Keras Bidirectional LSTM model"""
        
    def __init__(self, time_steps: int = 1):
        self.params = None
        self.time_steps = time_steps
        self.model = self._bidirectional_lstm_model()
        self.callbacks = None
        self.train_generator = None 
        self.eval_generator = None
        self.logdir = f'BILSTM_logs/scalars/{datetime.now().strftime("%Y%m%d-%H%M%S")}'

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
        tensorboard_callback = TensorBoard(log_dir=self.logdir)
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
        epochs: int = 10):
        """Trains Bidirectional LSTM model using TimeSeriesGenerator"""

        self.train_generator = TimeseriesGenerator(
            data=dfit.x, 
            targets=dfit.y[tournament],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=15)

        self.eval_generator = TimeseriesGenerator(
            data=eval_set[0], 
            targets=eval_set[1],
            length=self.time_steps, 
            sampling_rate=1,
            batch_size=15)
                    
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=epochs,
            verbose=2,
            validation_data=self.eval_generator,
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