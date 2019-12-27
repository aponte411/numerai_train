import numerox as nx
import numerapi
import os
import models
from typing import Any, Tuple
import click
from utils import get_logger, prepare_tournament_data

LOGGER = get_logger(__name__)


def train_linear_model() -> Any:
    """Train model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(tournaments)

    for tournament_name in tournaments:
        model = models.LinearModel()
        try:
            LOGGER.info(f"fitting model for {tournament_name}")
            model.fit(
                data['train'], 
                tournament_name
            )
        except Exception as e:
            LOGGER.error(f"Training failed with {e}")
            raise e

        LOGGER.info(f"saving model for {tournament_name}")
        model.save(f'model_trained_{tournament_name}')


def train_keras_model() -> Any:
    """Train model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(tournaments)

    for tournament_name in tournaments:
        eval_set = (
            data['validation'].x, data['validation'].y[tournament_name]
        )
        model = models.KerasModel()
        try:
            LOGGER.info(f"fitting model for {tournament_name}")
            model.fit(
                data['train'], 
                tournament_name,
                eval_set=eval_set
            )
        except Exception as e:
            LOGGER.error(f"Training failed with {e}")
            raise e

        LOGGER.info(f"saving model for {tournament_name}")
        model.save(f'keras_model_trained_{tournament_name}')


def train_and_save_xgboost_model(tournament: str, data: nx.data.Data) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'xgboost_prediction_model_{tournament}'
    if os.path.exists(saved_model_name):
        LOGGER.info(f"using saved model for {tournament}")
        model = models.XGBoostModel().load(saved_model_name)
    else:
        LOGGER.info(f"Saved model not found for {tournament}")
        model = models.XGBoostModel()
        LOGGER.info(f"Training XGBoost model for {tournament}")
        eval_set = [(
            data['validation'].x, data['validation'].y[tournament]
        )]
        model.fit(
            dfit=data['train'], 
            tournament=tournament,
            eval_set=eval_set
        )
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'xgboost_prediction_model_{tournament}')
        
    return model


def train_and_save_lstm_model(tournament: str, data: nx.data.Data) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'lstm_prediction_model_{tournament}'
    if os.path.exists(saved_model_name):
        LOGGER.info(f"using saved model for {tournament}")
        model = models.LSTMModel().load(saved_model_name)
    else:
        LOGGER.info(f"Saved model not found for {tournament}")
        model = models.LSTMModel(time_steps=1)
        LOGGER.info(f"Training LSTM model for {tournament}")
        eval_set = (
            data['validation'].x, data['validation'].y[tournament]
        )
        model.fit(
            dfit=data['train'], 
            tournament=tournament,
            eval_set=eval_set,
            epochs=1,
            batch_size=30
        )
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'lstm_prediction_model_{tournament}')
        
    return model


def train_and_save_functional_lstm_model(tournament: str, data: nx.data.Data) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'functional_lstm_prediction_model_{tournament}'
    if os.path.exists(saved_model_name):
        LOGGER.info(f"using saved model for {tournament}")
        model = models.FunctionalLSTMModel().load(saved_model_name)
    else:
        LOGGER.info(f"Saved model not found for {tournament}")
        model = models.FunctionalLSTMModel(time_steps=1)
        LOGGER.info(f"Training Functional LSTM model for {tournament}")
        eval_set = (
            data['validation'].x, data['validation'].y[tournament]
        )
        model.fit(
            dfit=data['train'], 
            tournament=tournament,
            eval_set=eval_set,
            epochs=1,
            batch_size=30
        )
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'functional_lstm_prediction_model_{tournament}')
        
    return model 


def train_bidirectional_lstm_model() -> Any:
    """Trains Bidirectional LSTM model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(tournaments) 

    for tournament_name in tournaments:
        eval_set = (
            data['validation'].x, data['validation'].y[tournament_name]
        )
        model = models.BidirectionalLSTMModel(time_steps=2)
        try:
            LOGGER.info(f"fitting Bidirectional LSTM model for {tournament_name}")
            model.fit(
                dfit=data['train'], 
                tournament=tournament_name,
                eval_set=eval_set,
                epochs=20
            )
        except Exception as e:
            LOGGER.error(f"Training failed with {e}")
            raise e

        LOGGER.info(f"saving model for {tournament_name}")
        model.save(f'keras_bidirectional_lstm_model_trained_{tournament_name}')


def train_and_save_catboost_model(tournament: str, data: nx.data.Data) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'catboost_prediction_model_{tournament}'
    if os.path.exists(saved_model_name):
        LOGGER.info(f"using saved model for {tournament}")
        model = models.CatBoostRegressorModel().load(saved_model_name)
    else:
        LOGGER.info(f"Saved model not found for {tournament}")
        model = models.CatBoostRegressorModel()
        LOGGER.info(f"Training CatBoostRegressor model for {tournament}")
        eval_set = (
            data['validation'].x, data['validation'].y[tournament]
        )
        model.fit(
            dfit=data['train'], 
            tournament=tournament,
            eval_set=eval_set
        )
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'catboost_prediction_model_{tournament}')
        
    return model

@click.command()
@click.option('-m', '--model', type=str, default='xgboost')
def main(model: str) -> Any:
    """Selects which model to train"""

    if model.lower() == 'xgboost':
        model = train_and_save_xgboost_model()
    if model.lower() == 'keras':
        train_keras_model()
    if model.lower() == 'lstm':
        model = train_and_save_lstm_model()
    if model.lower() == 'flstm':
        model = train_and_save_functional_lstm_model()
    if model.lower() == 'bilstm':
        train_bidirectional_lstm_model()
    if model.lower() == 'linear':
        train_linear_model()
    else:
        LOGGER.info("None of the models were chosen..")


if __name__ == "__main__":
    main()
