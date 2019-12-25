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


def train_xgboost_model() -> Any:
    """Train model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(tournaments)

    for tournament_name in tournaments:
        eval_set = [(
            data['validation'].x, data['validation'].y[tournament_name]
        )]
        model = models.XGBoostModel()
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


def train_lstm_model() -> Any:
    """Train LSTM model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(tournaments)

    for tournament_name in tournaments:
        eval_set = (
            data['validation'].x, data['validation'].y[tournament_name]
        )
        model = models.LSTMModel(time_steps=2)
        try:
            LOGGER.info(f"fitting LSTM model for {tournament_name}")
            model.fit(
                dfit=data['train'], 
                tournament=tournament_name,
                eval_set=eval_set,
                epochs=100
            )
        except Exception as e:
            LOGGER.error(f"Training failed with {e}")
            raise e

        LOGGER.info(f"saving model for {tournament_name}")
        model.save(f'keras_lstm_model_trained_{tournament_name}')


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


@click.command()
@click.option('-m', '--model', type=str, default='xgboost')
def main(model: str) -> Any:
    """Selects which model to train"""

    if model.lower() == 'xgboost':
        train_xgboost_model()
    if model.lower() == 'keras':
        train_keras_model()
    if model.lower() == 'lstm':
        train_lstm_model()
    if model.lower() == 'bilstm':
        train_bidirectional_lstm_model()
    if model.lower() == 'linear':
        train_linear_model()
    else:
        LOGGER.info("None of the models were chosen..")


if __name__ == "__main__":
    main()
