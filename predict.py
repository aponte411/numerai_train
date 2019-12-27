
import numerox as nx
import numerapi
import os
import models
from typing import Any, Tuple, List
import click
from utils import get_logger, prepare_tournament_data
import train

LOGGER = get_logger(__name__)


def make_predictions_and_prepare_submission(
    model: nx.Model,
    model_name: str, 
    data: nx.data.Data, 
    tournament: str) -> nx.Prediction:
    """
    Make predictions using the .predict() method
    and save to CSV undet tmp folder.
    """

    LOGGER.info(f"Making predictions...")
    prediction: nx.Prediction = model.predict(data['tournament'], tournament)
    prediction_filename: str = f'/tmp/{model_name}_prediction_{tournament}.csv'
    LOGGER.info(f"Saving predictions to CSV: {prediction_filename}")
    try:
        prediction.to_csv(prediction_filename)
    except Exception as e:
        LOGGER.error(f'Failed to save predictions with {e}')
        raise e

    return prediction


def train_and_predict_lstm_model() -> Any:
    """Train LSTM model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_lstm_model(
            tournament=tournament_name, 
            data=data
            )
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='lstm',
            data=data,
            tournament=tournament_name
            )

    return predictions


def train_and_predict_xgboost_model() -> Any:
    """Trains XGBoost model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_xgboost_model(
            tournament=tournament_name, 
            data=data
            )
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='xgboost',
            data=data,
            tournament=tournament_name
            )

    return predictions


def train_and_predict_functional_lstm_model() -> Any:
    """Trains Functional LSTM model and saves weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_functional_lstm_model(
            tournament=tournament_name, 
            data=data
            )
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='functional_lstm',
            data=data,
            tournament=tournament_name
            )

    return predictions

def train_and_predict_catboost_model() -> Any:
    """Trains Catboost model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_catboost_model(
            tournament=tournament_name, 
            data=data
            )
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='catboost',
            data=data,
            tournament=tournament_name
            )

    return predictions


@click.command()
@click.option('-m', '--model', type=str, default='xgboost')
def main(model: str) -> Any:

    if model == 'lstm':
        return train_and_predict_lstm_model()
    if model == 'xgboost':
        return train_and_predict_xgboost_model()
    if model == 'catboost':
        return train_and_predict_catboost_model()
    if model == 'flstm':
        return train_and_predict_functional_lstm_model()
    else:
        LOGGER.info('None of the models were chosen..')


if __name__ == "__main__":
    predictions = main()
    print(predictions.shape)
    print(predictions)