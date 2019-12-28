
import numerox as nx
import numpy as np
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
        LOGGER.info(
            predictions.summaries(
            data['validation'], 
            tournament=tournament_name)
        )
        LOGGER.info(
            predictions[:, tournament_name].metric_per_era(
                data=data['validation'], 
                tournament=tournament_name)
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
        LOGGER.info(
            predictions.summaries(
            data['validation'], 
            tournament=tournament_name)
        )
        LOGGER.info(
            predictions[:, tournament_name].metric_per_era(
                data=data['validation'], 
                tournament=tournament_name)
            )

    return predictions


def train_and_predict_catboost_model() -> Any:
    """Trains CatBoost model and saves weights"""

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
        LOGGER.info(
            predictions.summaries(
            data['validation'], 
            tournament=tournament_name)
        )
        LOGGER.info(
            predictions[:, tournament_name].metric_per_era(
                data=data['validation'], 
                tournament=tournament_name)
            )
    
    return predictions


def train_and_predict_lightgbm_model() -> Any:
    """Trains LightGBM model and saves weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_lightgbm_model(
            tournament=tournament_name, 
            data=data
            )
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='lightgbm',
            data=data,
            tournament=tournament_name
            )
        LOGGER.info(
            predictions.summaries(
            data['validation'], 
            tournament=tournament_name)
        )
        LOGGER.info(
            predictions[:, tournament_name].metric_per_era(
                data=data['validation'], 
                tournament=tournament_name)
            )
    
    return predictions

# WIP
def train_and_predict_ensemble():

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    final_prediction = nx.Prediction()
    for tournament_name in tournaments:
        LOGGER.info(f'Training/Loading Ensemble Model...')
        models: List[nx.Model] = train.train_ensemble_model(
            tournament=tournament_name, 
            data=data
            )
        LOGGER.info(f'Making predictions for each of the individual models...')
        predictions = [
            model.predict(data['tournament'], tournament_name) for model in models
            ]
        LOGGER.info(f'Averaging the predictions...')
        mean_prediction = np.mean([prediction.y for prediction in predictions])
        LOGGER.info(f'Preparing nx.Prediction objects...')
        final_prediction = final_prediction.merge_arrays(
            data['tournament'].ids,
            mean_prediction,
            'EnsembleModel',
            tournament_name)
        LOGGER.info(f'Model averaging complete...preparing predictions for summary')
        LOGGER.info(
            final_prediction.summaries(
            data['validation'], 
            tournament=tournament_name)
        )
        LOGGER.info(
            final_prediction[:, tournament_name].metric_per_era(
                data=data['validation'], 
                tournament=tournament_name)
            )
        # LOGGER.info(
        #     final_prediction[:, tournament_name].dominance(data['validation'])
        # )
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
    if model == 'lightgbm':
        return train_and_predict_lightgbm_model()
    if model == 'flstm':
        return train_and_predict_functional_lstm_model()
    if model == 'ensemble':
        return train_and_predict_ensemble()
    else:
        LOGGER.info('None of the models were chosen..')


if __name__ == "__main__":
    predictions = main()
    print(predictions.shape)
    print(predictions)