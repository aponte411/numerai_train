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

# DEFINE PARAMS HERE
LSTM_PARAMS = {'epochs': 1, 'batch_size': 120}

XGBOOST_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.00123,
    "l2": 0.02,
    "n_estimators": 2579
}

LIGHTGBM_PARAMS = {
    "n_estimators": 2579,
    "learning_rate": 0.00111325,
    "reg_lambda": 0.0111561
}

CATBOOST_PARAMS = {
    'depth': 5,
    'learning_rate': 0.001,
    'l2': 0.01,
    'iterations': 2000
}


def make_predictions_and_prepare_submission(
        model: nx.Model,
        model_name: str,
        data: nx.data.Data,
        tournament: str,
        submit: bool = False) -> nx.Prediction:
    """
    Make predictions using the .predict() method
    and save to CSV undet tmp folder.

    Requires environmental variables for PUBLIC_ID and
    SECRET_KEY.
    """

    public_id = os.environ["NUMERAI_PUBLIC_ID"]
    secret_key = os.environ["NUMERAI_SECRET_KEY"]

    LOGGER.info(f"Making predictions...")
    prediction: nx.Prediction = model.predict(data['tournament'], tournament)
    prediction_filename: str = f'/tmp/{model_name}_prediction_{tournament}.csv'
    try:
        LOGGER.info(f"Saving predictions to CSV: {prediction_filename}")
        prediction.to_csv(prediction_filename)
    except Exception as e:
        LOGGER.error(f'Failed to save predictions with {e}')
        raise e

    if submit:
        try:
            submission_id = nx.upload(filename=prediction_filename,
                                      tournament=tournament,
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


def train_and_predict_xgboost_model(submit_to_numerai) -> Any:
    """Trains XGBoost model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_xgboost_model(
            tournament=tournament_name,
            data=data,
            load_model=False,
            save_model=True,
            params=XGBOOST_PARAMS)
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='xgboost',
            data=data,
            tournament=tournament_name,
            submit=submit_to_numerai)
        LOGGER.info(
            predictions.summaries(data['validation'],
                                  tournament=tournament_name))
        LOGGER.info(predictions[:, tournament_name].metric_per_era(
            data=data['validation'], tournament=tournament_name))

    return predictions


def train_and_predict_lightgbm_model(submit_to_numerai: bool) -> Any:
    """Trains LightGBM model and saves weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_lightgbm_model(
            tournament=tournament_name,
            data=data,
            load_model=True,
            save_model=False,
            params=LIGHTGBM_PARAMS)
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='lightgbm',
            data=data,
            tournament=tournament_name,
            submit=submit_to_numerai)
        LOGGER.info(
            predictions.summaries(data['validation'],
                                  tournament=tournament_name))
        LOGGER.info(predictions[:, tournament_name].metric_per_era(
            data=data['validation'], tournament=tournament_name))

    return predictions


def train_and_predict_catboost_model(submit_to_numerai: bool) -> nx.Prediction:
    """Trains CatBoost model and saves weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_catboost_model(
            tournament=tournament_name,
            data=data,
            load_model=False,
            save_model=True,
            params=CATBOOST_PARAMS)
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='catboost',
            data=data,
            tournament=tournament_name,
            submit=submit_to_numerai)
        LOGGER.info(
            predictions.summaries(data['validation'],
                                  tournament=tournament_name))
        LOGGER.info(predictions[:, tournament_name].metric_per_era(
            data=data['validation'], tournament=tournament_name))

    return predictions


def train_and_predict_lstm_model(submit_to_numerai: bool) -> nx.Prediction:
    """Train LSTM model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_lstm_model(
            tournament=tournament_name,
            data=data,
            load_model=False,
            save_model=True,
            params=LSTM_PARAMS)
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='lstm',
            data=data,
            tournament=tournament_name,
            submit=submit_to_numerai)
        LOGGER.info(
            predictions.summaries(data['validation'],
                                  tournament=tournament_name))
        LOGGER.info(predictions[:, tournament_name].metric_per_era(
            data=data['validation'], tournament=tournament_name))

    return predictions


def train_and_predict_functional_lstm_model(
        submit_to_numerai: bool = None) -> nx.Prediction:
    """Trains Functional LSTM model and saves weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_functional_lstm_model(
            tournament=tournament_name, data=data)
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='functional_lstm',
            data=data,
            tournament=tournament_name)
        LOGGER.info(
            predictions.summaries(data['validation'],
                                  tournament=tournament_name))
        LOGGER.info(predictions[:, tournament_name].metric_per_era(
            data=data['validation'], tournament=tournament_name))

    return predictions


def train_and_predict_voting_regressor_model(
        submit_to_numerai: bool) -> nx.Prediction:
    """Trains VotingRegressorModel and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_voting_regressor_model(
            tournament=tournament_name,
            data=data,
            load_model=False,
            save_model=True,
        )
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='voting_regressor',
            data=data,
            tournament=tournament_name,
            submit=submit_to_numerai)
        LOGGER.info(
            predictions.summaries(data['validation'],
                                  tournament=tournament_name))
        LOGGER.info(predictions[:, tournament_name].metric_per_era(
            data=data['validation'], tournament=tournament_name))

    return predictions


def train_and_predict_stacking_regressor_model(
        submit_to_numerai: bool) -> nx.Prediction:
    """Trains StackingRegressorModel and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train.train_and_save_stacking_regressor_model(
            tournament=tournament_name,
            data=data,
            load_model=False,
            save_model=True,
        )
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            model_name='stacking_regressor',
            data=data,
            tournament=tournament_name,
            submit=submit_to_numerai)
        LOGGER.info(
            predictions.summaries(data['validation'],
                                  tournament=tournament_name))
        LOGGER.info(predictions[:, tournament_name].metric_per_era(
            data=data['validation'], tournament=tournament_name))

    return predictions


@click.command()
@click.option('-m', '--model', type=str, default='xgboost')
@click.option('-s', '--submit', type=bool, default=False)
def main(model: str, submit) -> nx.Prediction:

    if model == 'xgboost':
        return train_and_predict_xgboost_model(submit)
    if model == 'lightgbm':
        return train_and_predict_lightgbm_model(submit)
    if model == 'catboost':
        return train_and_predict_catboost_model(submit)
    if model == 'lstm':
        return train_and_predict_lstm_model(submit)
    if model == 'flstm':
        return train_and_predict_functional_lstm_model()
    if model == 'voting_regressor':
        return train_and_predict_voting_regressor_model(submit)
    if model == 'stacking_regressor':
        return train_and_predict_stacking_regressor_model(submit)


if __name__ == "__main__":
    predictions = main()
    LOGGER.info(predictions.shape)
    LOGGER.info(predictions)