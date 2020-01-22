import numerox as nx
import numpy as np
import numerapi
import os
import models
from typing import Any, Tuple, List, Dict
import click
import utils
import trainers

LOGGER = utils.get_logger(__name__)


def train_and_predict_xgboost_model(load_model: bool, save_model: bool,
                                    submit_to_numerai: bool,
                                    params: Dict) -> None:
    """Train/load model and conduct inference"""

    tournaments = utils.get_tournament_names()
    data = utils.get_tournament_data()
    for tournament_name in tournaments:
        saved_model_name = f'xgboost_prediction_model_{tournament_name}'
        trainer = trainers.XGBoostTrainer(data=data,
                                          tournament=tournament_name)
        if load_model:
            trainer.load_from_s3(saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_to_s3(saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


def train_and_predict_lightgbm_model(load_model: bool, save_model: bool,
                                     submit_to_numerai: bool,
                                     params: Dict) -> None:
    """Train/load model and conduct inference"""

    tournaments = utils.get_tournament_names()
    data = utils.get_tournament_data()
    for tournament_name in tournaments:
        saved_model_name = f'lightgbm_prediction_model_{tournament_name}'
        trainer = trainers.LightGBMTrainer(data=data,
                                           tournament=tournament_name)
        if load_model:
            trainer.load_from_s3(saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_to_s3(saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


def train_and_predict_catboost_model(load_model: bool, save_model: bool,
                                     submit_to_numerai: bool,
                                     params: Dict) -> None:
    """Train/load model and conduct inference"""

    tournaments = utils.get_tournament_names()
    data = utils.get_tournament_data()
    for tournament_name in tournaments:
        saved_model_name = f'catboost_prediction_model_{tournament_name}'
        trainer = trainers.CatBoostTrainer(data=data,
                                           tournament=tournament_name)
        if load_model:
            trainer.load_from_s3(saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_to_s3(saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


def train_and_predict_lstm_model(load_model: bool, save_model: bool,
                                 submit_to_numerai: bool,
                                 params: Dict) -> None:
    """Train/load model and conduct inference"""

    tournaments = utils.get_tournament_names()
    data = utils.get_tournament_data()
    for tournament_name in tournaments:
        saved_model_name = f'lstm_prediction_model_{tournament_name}'
        trainer = trainers.LSTMTrainer(data=data, tournament=tournament_name)
        if load_model:
            trainer.load_from_s3(saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_to_s3(saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


# def train_and_predict_lstm_model(submit_to_numerai: bool,
#                                  params: Dict) -> nx.Prediction:
#     """Train LSTM model and save weights"""

#     tournaments, data = prepare_tournament_data()
#     LOGGER.info(f'Training and making predictions for {tournaments}')
#     for tournament_name in tournaments:
#         model: nx.Model = trainers.train_and_save_lstm_model(
#             tournament=tournament_name,
#             data=data,
#             load_model=False,
#             save_model=True,
#             params=params)
#         predictions: nx.Prediction = make_predictions_and_prepare_submission(
#             model=model,
#             model_name='lstm',
#             data=data,
#             tournament=tournament_name,
#             submit=submit_to_numerai)
#         LOGGER.info(
#             predictions.summaries(data['validation'],
#                                   tournament=tournament_name))
#         LOGGER.info(predictions[:, tournament_name].metric_per_era(
#             data=data['validation'], tournament=tournament_name))

#     return predictions

# def train_and_predict_functional_lstm_model(
#         submit_to_numerai: bool = None) -> nx.Prediction:
#     """Trains Functional LSTM model and saves weights"""

#     tournaments, data = prepare_tournament_data()
#     LOGGER.info(f'Training and making predictions for {tournaments}')
#     for tournament_name in tournaments:
#         model: nx.Model = trainers.train_and_save_functional_lstm_model(
#             tournament=tournament_name,
#             data=data,
#             load_model=False,
#             save_model=True,
#             params=FLSTM_PARAMS)
#         predictions: nx.Prediction = make_predictions_and_prepare_submission(
#             model=model,
#             model_name='functional_lstm',
#             data=data,
#             tournament=tournament_name)
#         LOGGER.info(
#             predictions.summaries(data['validation'],
#                                   tournament=tournament_name))
#         LOGGER.info(predictions[:, tournament_name].metric_per_era(
#             data=data['validation'], tournament=tournament_name))

#     return predictions

# def train_and_predict_bidirectional_lstm_model(
#         submit_to_numerai: bool = None) -> nx.Prediction:
#     """Trains Bidirectional LSTM model and saves weights"""

#     tournaments, data = prepare_tournament_data()
#     LOGGER.info(f'Training and making predictions for {tournaments}')
#     for tournament_name in tournaments:
#         model: nx.Model = trainers.train_and_save_bidirectional_lstm_model(
#             tournament=tournament_name,
#             data=data,
#             load_model=False,
#             save_model=True,
#             params=BILSTM_PARAMS)
#         predictions: nx.Prediction = make_predictions_and_prepare_submission(
#             model=model,
#             model_name='bidirectional_lstm',
#             data=data,
#             tournament=tournament_name)
#         LOGGER.info(
#             predictions.summaries(data['validation'],
#                                   tournament=tournament_name))
#         LOGGER.info(predictions[:, tournament_name].metric_per_era(
#             data=data['validation'], tournament=tournament_name))

#     return predictions

# def train_and_predict_voting_regressor_model(
#         submit_to_numerai: bool) -> nx.Prediction:
#     """Trains VotingRegressorModel and save weights"""

#     tournaments, data = prepare_tournament_data()
#     LOGGER.info(f'Training and making predictions for {tournaments}')
#     for tournament_name in tournaments:
#         model: nx.Model = trainers.train_and_save_voting_regressor_model(
#             tournament=tournament_name,
#             data=data,
#             load_model=False,
#             save_model=True,
#         )
#         predictions: nx.Prediction = make_predictions_and_prepare_submission(
#             model=model,
#             model_name='voting_regressor',
#             data=data,
#             tournament=tournament_name,
#             submit=submit_to_numerai)
#         LOGGER.info(
#             predictions.summaries(data['validation'],
#                                   tournament=tournament_name))
#         LOGGER.info(predictions[:, tournament_name].metric_per_era(
#             data=data['validation'], tournament=tournament_name))

#     return predictions

# def train_and_predict_stacking_regressor_model(
#         submit_to_numerai: bool) -> nx.Prediction:
#     """Trains StackingRegressorModel and save weights"""

#     tournaments, data = utils.prepare_tournament_data()
#     LOGGER.info(f'Training and making predictions for {tournaments}')
#     for tournament_name in tournaments:
#         model: nx.Model = trainers.train_and_save_stacking_regressor_model(
#             tournament=tournament_name,
#             data=data,
#             load_model=False,
#             save_model=True,
#         )
#         predictions: nx.Prediction = make_predictions_and_prepare_submission(
#             model=model,
#             model_name='stacking_regressor',
#             data=data,
#             tournament=tournament_name,
#             submit=submit_to_numerai)
#         LOGGER.info(
#             predictions.summaries(data['validation'],
#                                   tournament=tournament_name))
#         LOGGER.info(predictions[:, tournament_name].metric_per_era(
#             data=data['validation'], tournament=tournament_name))

#     return predictions


@click.command()
@click.option('-m', '--model', type=str, default='xgboost')
@click.option('-lm', '--load-model', type=bool, default=True)
@click.option('-sm', '--save-model', type=bool, default=False)
@click.option('-s', '--submit', type=bool, default=False)
def main(model: str, load_model: bool, save_model: bool,
         submit: bool) -> nx.Prediction:

    if model == 'xgboost':
        XGBOOST_PARAMS = {
            "max_depth": 7,
            "learning_rate": 0.00123,
            "l2": 0.015,
            "n_estimators": 2511
        }
        return train_and_predict_xgboost_model(load_model=load_model,
                                               save_model=save_model,
                                               submit_to_numerai=submit,
                                               params=XGBOOST_PARAMS)
    if model == 'lightgbm':
        LIGHTGBM_PARAMS = {
            "n_estimators": 1234,
            "learning_rate": 0.00111325,
            "reg_lambda": 0.0111561
        }
        return train_and_predict_lightgbm_model(load_model=load_model,
                                                save_model=save_model,
                                                submit_to_numerai=submit,
                                                params=LIGHTGBM_PARAMS)
    if model == 'catboost':
        CATBOOST_PARAMS = {
            'depth': 5,
            'learning_rate': 0.001,
            'l2': 0.01,
            'iterations': 2000
        }
        return train_and_predict_catboost_model(load_model=load_model,
                                                save_model=save_model,
                                                submit_to_numerai=submit,
                                                params=CATBOOST_PARAMS)
    if model == 'lstm':
        LSTM_PARAMS = {'epochs': 1, 'batch_size': 1, 'time_steps': 1}
        return train_and_predict_lstm_model(load_model=load_model,
                                            save_model=save_model,
                                            submit_to_numerai=submit,
                                            params=LSTM_PARAMS)
    # if model == 'flstm':
    #     FLSTM_PARAMS = {'epochs': 1, 'batch_size': 120}
    #     return train_and_predict_functional_lstm_model(submit)
    # if model == 'bilstm':
    #     BILSTM_PARAMS = {'epochs': 1, 'batch_size': 120}
    #     return train_and_predict_bidirectional_lstm_model(submit)
    # if model == 'voting_regressor':
    #     return train_and_predict_voting_regressor_model(submit)
    # if model == 'stacking_regressor':
    #     return train_and_predict_stacking_regressor_model(submit)


if __name__ == "__main__":
    predictions = main()
    LOGGER.info(predictions.shape)
    LOGGER.info(predictions)