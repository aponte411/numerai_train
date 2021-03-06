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
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
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
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
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
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
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
        trainer = trainers.LSTMTrainer(data=data,
                                       tournament=tournament_name,
                                       gpu=1)
        if load_model:
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


def train_and_predict_funclstm_model(load_model: bool, save_model: bool,
                                     submit_to_numerai: bool,
                                     params: Dict) -> None:
    """Train/load model and conduct inference"""

    tournaments = utils.get_tournament_names()
    data = utils.get_tournament_data()
    for tournament_name in tournaments:
        saved_model_name = f'funclstm_prediction_model_{tournament_name}'
        trainer = trainers.FunctionalLSTMTrainer(data=data,
                                                 tournament=tournament_name,
                                                 gpu=1)
        if load_model:
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


def train_and_predict_bilstm_model(load_model: bool, save_model: bool,
                                   submit_to_numerai: bool,
                                   params: Dict) -> None:
    """Train/load model and conduct inference"""

    tournaments = utils.get_tournament_names()
    data = utils.get_tournament_data()
    for tournament_name in tournaments:
        saved_model_name = f'bilstm_prediction_model_{tournament_name}'
        trainer = trainers.BidirectionalLSTMTrainer(data=data,
                                                    tournament=tournament_name,
                                                    gpu=1)
        if load_model:
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


def train_and_predict_voting_regressor_model(load_model: bool,
                                             save_model: bool,
                                             submit_to_numerai: bool,
                                             params: Dict) -> None:
    """Train/load model and conduct inference"""

    tournaments = utils.get_tournament_names()
    data = utils.get_tournament_data()
    for tournament_name in tournaments:
        saved_model_name = f'voting_regressor_prediction_model_{tournament_name}'
        trainer = trainers.VotingRegressorTrainer(data=data,
                                                  tournament=tournament_name)
        if load_model:
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


def train_and_predict_stacking_regressor_model(load_model: bool,
                                               save_model: bool,
                                               submit_to_numerai: bool,
                                               params: Dict) -> None:
    """Train/load model and conduct inference"""

    tournaments = utils.get_tournament_names()
    data = utils.get_tournament_data()
    for tournament_name in tournaments:
        saved_model_name = f'stacking_regressor_prediction_model_{tournament_name}'
        trainer = trainers.StackingRegressorTrainer(data=data,
                                                    tournament=tournament_name)
        if load_model:
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


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
            "learning_rate": 0.000123,
            "l2": 0.02,
            "n_estimators": 3000,
            "tree_method": "gpu_hist"
        }
        return train_and_predict_xgboost_model(load_model=load_model,
                                               save_model=save_model,
                                               submit_to_numerai=submit,
                                               params=XGBOOST_PARAMS)
    if model == 'lightgbm':
        LIGHTGBM_PARAMS = {
            "n_estimators": 1234,
            "learning_rate": 0.01,
            "reg_lambda": 0.1
        }
        return train_and_predict_lightgbm_model(load_model=load_model,
                                                save_model=save_model,
                                                submit_to_numerai=submit,
                                                params=LIGHTGBM_PARAMS)
    if model == 'catboost':
        CATBOOST_PARAMS = {
            'depth': 7,
            'learning_rate': 0.0123,
            'l2': 0.1,
            'iterations': 2000,
            'task_type': 'GPU'
        }
        return train_and_predict_catboost_model(load_model=load_model,
                                                save_model=save_model,
                                                submit_to_numerai=submit,
                                                params=CATBOOST_PARAMS)
    if model == 'lstm':
        LSTM_PARAMS = {
            'training_epochs': 5,
            'epochs': 120,
            'batch_size': 1,
            'time_steps': 1
        }
        return train_and_predict_lstm_model(load_model=load_model,
                                            save_model=save_model,
                                            submit_to_numerai=submit,
                                            params=LSTM_PARAMS)
    if model == 'funclstm':
        FLSTM_PARAMS = {'epochs': 1, 'batch_size': 120}
        return train_and_predict_funclstm_model(load_model=load_model,
                                                save_model=save_model,
                                                submit_to_numerai=submit,
                                                params=FLSTM_PARAMS)
    if model == 'bilstm':
        BILSTM_PARAMS = {'epochs': 1, 'batch_size': 120}
        return train_and_predict_bilstm_model(load_model=load_model,
                                              save_model=save_model,
                                              submit_to_numerai=submit,
                                              params=BILSTM_PARAMS)
    if model == 'voting_regressor':
        return train_and_predict_voting_regressor_model(
            load_model=load_model,
            save_model=save_model,
            submit_to_numerai=submit)
    if model == 'stacking_regressor':
        return train_and_predict_stacking_regressor_model(
            load_model=load_model,
            save_model=save_model,
            submit_to_numerai=submit)


if __name__ == "__main__":
    predictions = main()
    LOGGER.info(predictions.shape)
    LOGGER.info(predictions)