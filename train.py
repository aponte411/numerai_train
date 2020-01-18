import numerox as nx
import numerapi
import os
import models
from typing import Any, Tuple, List, Dict
import click
from utils import get_logger, prepare_tournament_data, S3Client
from metrics import correlations

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import (Pipeline, make_pipeline, FeatureUnion,
                              make_union)
from sklearn.feature_selection import SelectFromModel

LOGGER = get_logger(__name__)

CURRENT_TOURNAMENT = 'kazutsugi'


def train_linear_model() -> None:
    """Train model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(tournaments)

    for tournament_name in tournaments:
        model = models.LinearModel()
        try:
            LOGGER.info(f"fitting model for {tournament_name}")
            model.fit(data['train'], tournament_name)
        except Exception as e:
            LOGGER.error(f"Training failed with {e}")
            raise e

        LOGGER.info(f"saving model for {tournament_name}")
        model.save(f'model_trained_{tournament_name}')


def train_and_save_xgboost_model(tournament: str,
                                 data: nx.data.Data,
                                 load_model: bool = True,
                                 save_model: bool = True,
                                 params: Dict = None) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'xgboost_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.XGBoostModel().load(saved_model_name)
    else:
        LOGGER.info(f"Building XGBoostModel from scratch for {tournament}")
        model = models.XGBoostModel(max_depth=params["max_depth"],
                                    learning_rate=params["learning_rate"],
                                    l2=params["l2"],
                                    n_estimators=params["n_estimators"])
        LOGGER.info(f"Training XGBoost model for {tournament}")
        eval_set = [(data['validation'].x, data['validation'].y[tournament])]
        model.fit(dfit=data['train'], tournament=tournament, eval_set=eval_set)
    if save_model:
        LOGGER.info(f"Saving model for {tournament} locally")
        model.save(f'xgboost_prediction_model_{tournament}')
        LOGGER.info(f"Saving model for {tournament} to s3 bucket")
        model.save_to_s3(filename=saved_model_name, key=saved_model_name)

    return model


def train_and_save_lstm_model(tournament: str,
                              data: nx.data.Data,
                              load_model: bool,
                              save_model: bool,
                              params: Dict = None) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'lstm_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.LSTMModel().load(saved_model_name)
    else:
        LOGGER.info(f"Building LSTMModel from scratch for {tournament}")
        model = models.LSTMModel(time_steps=1)
        LOGGER.info(f"Training LSTM model for {tournament}")
        eval_set = (data['validation'].x, data['validation'].y[tournament])
        model.fit(dfit=data['train'],
                  tournament=tournament,
                  eval_set=eval_set,
                  epochs=params['epochs'],
                  batch_size=params['batch_size'])
    if save_model:
        LOGGER.info(f"Saving model for {tournament} locally")
        model.save(saved_model_name)
        LOGGER.info(f"Saving model for {tournament} to s3 bucket")
        model.save_to_s3(filename=saved_model_name, key=saved_model_name)
    return model


def train_and_save_functional_lstm_model(tournament: str,
                                         data: nx.data.Data,
                                         load_model: bool = True,
                                         save_model: bool = False) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'functional_lstm_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.FunctionalLSTMModel().load(saved_model_name)
    else:
        LOGGER.infp(
            f"Building FunctionalLSTMModel from scratch for {tournament}")
        model = models.FunctionalLSTMModel(time_steps=1)
        LOGGER.info(f"Training Functional LSTM model for {tournament}")
        eval_set = (data['validation'].x, data['validation'].y[tournament])
        model.fit(dfit=data['train'],
                  tournament=tournament,
                  eval_set=eval_set,
                  epochs=1,
                  batch_size=30)
    if save_model:
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'functional_lstm_prediction_model_{tournament}')

    return model


def train_and_save_bidirectional_lstm_model(tournament: str,
                                            data: nx.data.Data,
                                            load_model: bool,
                                            save_model: bool) -> nx.Model:
    """Trains Bidirectional LSTM model and saves weights"""

    saved_model_name = f'bidirectional_lstm_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.BidirectionalLSTMModel().load(saved_model_name)
    else:
        LOGGER.info(
            f"Building BidirectionalLSTMModel from scratch for {tournament}")
        model = models.BidirectionalLSTMModel(time_steps=1)
        LOGGER.info(f"Training BidirectionalLSTM model for {tournament}")
        eval_set = (data['validation'].x, data['validation'].y[tournament])
        model.fit(dfit=data['train'],
                  tournament=tournament,
                  eval_set=eval_set,
                  epochs=1,
                  batch_size=30)
    if save_model:
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'bidirectional_lstm_prediction_model_{tournament}')

    return model


def train_and_save_catboost_model(tournament: str,
                                  data: nx.data.Data,
                                  load_model: bool = True,
                                  save_model: bool = True,
                                  params: Dict = None) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'catboost_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.CatBoostRegressorModel().load(saved_model_name)
    else:
        LOGGER.info(
            f"Training CatBoostRegressorModel from scratch for {tournament} tournament"
        )
        model = models.CatBoostRegressorModel(
            depth=params['depth'],
            learning_rate=params['learning_rate'],
            l2_leaf_reg=params['l2'],
            iterations=params['iterations'],
        )
        LOGGER.info(f"Training CatBoostRegressorModel for {tournament}")
        eval_set = (data['validation'].x, data['validation'].y[tournament])
        model.fit(dfit=data['train'], tournament=tournament, eval_set=eval_set)
    if save_model:
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'catboost_prediction_model_{tournament}')

    return model


def train_and_save_lightgbm_model(tournament: str,
                                  data: nx.data.Data,
                                  load_model: bool = True,
                                  save_model: bool = True,
                                  params: Dict = None) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'lightgbm_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.LightGBMRegressorModel().load(saved_model_name)
    else:
        LOGGER.info(
            f"Building LightGBMRegressorModel from scratch for {tournament}")
        model = models.LightGBMRegressorModel(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            reg_lambda=params['reg_lambda'])
        LOGGER.info(f"Training LightGBMRegressor model for {tournament}")
        eval_set = (data['validation'].x, data['validation'].y[tournament])
        model.fit(dfit=data['train'], tournament=tournament, eval_set=eval_set)
    if save_model:
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'lightgbm_prediction_model_{tournament}')

    return model


def train_and_save_voting_regressor_model(tournament: str,
                                          data: nx.data.Data,
                                          load_model: bool = True,
                                          save_model: bool = True) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'voting_regressor_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.VotingRegressorModel().load(saved_model_name)
    else:
        LOGGER.info(
            f"Training VotingRegressorModel from scratch for {tournament} tournament"
        )
        model = models.VotingRegressorModel()
        LOGGER.info(f"Training VotingRegressorModel for {tournament}")
        model.fit(dfit=data['train'], tournament=tournament)
    if save_model:
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'voting_regressor_prediction_model_{tournament}')

    return model


def train_and_save_stacking_regressor_model(
        tournament: str,
        data: nx.data.Data,
        load_model: bool = True,
        save_model: bool = True) -> nx.Model:
    """Train and persist model weights"""

    saved_model_name = f'stacking_regressor_prediction_model_{tournament}'
    if load_model:
        LOGGER.info(f"using saved model for {tournament}")
        model = models.StackingRegressorModel().load(saved_model_name)
    else:
        LOGGER.info(
            f"Training StackingRegressorModel from scratch for {tournament} tournament"
        )
        model = models.StackingRegressorModel()
        LOGGER.info(f"Training StackingRegressorModel for {tournament}")
        model.fit(dfit=data['train'], tournament=tournament)
    if save_model:
        LOGGER.info(f"Saving model for {tournament}")
        model.save(f'stacking_regressor_prediction_model_{tournament}')

    return model


@click.command()
@click.option('-m', '--model', type=str, default='xgboost')
@click.option('-t', '--tournament', type=str, default=CURRENT_TOURNAMENT)
@click.option('-l', '--load-model', type=bool, default=True)
@click.option('-s', '--save-model', type=bool, default=False)
def main(model: str, tournament: str, load_model: bool,
         save_model: bool) -> None:
    """Selects which model to train"""

    tournaments, data = prepare_tournament_data()

    if model.lower() == 'xgboost':
        trained_model = train_and_save_xgboost_model(tournament=tournament,
                                                     data=data,
                                                     load_model=load_model,
                                                     save_model=save_model)
    if model.lower() == 'catboost':
        trained_model = train_and_save_catboost_model(tournament=tournament,
                                                      data=data,
                                                      load_model=load_model,
                                                      save_model=save_model)
    if model.lower() == 'lightgbm':
        trained_model = train_and_save_lightgbm_model(tournament=tournament,
                                                      data=data,
                                                      load_model=load_model,
                                                      save_model=save_model)
    if model.lower() == 'lstm':
        trained_model = train_and_save_lstm_model(tournament=tournament,
                                                  data=data,
                                                  load_model=load_model,
                                                  save_model=save_model)
    if model.lower() == 'flstm':
        trained_model = train_and_save_functional_lstm_model(
            tournament=tournament,
            data=data,
            load_model=load_model,
            save_model=save_model)
    if model.lower() == 'bilstm':
        trained_model = train_and_save_bidirectional_lstm_model(
            tournament=tournament,
            data=data,
            load_model=load_model,
            save_model=save_model)
    if model.lower() == 'linear':
        train_linear_model()


if __name__ == "__main__":
    main()
