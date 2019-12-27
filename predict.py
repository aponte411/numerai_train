
import numerox as nx
import numerapi
import os
import models
from typing import Any, Tuple, List
import click
from utils import get_logger, prepare_tournament_data

LOGGER = get_logger(__name__)


def train_and_predict_lstm_model() -> Any:
    """Train LSTM model and save weights"""

    def train_and_save_model(tournament: str, data: nx.data.Data) -> nx.Model:
        """Train and persist model weights"""

        saved_model_name = f'lstm_prediction_model_{tournament_name}'
        if os.path.exists(saved_model_name):
            LOGGER.info(f"using saved model for {tournament_name}")
            model = models.LSTMModel().load(saved_model_name)
        else:
            LOGGER.info(f"Saved model not found for {tournament_name}")
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

    def make_predictions_and_prepare_submission(
        model: nx.Model, 
        data: nx.data.Data, 
        tournament: str) -> nx.Prediction:
        """Make predictions and save to CSV"""

        LOGGER.info(f"Making predictions with LSTM .predict() method")
        yhat = model.predict(data['tournament'], tournament)
        # prediction: nx.Prediction = model.predict(data['tournament'], tournament)
        # prediction_filename: str = f'/tmp/lstm_prediction_{tournament}.csv'
        # LOGGER.info(f"Saving predictions to CSV: {prediction_filename}")
        # prediction.to_csv(prediction_filename)

        return yhat

    tournaments, data = prepare_tournament_data()
    LOGGER.info(f'Training and making predictions for {tournaments}')
    for tournament_name in tournaments:
        model: nx.Model = train_and_save_model(
            tournament=tournament_name, 
            data=data
            )
        predictions: nx.Prediction = make_predictions_and_prepare_submission(
            model=model,
            data=data,
            tournament=tournament_name
            )

    return predictions


if __name__ == "__main__":
    predictions = train_and_predict_lstm_model()
    print(predictions)