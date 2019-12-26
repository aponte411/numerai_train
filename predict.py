
import numerox as nx
import numerapi
import os
import models
from typing import Any, Tuple
import click

from utils import get_logger, prepare_tournament_data

LOGGER = get_logger(__name__)


def train_and_predict_lstm_model() -> Any:
    """Train LSTM model and save weights"""

    tournaments, data = prepare_tournament_data()
    LOGGER.info(tournaments)

    for tournament_name in tournaments:
        eval_set = (
            data['validation'].x, data['validation'].y[tournament_name]
        )
        model = models.LSTMModel(time_steps=1)

        LOGGER.info(f"Training LSTM model for {tournament_name}")
        model.fit(
            dfit=data['train'], 
            tournament=tournament_name,
            eval_set=eval_set,
            epochs=1,
            batch_size=30
        )

        LOGGER.info(f"Making predictions with LSTM .predict() method")
        prediction: nx.Prediction = model.predict(data['tournament'], tournament_name)
        prediction_filename = f'/tmp/lstm_prediction_{tournament_name}.csv'
        LOGGER.info(f"Saving predictions to CSV: {prediction_filename}")
        prediction.to_csv(prediction_filename)

        # LOGGER.info(f"Making predictions with LSTM .fit_predict() method")
        # production_response = nx.production( 
        #     model, data, tournament=tournament_name
        # )
        # LOGGER.info(f'Production response has {production_response.shape} shape')

        return predictions


if __name__ == "__main__":
    predictions = train_and_predict_lstm_model()
    print(predictions)