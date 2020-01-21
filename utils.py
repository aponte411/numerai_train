import logging
import numpy as np
import pandas as pd
from typing import Tuple, Any, List
import os
import numerox as nx
import boto3
from boto3.s3 import transfer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s = %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name, level=logging.INFO) -> Any:
    """Returns logger object with given name"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


LOGGER = get_logger(__name__)


def get_tournament_data() -> nx.data.Data:
    """Downloads latest the tournament data from numerox"""
    try:
        data: nx.data.Data = nx.download('numerai_dataset.zip')
    except Exception as e:
        LOGGER.info(f'Failure to download numerai data with {e}')
        raise e

    return data


def get_tournament_names() -> List[str]:
    """Get latest tournament names"""

    return nx.tournament_names()


def evaluate_predictions(predictions: nx.Prediction, trainer: Any,
                         tournament: str) -> None:
    """Evaluate the validation set predictions"""

    LOGGER.info(
        predictions.summaries(trainer.data['validation'],
                              tournament=tournament))
    LOGGER.info(predictions[:, tournament].metric_per_era(
        data=trainer.data['validation'], tournament=tournament))


"""
Helper functions that require datasets to be in pandas dataframes
format. You can use get_all_data_for_model() to return train, val,
and test set.
"""


def get_correlations(training_data: pd.DataFrame,
                     tournament_data: pd.DataFrame) -> Any:
    """Assesses performance on the training and validation data"""

    train_correlations = training_data.groupby("era").apply(score)
    print(
        f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}"
    )
    print(
        f"On training the average per-era payout is {payout(train_correlations).mean()}"
    )

    validation_data = tournament_data.loc[tournament_data.data_type ==
                                          "validation"]
    validation_correlations = validation_data.groupby("era").apply(score)
    print(
        f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}"
    )
    print(
        f"On validation the average per-era payout is {payout(validation_correlations).mean()}"
    )


# TEMP, NEED TO FIX
def score(df) -> np.array:
    """
    Submissions are scored by spearman correlation.
    method="first" breaks ties based on order in array
    """

    # yhat = pd.Series(predictions)
    PREDICTION_NAME = "prediction_kazutsugi"
    yhat = df[PREDICTION_NAME]
    ranked_predictions = yhat.rank(pct=True, method="first")
    correlation = np.corrcoef(yhat, ranked_predictions)[0, 1]

    return correlation


def payout(scores) -> float:
    """The payout function"""

    BENCHMARK = 0
    BAND = 0.2

    return ((scores - BENCHMARK) / BAND).clip(lower=-1, upper=1)


def get_latest_val_data() -> pd.DataFrame:

    full_test = pd.read_csv(
        "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz"
    )
    print(f"Loaded test data with {full_test.shape[0]} rows "
          f"and {full_test.shape[1]} columns")

    val = full_test.loc[full_test.data_type == 'validation']
    del full_test

    print(f'Validation: {val.shape}')
    print()

    return val


def get_latest_test_data() -> pd.DataFrame:

    full_test = pd.read_csv(
        "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz"
    )
    print(f"Loaded test data with {full_test.shape[0]} rows "
          f"and {full_test.shape[1]} columns")

    test = full_test.loc[full_test.data_type == 'test']
    del full_test

    print(f'Test: {test.shape}')
    print()

    return test


def get_latest_training_data() -> pd.DataFrame:
    """Downloads the latest datasets"""

    print()
    train = pd.read_csv(
        "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz"
    )
    print(f"Loaded training data with {train.shape[0]} rows "
          f"and {train.shape[1]} columns")

    print()
    print(f'Train: {train.shape}')

    return train


def get_all_data_for_model() -> Tuple:
    """Gets Train, Val, and Test data"""

    train = get_latest_training_data()
    val = get_latest_val_data()
    test = get_latest_test_data()

    return train, val, test


class S3Client:
    """
    Sets up a client to download/upload
    s3 files.

    : aws access key id: login
    : aws secret access key: password
    : bucket: path to file
    """
    def __init__(self,
                 user=os.environ["AWS_ACCESS_KEY_ID"],
                 password=os.environ["AWS_SECRET_ACCESS_KEY"],
                 bucket=os.environ["BUCKET"]):
        self.bucket = bucket
        self.client = boto3.client('s3',
                                   aws_access_key_id=user,
                                   aws_secret_access_key=password)

    def upload_file(self, filename: str, key: str) -> None:

        s3t = transfer.S3Transfer(self.client)
        s3t.upload_file(filename, self.bucket, key)
        LOGGER.info('File successfully uploaded!')

    def download_file(self, filename: str, key: str) -> None:

        s3t = transfer.S3Transfer(self.client)
        s3t.download_file(self.bucket, key, filename)
        LOGGER.info('File successfully downloaded!')
