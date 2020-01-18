import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf
from scipy.stats import spearmanr
from typing import Tuple


def get_spearman_rankcor(y_true, y_pred):
    return (tf.py_function(
        spearmanr, [tf.cast(y_pred, tf.float32),
                    tf.cast(y_true, tf.float32)],
        Tout=tf.float32))


def correlation_metric(y_true, y_pred):

    mx = tf.math.reduce_mean(y_true, y_pred)
    my = tf.math.reduce_mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)

    return r_num / r_den


def correlation_coefficient_loss(y_true: tf.Tensor,
                                 y_pred: tf.Tensor) -> float:
    """Spearman correlation coefficient"""

    x = y_true
    y = y_pred
    # y = tf.dtypes.cast(
    #     tf.rank(y_pred),
    #     tf.float32
    #     )
    # mx = K.mean(x)
    # my = K.mean(y)
    xm, ym = x - K.mean(x), y - K.mean(y)
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)

    return 1 - K.square(r)


def correlations(y_true, dtrain) -> float:
    """Spearman correlation for XGBoost"""

    y = dtrain.get_label()
    ranked_prediction = pd.Series(y).rank(pct=True, method='first')

    return "Spearman Correlation", np.corrcoef(y_true,
                                               ranked_prediction.values)[0, 1]


def payout(y_true, y_pred) -> float:
    """The payout function"""
    def _correlations(y_pred) -> float:
        ranked_predictions = pd.Series(y_pred).rank(pct=True, method="first")
        return np.corrcoef(y_pred, ranked_predictions)[0, 1]

    scores = _correlations(y_pred=y_pred)
    BENCHMARK = 0
    BAND = 0.2

    return ((scores - BENCHMARK) / BAND).clip(lower=-1, upper=1)
