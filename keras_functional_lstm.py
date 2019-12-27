import tensorflow as tf


class LstmModel(tf.keras.Model):
    """
    Tensorflow implementation of LSTM
    architecture using Keras API. Layers 
    are joined using Functional approach
    within the .call() method.
    """

    def __init__(self, timesteps=1):
        super(LstmModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(
            units=300, 
            return_sequences=True, 
            kernel_initializer=tf.initializers.VarianceScaling,
            input_shape=(timesteps, 310)
            )
        self.lstm2 = tf.keras.layers.LSTM(
            units=150, 
            return_sequences=True, 
            kernel_initializer=tf.initializers.VarianceScaling
            )
        self.lstm3 = tf.keras.layers.LSTM(
            units=timesteps, 
            return_sequences=True,
            kernel_initializer=tf.initializers.VarianceScaling
            )
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, x):

        x = self.lstm1(x)
        x = self.dropout(x)
        x = self.lstm2(x)
        x = self.dropout(x)
        x = self.lstm3(x)
        x = self.dropout(x)

        return self.dense(x) 