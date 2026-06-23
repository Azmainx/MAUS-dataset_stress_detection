"""
model.py — 1D-CNN -> BiLSTM -> Self-Attention -> Dense head, built in tf.keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import keras

import config as cfg


@keras.saving.register_keras_serializable(package="maus_pipeline")
class AttentionLayer(layers.Layer):
    """Additive (Bahdanau-style) self-attention over the time axis.

    Input:  (batch, T, D)  -- BiLSTM output sequence
    Output: (batch, D)     -- context vector (weighted sum over T)
            attention_weights (batch, T, 1)  exposed via `last_attention_weights`
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)
        self.last_attention_weights = None

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    def call(self, inputs):
        # inputs: (batch, T, D)
        score = self.V(self.W(inputs))               # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)         # (batch, T, 1)
        self.last_attention_weights = weights
        context = tf.reduce_sum(inputs * weights, axis=1)  # (batch, D)
        return context


def build_model(input_len, n_channels=3, n_classes=1, task="binary"):
    inputs = layers.Input(shape=(input_len, n_channels), name="signal_input")

    x = inputs
    # ---- 1D-CNN block (2 layers) ----
    for filters, kernel in zip(cfg.CNN_FILTERS, cfg.CNN_KERNELS):
        x = layers.Conv1D(filters, kernel, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(cfg.POOL_SIZE)(x)

    # ---- BiLSTM block (2 layers) ----
    for i in range(cfg.LSTM_LAYERS):
        x = layers.Bidirectional(
            layers.LSTM(cfg.LSTM_UNITS, return_sequences=True)
        )(x)

    # ---- Self-attention ----
    attn = AttentionLayer(cfg.ATTENTION_UNITS, name="attention")
    context = attn(x)

    # ---- Dense + Dropout head ----
    h = context
    for units in cfg.DENSE_UNITS:
        h = layers.Dense(units, activation="relu")(h)
        h = layers.Dropout(cfg.DROPOUT)(h)

    if task == "binary":
        outputs = layers.Dense(1, activation="sigmoid", name="output")(h)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(n_classes, activation="softmax", name="output")(h)
        loss = "sparse_categorical_crossentropy"

    model = models.Model(inputs, outputs, name="CNN_BiLSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_model(input_len=cfg.WINDOW_LEN // cfg.DOWNSAMPLE_FACTOR, task="binary")
    m.summary()
