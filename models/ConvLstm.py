from tensorflow import keras
from tensorflow.keras import layers
def ConvLstm():
    seq = keras.Sequential(
        [
            keras.Input(
                shape=(None, 512, 832, 3)
            ),  # Variable-length sequence of 40x40x1 frames
            layers.ConvLSTM2D(
                filters=3, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=3, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=3, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=3, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.Conv3D(
                filters=3, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
            ),
        ]
    )
    seq.compile(loss="binary_crossentropy", optimizer="adadelta")
    return seq
