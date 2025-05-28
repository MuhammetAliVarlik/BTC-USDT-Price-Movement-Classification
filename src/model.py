from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC

def build_cnn_lstm_model(input_shape):
    """
    Builds a CNN-LSTM model for binary classification of time series data.

    Args:
        input_shape (tuple): Shape of input data (timesteps, features).

    Returns:
        keras.Model: Compiled CNN-LSTM model.
    """

    model = Sequential()

    # 1D Convolution layer: extract local temporal features from input sequences
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))

    # Another Conv1D layer to deepen feature extraction
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))

    # LSTM layer: captures sequential dependencies and long-term patterns in features
    # return_sequences=False outputs last output only, feeding into Dense layer
    model.add(LSTM(128, activation='tanh', return_sequences=False))

    # Dropout layer to reduce overfitting by randomly setting 30% of inputs to zero
    model.add(Dropout(0.3))

    # Final Dense layer with sigmoid activation for binary classification output (0 or 1)
    model.add(Dense(1, activation='sigmoid'))

    # Compile model with:
    # - binary_crossentropy loss (suitable for binary classification)
    # - Adam optimizer with learning rate 0.001
    # - Metrics: accuracy, precision, recall, and AUC for model evaluation
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )

    # Print model architecture summary
    model.summary()

    return model
