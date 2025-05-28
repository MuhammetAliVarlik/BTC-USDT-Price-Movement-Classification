import os
from src import load_bitcoin_ohlcv, add_technical_indicators, build_cnn_lstm_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load Bitcoin OHLCV (Open, High, Low, Close, Volume) data
print("Loading Bitcoin OHLCV data...")
df, frame = load_bitcoin_ohlcv()

# Add technical indicators to the dataset
print("Adding technical indicators...")
df = add_technical_indicators(df, frame=frame)

# Drop rows with missing values that may have appeared after adding indicators and reset the index
print("Dropping NaN values and resetting index...")
df = df.dropna().reset_index(drop=True)

# Define the list of features (technical indicators) to be used for the model
features = [
    "MACD_line",
    "MACD_signal",
    "RSI_28h",
    "Williams_%R_28h",
    "Momentum_14h",
    "Momentum_10h",
    "Momentum_5h",
    "Stoch_%K",
    "Stoch_%D",
    "Williams_%R_14h",
    "RSI_14h",
    "CCI_14h"
]

# Extract feature matrix X and target vector y from the dataframe
print("Extracting features and target...")
X = np.array(df[features])
y = np.array(df['Target'])

# Scale features using StandardScaler (mean=0, std=1)
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define sequence length based on frame rate (e.g. 6 hours worth of data)
sequence_length = int(6 * 60 / frame)

# Split data into training, validation, and test sets (70%, 15%, 15%)
print("Splitting data into train, validation, and test sets...")
X_train, y_train = X_scaled[:int(len(X_scaled)*0.7)], y[:int(len(y)*0.7)]
X_val, y_val = X_scaled[int(len(X_scaled)*0.7):int(len(X_scaled)*0.85)], y[int(len(y)*0.7):int(len(y)*0.85)]
X_test, y_test = X_scaled[int(len(X_scaled)*0.85):], y[int(len(y)*0.85):]

# Create time series generators for Keras model training and evaluation
print("Creating time series generators...")
train_generator = TimeseriesGenerator(X_train, y_train, length=sequence_length, batch_size=32)
val_generator = TimeseriesGenerator(X_val, y_val, length=sequence_length, batch_size=32)
test_generator = TimeseriesGenerator(X_test, y_test, length=sequence_length, batch_size=32)

# Path to the saved model file
model_path = 'models/bitcoin_cnn_lstm_model.h5'

#  Build and train a new model

print("Building the CNN-LSTM model...")
model = build_cnn_lstm_model(input_shape=(sequence_length, len(features)))

print("Starting model training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    verbose=1
)

print("Saving the model...")
os.makedirs('models', exist_ok=True)
model.save(model_path)
print("Model saved successfully.")

print("Plotting training history...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Save the plots to the 'docs' directory
os.makedirs('docs/training', exist_ok=True)
plot_path = os.path.join('docs/training', 'training_history.png')
plt.savefig(plot_path)
print(f"Training history plot saved to {plot_path}")

# Evaluate the trained model on the test dataset
print("Evaluating model on test set...")
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Predict labels for the test set to calculate the F1 score
print("Predicting test set labels for F1 score calculation...")
y_test_pred_prob = model.predict(test_generator)
y_test_pred = (y_test_pred_prob > 0.5).astype(int).flatten()
y_test_true = y_test[sequence_length:]

print(f"Test F1 Score: {f1_score(y_test_true, y_test_pred):.4f}")

cr = classification_report(y_test_true, y_test_pred)
print("Classification Report:")
print(cr)

print("Generating confusion matrix...")
cm = confusion_matrix(y_test_true, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
conf_matrix_path = os.path.join('docs/training', 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
print(f"Confusion matrix saved to {conf_matrix_path}")
