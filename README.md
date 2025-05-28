# 📈 BTC/USDT Price Movement Classification

A deep learning-based time series classification project designed to predict short-term BTC/USDT price direction using technical indicators and a hybrid CNN-LSTM architecture.

---

## 📌 Objective

This project aims to classify whether the price of BTC/USDT will rise or fall within the next 6 hours using historical candlestick data and derived technical indicators. The task is framed as a binary classification problem.

---

## 🧠 Approach

1. **Data Preparation**  
   30-minute interval OHLCV data is used to generate features and labels. The target label is defined as:
   - 1 if the price after 6 hours is higher than the current price
   - 0 otherwise

2. **Feature Engineering**  
   A rich set of technical indicators is generated, including:
   - Moving Averages (SMA, EMA)
   - Momentum Indicators (RSI, MACD, Stochastic, Williams %R)
   - Volume Indicators (ADI)
   - Oscillators (CCI)
   
   Point-biserial correlation is computed to evaluate the predictive power of each feature with respect to the binary target.

3. **Model Architecture**  
   A hybrid model combining CNN and LSTM layers is implemented:
   - Conv1D layers to extract local patterns
   - LSTM layer to capture temporal dependencies
   - Dropout and Dense layers for regularization and classification

4. **Training Strategy**  
   - The dataset is split into Train (70%), Validation (15%), and Test (15%) sets.
   - Features are standardized using `StandardScaler`.
   - `TimeseriesGenerator` is used to provide sequential inputs.
   - Learning rate scheduling and early stopping are applied to optimize training.

---

## 📊 Evaluation

Model performance is assessed using:
- Accuracy
- F1 Score
- Confusion Matrix
- Precision & Recall
- ROC AUC

**Validation Results**  
```yaml 
- Accuracy: ~51.21%  
- F1 Score: ~0.59
```


**Test Results**  
```yaml 
- Accuracy: ~52.0% 
- F1 Score: ~0.57
- Loss : ~0.69
```

```yaml
Classification Report:
              precision    recall  f1-score   support

           0       0.48      0.44      0.46      1195
           1       0.55      0.59      0.57      1411

    accuracy                           0.52      2606
   macro avg       0.52      0.52      0.51      2606
weighted avg       0.52      0.52      0.52      2606

```

The model shows a modest improvement over random guessing, with slightly better performance on upward movement prediction.

---
## Project Structure
```yaml

├── main.py # Main script to run the model training and evaluation
├── environment.yml # Conda environment specification for reproducibility
├── models/ # Pretrained and saved model files
├── notebooks/
│           ├── btc-price-movement-classification.ipynb # Core Jupyter notebook for exploration & experimentation
│           └── tools/
│                   └── LearningPlot.py # Custom utility for visualization during training
└── src/
        ├── data_utils.py # Data fetching and preprocessing utilities
        ├── feature_engineering.py # Technical indicator calculation and feature engineering modules
        └── model.py # Model architectures, training, and evaluation scripts

```
---
## ⚙️ Installation & Setup

Follow the steps below to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/MuhammetAliVarlik/BTC-USDT-Price-Movement-Classification.git
cd BTC-USDT-Price-Movement-Classification
```

### 2. Create the Conda Environment
- The environment includes TensorFlow, NumPy, scikit-learn, pandas, matplotlib, seaborn, and other dependencies.

```bash
conda env create -f environment.yml
conda activate btc-prediction
```
### 3. Run the Main Script
- Once the environment is active, execute the main pipeline:

```bash
python main.py
```
