import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import os
import joblib

def prepare_lstm_data(data: pd.Series, window_size: int = 30):
    """
    Prepare data for LSTM: scales and windowed sequences.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(price_series: pd.Series, model_path: str = "models/lstm_model.h5", window_size: int = 30):
    X, y, scaler = prepare_lstm_data(price_series, window_size)
    model = build_lstm_model((X.shape[1], 1))

    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, callbacks=[es], verbose=1)

    model.save(model_path)
    joblib.dump(scaler, model_path.replace(".h5", "_scaler.pkl"))
    print(f"LSTM model and scaler saved to {model_path}")

def forecast_next_days(price_series: pd.Series, model_path: str, days: int = 30, window_size: int = 30):
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    scaler = joblib.load(model_path.replace(".h5", "_scaler.pkl"))
    
    scaled_data = scaler.transform(price_series.values.reshape(-1, 1))
    input_seq = scaled_data[-window_size:].reshape(1, window_size, 1)
    
    forecasts = []
    for _ in range(days):
        pred = model.predict(input_seq)[0, 0]
        forecasts.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    return forecasts

def plot_forecast(price_series: pd.Series, forecasted: np.ndarray):
    forecast_index = pd.date_range(start=price_series.index[-1], periods=len(forecasted)+1, freq='D')[1:]
    forecast_series = pd.Series(forecasted, index=forecast_index)
    
    plt.figure(figsize=(12, 5))
    plt.plot(price_series, label='Historical')
    plt.plot(forecast_series, label='Forecast', linestyle='--')
    plt.title('LSTM Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
