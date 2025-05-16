import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import joblib

def prepare_prophet_data(price_series: pd.Series) -> pd.DataFrame:
    df = price_series.reset_index()
    df.columns = ['ds', 'y']
    return df

def train_prophet_model(price_series: pd.Series, model_path: str = "models/prophet_model.pkl") -> Prophet:
    df = prepare_prophet_data(price_series)
    model = Prophet()
    model.fit(df)
    joblib.dump(model, model_path)
    print(f"Prophet model saved to {model_path}")
    return model

def forecast_prophet(price_series: pd.Series, days: int = 30, model_path: str = "models/prophet_model.pkl") -> pd.DataFrame:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = train_prophet_model(price_series, model_path)

    last_date = price_series.index[-1]
    df = prepare_prophet_data(price_series)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast.tail(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def plot_prophet_forecast(price_series: pd.Series, forecast_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(price_series.index, price_series.values, label='Historical')
    ax.plot(forecast_df['ds'], forecast_df['yhat'], linestyle='--', label='Forecast')
    ax.fill_between(
        forecast_df['ds'],
        forecast_df['yhat_lower'],
        forecast_df['yhat_upper'],
        alpha=0.2,
        label='Uncertainty'
    )
    ax.set_title('Prophet Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig