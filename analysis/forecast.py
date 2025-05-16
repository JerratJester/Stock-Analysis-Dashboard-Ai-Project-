import pandas as pd
import matplotlib.pyplot as plt
from models.lstm_model import forecast_next_days
from models.prophet_model import forecast_prophet

# Load historical price data
price_df = pd.read_csv("data/price_data.csv", parse_dates=["Date"], index_col="Date")
price_series = price_df["Close"]

# Forecast parameters
MODEL_PATH_LSTM = "models/lstm_model.h5"
MODEL_PATH_PROPHET = "models/prophet_model.pkl"
FORECAST_DAYS = 30

# Generate LSTM forecast
lstm_forecast = forecast_next_days(
    price_series=price_series,
    model_path=MODEL_PATH_LSTM,
    days=FORECAST_DAYS
)

# Generate Prophet forecast
prophet_df = forecast_prophet(
    price_series=price_series,
    days=FORECAST_DAYS,
    model_path=MODEL_PATH_PROPHET
)
prophet_forecast = prophet_df["yhat"].values
prophet_dates = prophet_df["ds"]

# Align dates for LSTM forecast
forecast_start_date = price_series.index[-1] + pd.Timedelta(days=1)
lstm_dates = pd.date_range(start=forecast_start_date, periods=FORECAST_DAYS, freq="D")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(price_series, label="Historical", color="black")
plt.plot(prophet_dates, prophet_forecast, label="Prophet Forecast", linestyle="--", color="blue")
plt.plot(lstm_dates, lstm_forecast, label="LSTM Forecast", linestyle="--", color="green")
plt.title("LSTM vs. Prophet Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
