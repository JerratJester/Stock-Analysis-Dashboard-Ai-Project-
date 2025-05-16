import matplotlib.pyplot as plt

def plot_price_and_moving_averages(df, title="Price + SMA/EMA"):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Close"], label="Close Price", color="black")
    if "SMA_20" in df:
        plt.plot(df["SMA_20"], label="SMA 20", linestyle="--")
    if "SMA_50" in df:
        plt.plot(df["SMA_50"], label="SMA 50", linestyle="--")
    if "EMA_20" in df:
        plt.plot(df["EMA_20"], label="EMA 20", linestyle=":")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_rsi(df):
    if "RSI" not in df:
        print("[INFO] RSI not found in DataFrame.")
        return
    plt.figure(figsize=(12, 3))
    plt.plot(df["RSI"], label="RSI", color="purple")
    plt.axhline(70, color="red", linestyle="--")
    plt.axhline(30, color="green", linestyle="--")
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_macd(df):
    if "MACD" not in df:
        print("[INFO] MACD not found in DataFrame.")
        return
    plt.figure(figsize=(12, 3))
    plt.plot(df["MACD"], label="MACD", color="blue")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("MACD")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
