# AI-Powered Stock & Crypto Dashboard 📊

This project is a full-featured dashboard for analyzing financial markets using artificial intelligence. It includes separate pages for **stocks** and **cryptocurrencies**, and features:

## 🔍 Features

- 📈 **Time Series Forecasting** (stock & crypto prices)
- 🧠 **AI-based Sentiment Analysis** from news and social media
- 📊 **Technical Indicators** (e.g., RSI, MACD, Moving Averages)
- 💹 Real-time market data integration
- 🧭 Easy navigation between Stocks and Crypto dashboards



## 🧪 Technologies Used

**Languages:**
- Python 3.10
- Markdown (for documentation)

**Frameworks & Libraries:**
- **Streamlit** – interactive web dashboard  
- **pandas, numpy** – data wrangling and analysis  
- **matplotlib, plotly** – data visualization  
- **ta** – technical analysis indicators  
- **yfinance** – historical market data  
- **Prophet** – time series forecasting  
- **TensorFlow / Keras** – LSTM forecasting models  
- **vaderSentiment** – sentiment analysis  
- **requests, dotenv** – API and config management  

---
## ⚠️ Known Issue
When selecting a ticker **for the first time**, you may see:
ValueError: The truth value of a Series is ambiguous.


**Fix:**  
Simply reload the Streamlit app:

```bash
Ctrl + C
streamlit run app.py

## 🛠️ Setup Instructions (Conda)
## 🛠️ Setup Instructions (Conda)

1. **Clone the repository**
   ```bash
   git clone https://github.com/jerratjester/StockAnalysisDashboardAiProject.git
   cd StockAnalysisDashboardAiProject
2. **Create the Conda repository**
    conda create -n stockdash python=3.10
    conda activate stockdash
3. **Install dependencies**
    pip install -r requirements.txt
4. **Set up your environment variables**

   - Create a `.env` file inside the `data` directory (i.e. `data/.env`).
   - Add your API keys and settings (see template below).

   ```env
   # Stock Data APIs
   ALPHA_VANTAGE_API_KEY=
   FINNHUB_API_KEY=
   POLYGON_API_KEY=

   # News & Sentiment APIs
   NEWS_API_KEY=
   GNEWS_API_KEY=

   # Social Media (optional)
   TWITTER_BEARER_TOKEN=

   # Custom Configs
   MODEL_CACHE_PATH=models/
5. **Run the App**
    streamlit run app.py

## 📌 Roadmap

- [x] Dashboard layout (Stocks / Crypto)
- [x] Financial data integration
- [x] LSTM & Prophet forecasting models
- [x] Technical indicators
- [x] Sentiment analysis engine
- [x] Buy / Hold / Sell recommendations
- [ ] UI/UX polish

---

## 🤖 Example Use Cases

- **Investors** monitoring trends  
- **Traders** identifying signals  
- **Analysts** evaluating sentiment  

---

## 📬 Contact

Built with ❤️ by **Jerrat Jester**

