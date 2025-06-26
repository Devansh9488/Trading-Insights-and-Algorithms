# Stock Analysis Dashboard

## Overview

The Stock Analysis Dashboard is an interactive web application that provides comprehensive stock analysis, combining technical indicators, fundamental data, and AI-powered investment insights. Built with Streamlit, it enables users to analyze stocks, visualize price trends, and receive actionable recommendations for both short-term and long-term investment decisions.

## Features

- **Interactive Dashboard:**  
  User-friendly interface to input stock tickers and select date ranges.

- **Technical Analysis:**

  - Candlestick price charts
  - SMA, RSI, MACD, and more
  - Customizable indicator parameters

- **Fundamental Analysis:**

  - Key financial metrics (PE, profit margins, ROE, etc.)
  - Company info and analyst opinions

- **AI-Powered Insights:**

  - Uses Google Gemini API for natural language investment analysis
  - Executive summaries, risk assessments, and tailored recommendations

- **Investment Verdicts:**
  - Short-term and long-term buy/hold/sell signals
  - Rationale for each verdict

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd Stock\ analysis
   ```

2. **Set up a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the project root.
   - Add your Google Gemini API key:
     ```
     GEMINI_API_KEY=your_google_gemini_api_key
     ```

## Usage

1. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**  
   Go to the local URL provided by Streamlit (usually http://localhost:8501).

3. **Enter a stock ticker (e.g., AAPL) and select a date range.**  
   Explore technical charts, fundamental data, and AI-generated insights.

## File Structure

```
Stock analysis/
│
├── app.py              # Streamlit dashboard (main UI)
├── main.py             # Core logic for data collection and AI analysis
├── stock_agent.py      # Technical and fundamental analysis agents
├── finance/            # (Currently empty or not used)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Dependencies

All dependencies are listed in `requirements.txt`. Main libraries include:

- streamlit
- yfinance
- plotly
- pandas
- numpy
- finta
- python-dotenv
- langchain-google-genai
- langchain-core

Install all dependencies with `pip install -r requirements.txt`.

## Customization

- **Add more indicators:**  
  Extend `stock_agent.py` to include additional technical or fundamental metrics.
- **Change AI model:**  
  Update the model or prompt templates in `main.py` for different LLMs or analysis styles.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for new features, bug fixes, or improvements.

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]
