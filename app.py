import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from main import FinancialBotInterface
from stock_agent import MasterAgent
from dotenv import load_dotenv
import os

load_dotenv()
# Set page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the Financial Bot
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Please set your GEMINI_API_KEY environment variable")
    st.stop()

bot_interface = FinancialBotInterface(GEMINI_API_KEY)

# Title and description
st.title("📈 Stock Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive stock analysis, including technical indicators, 
fundamental data, and AI-powered insights. Enter a ticker symbol to get started.
""")

with st.sidebar:
    st.header("Stock Selection")
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    
    # Date range selection
    st.subheader("Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = st.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )

# Main content area
if ticker:
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=date_range[0], end=date_range[1])
        
        # Get stock analysis verdict
        master_agent = MasterAgent(ticker)
        analysis_result = master_agent.get_final_verdict()
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📊 Technical Analysis", "📑 Fundamental Analysis", "💬 AI Assistant"])
        
        with tab1:
            st.subheader("Price Chart")
            
            # Technical indicators selection
            col1, col2, col3 = st.columns(3)
            with col1:
                show_sma = st.checkbox("Show SMA", value=True)
                sma_period = st.slider("SMA Period", 5, 200, 20)
            with col2:
                show_rsi = st.checkbox("Show RSI", value=True)
                rsi_period = st.slider("RSI Period", 5, 30, 14)
            with col3:
                show_macd = st.checkbox("Show MACD", value=True)
            
            # Create subplot for price and indicators
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.05,
                              row_heights=[0.6, 0.2, 0.2])
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Add SMA if selected
            if show_sma:
                sma = hist['Close'].rolling(window=sma_period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=sma,
                        name=f"SMA {sma_period}",
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
            
            # Add RSI if selected
            if show_rsi:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                fig.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=rsi,
                        name="RSI",
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                
                # Add RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Add MACD if selected
            if show_macd:
                exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
                exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=macd,
                        name="MACD",
                        line=dict(color='blue')
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=signal,
                        name="Signal",
                        line=dict(color='orange')
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                xaxis_rangeslider_visible=False,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display Investment Verdict
            st.subheader("Investment Verdict")
            
            # Create columns for verdicts
            col1, col2 = st.columns(2)
            
            # Short-term verdict
            with col1:
                st.metric(
                    "Short-term Verdict",
                    analysis_result["verdict"]["short_term"],
                    delta=None,
                    delta_color="normal"
                )
                st.write("Technical Analysis:")
                for point in analysis_result["rationale"]["technical"]:
                    st.write(f"• {point}")
            
            # Long-term verdict
            with col2:
                st.metric(
                    "Long-term Verdict",
                    analysis_result["verdict"]["long_term"],
                    delta=None,
                    delta_color="normal"
                )
                st.write("Fundamental Analysis:")
                for point in analysis_result["rationale"]["fundamental"]:
                    st.write(f"• {point}")
            
            # Add a note about the verdict
            st.info("""
            The verdict is based on a combination of technical and fundamental analysis:
            - Technical Analysis: Price patterns, momentum indicators, and trend analysis
            - Fundamental Analysis: Financial metrics, growth rates, and company health
            """)
        
        with tab2:
            st.subheader("Fundamental Analysis")
            
            # Get company info
            info = stock.info
            
            # Display key metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.0f}")
                st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            
            with col2:
                st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
                st.metric("Volume", f"{info.get('volume', 'N/A'):,.0f}")
            
            with col3:
                st.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A')*100:.2f}%")
                st.metric("Beta", f"{info.get('beta', 'N/A')}")
                st.metric("EPS", f"${info.get('trailingEps', 'N/A')}")
            
            # Company description
            st.subheader("Company Description")
            st.write(info.get('longBusinessSummary', 'N/A'))
            
            # Financial metrics
            st.subheader("Financial Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Revenue Growth', 'Profit Margin', 'ROE', 'Debt to Equity'],
                'Value': [
                    f"{info.get('revenueGrowth', 'N/A')*100:.2f}%",
                    f"{info.get('profitMargins', 'N/A')*100:.2f}%",
                    f"{info.get('returnOnEquity', 'N/A')*100:.2f}%",
                    f"{info.get('debtToEquity', 'N/A'):.2f}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True)
        
        with tab3:
            st.subheader("AI Assistant")
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the stock"):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    response = bot_interface.ask(ticker, prompt)
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")
else:
    st.info("Please enter a ticker symbol in the sidebar to begin analysis.") 