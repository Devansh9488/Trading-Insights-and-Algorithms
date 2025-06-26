import yfinance as yf
import pandas as pd
import numpy as np
from finta import TA
import json
from dotenv import load_dotenv

load_dotenv()

class TechnicalAnalysisAgent:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, period="6mo", interval="1d")
        df.dropna(inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df

    def analyze(self) -> dict:
        df = self.data.copy()
        rationale = []
        short_term, long_term = "Hold", "Hold"

        try:
            # Core indicators
            df['RSI'] = TA.RSI(df)
            macd_data = TA.MACD(df)
            df['MACD'] = macd_data['MACD']
            df['SMA_20'] = TA.SMA(df, 20)
            df['SMA_50'] = TA.SMA(df, 50)

            # Additional indicators
            df['EMA_20'] = TA.EMA(df, 20)
            df['ADX'] = TA.ADX(df)
            df['MOM'] = TA.MOM(df)
            df['CCI'] = TA.CCI(df)
            df['OBV'] = TA.OBV(df)

            latest = df.iloc[-1]

            # Short-term logic
            if latest['RSI'] < 30 and latest['MACD'] > 0 and latest['MOM'] > 0:
                short_term = "Buy"
                rationale.append("RSI < 30, MACD > 0, MOM > 0 → bullish short-term momentum.")
            elif latest['RSI'] > 70 and latest['MACD'] < 0 and latest['MOM'] < 0:
                short_term = "Sell"
                rationale.append("RSI > 70, MACD < 0, MOM < 0 → bearish short-term momentum.")
            elif latest['CCI'] > 100:
                short_term = "Buy"
                rationale.append("CCI > 100 → potential strong upside.")
            elif latest['CCI'] < -100:
                short_term = "Sell"
                rationale.append("CCI < -100 → possible downside risk.")

            if latest['SMA_20'] > latest['SMA_50'] and latest['EMA_20'] > latest['SMA_50']:
                long_term = "Buy"
                rationale.append("SMA 20 & EMA 20 > SMA 50 → long-term uptrend.")
            elif latest['SMA_20'] < latest['SMA_50'] and latest['EMA_20'] < latest['SMA_50']:
                long_term = "Sell"
                rationale.append("SMA 20 & EMA 20 < SMA 50 → long-term downtrend.")
            else:
                rationale.append("Mixed SMA/EMA signals → long-term Hold.")

            # Trend strength (ADX)
            if latest['ADX'] > 25:
                rationale.append("ADX > 25 → strong trend present.")
            else:
                rationale.append("ADX ≤ 25 → weak or sideways trend.")

        except Exception as e:
            return {
                "agent": "technical",
                "verdict": {"short_term": "Hold", "long_term": "Hold"},
                "rationale": [f"Error computing technical indicators: {e}"]
            }

        return {
            "agent": "technical",
            "verdict": {"short_term": short_term, "long_term": long_term},
            "rationale": rationale
        }


class FundamentalAnalysisAgent:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = self._fetch_data()

    def _fetch_data(self) -> dict:
        return yf.Ticker(self.ticker).info

    def _safe_get(self, key: str, default=None):
        value = self.data.get(key, default)
        return value if value not in [None, np.nan] else default

    def analyze(self) -> dict:
        rationale = []
        score = 0

        # Valuation
        pe = self._safe_get("trailingPE")
        if pe is not None:
            if pe < 15:
                score += 1
                rationale.append("PE < 15 → undervalued.")
            elif pe > 30:
                score -= 1
                rationale.append("PE > 30 → potentially overvalued.")
            else:
                rationale.append("PE within average range.")
        else:
            rationale.append("PE data unavailable.")

        # Profitability
        profit_margin = self._safe_get("profitMargins")
        if profit_margin is not None:
            if profit_margin > 0.15:
                score += 1
                rationale.append("Profit margin > 15% → strong profitability.")
            elif profit_margin < 0.05:
                score -= 1
                rationale.append("Profit margin < 5% → weak profitability.")
            else:
                rationale.append("Profit margin is average.")
        else:
            rationale.append("Profit margin data unavailable.")

        # Efficiency
        roe = self._safe_get("returnOnEquity")
        if roe is not None:
            if roe > 0.15:
                score += 1
                rationale.append("ROE > 15% → efficient capital use.")
            elif roe < 0.07:
                score -= 1
                rationale.append("ROE < 7% → inefficient use of equity.")
            else:
                rationale.append("ROE is moderate.")
        else:
            rationale.append("ROE data unavailable.")

        # Liquidity
        current_ratio = self._safe_get("currentRatio")
        if current_ratio is not None:
            if current_ratio >= 1.5:
                score += 1
                rationale.append("Current ratio ≥ 1.5 → healthy liquidity.")
            elif current_ratio < 1.0:
                score -= 1
                rationale.append("Current ratio < 1.0 → liquidity risk.")
            else:
                rationale.append("Current ratio is acceptable.")
        else:
            rationale.append("Current ratio data unavailable.")

        # Leverage
        debt_to_equity = self._safe_get("debtToEquity")
        if debt_to_equity is not None:
            if debt_to_equity < 1.0:
                score += 1
                rationale.append("Debt/Equity < 1.0 → manageable debt.")
            elif debt_to_equity > 2.0:
                score -= 1
                rationale.append("Debt/Equity > 2.0 → financial risk.")
            else:
                rationale.append("Debt/Equity is reasonable.")
        else:
            rationale.append("Debt/Equity data unavailable.")

        # Cash Flow
        free_cash_flow = self._safe_get("freeCashflow")
        total_revenue = self._safe_get("totalRevenue")
        if free_cash_flow and total_revenue:
            fcf_margin = free_cash_flow / total_revenue
            if fcf_margin > 0.1:
                score += 1
                rationale.append("Free cash flow > 10% of revenue → strong cash generation.")
            elif fcf_margin < 0.02:
                score -= 1
                rationale.append("Low free cash flow relative to revenue.")
            else:
                rationale.append("Moderate free cash flow margin.")
        else:
            rationale.append("Cash flow data unavailable.")

        # Growth
        earnings_growth = self._safe_get("earningsQuarterlyGrowth")
        if earnings_growth is not None:
            if earnings_growth > 0.2:
                score += 1
                rationale.append("Earnings growth > 20% YoY → strong growth.")
            elif earnings_growth < 0:
                score -= 1
                rationale.append("Earnings shrinking → negative outlook.")
            else:
                rationale.append("Moderate earnings growth.")
        else:
            rationale.append("Earnings growth data unavailable.")

        long_term = "Buy" if score >= 4 else "Sell" if score <= -2 else "Hold"

        if profit_margin is not None and profit_margin > 0.15 and earnings_growth and earnings_growth > 0.15:
            short_term = "Buy"
        elif profit_margin is not None and profit_margin < 0.05 and earnings_growth and earnings_growth < 0:
            short_term = "Sell"
        else:
            short_term = "Hold"

        return {
            "agent": "fundamental",
            "verdict": {
                "short_term": short_term,
                "long_term": long_term
            },
            "rationale": rationale
        }


class MasterAgent:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.ta_agent = TechnicalAnalysisAgent(ticker)
        self.fa_agent = FundamentalAnalysisAgent(ticker)

    def _majority_vote(self, ta_vote: str, fa_vote: str) -> str:
        if ta_vote == fa_vote:
            return ta_vote
        if "Hold" in [ta_vote, fa_vote]:
            return ta_vote if fa_vote == "Hold" else fa_vote
        return "Hold"

    def get_final_verdict(self) -> dict:
        ta_result = self.ta_agent.analyze()
        fa_result = self.fa_agent.analyze()

        combined = {
            "short_term": self._majority_vote(ta_result["verdict"]["short_term"], fa_result["verdict"]["short_term"]),
            "long_term": self._majority_vote(ta_result["verdict"]["long_term"], fa_result["verdict"]["long_term"])
        }

        return {
            "ticker": self.ticker,
            "verdict": combined,
            "rationale": {
                "technical": ta_result["rationale"],
                "fundamental": fa_result["rationale"]
            }
        }


if __name__ == "__main__":
    ticker = "AAPL"
    master_agent = MasterAgent(ticker)
    verdict = master_agent.get_final_verdict()
    print(json.dumps(verdict, indent=4))