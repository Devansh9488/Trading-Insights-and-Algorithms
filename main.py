import json
import os
import yfinance as yf
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Import your existing stock agent
from stock_agent import MasterAgent


@dataclass
class StockData:
    ticker: str
    analysis_result: Dict[str, Any]
    company_info: Dict[str, Any]
    error: Optional[str] = None


class StockDataCollector:
    
    @staticmethod
    def get_stock_analysis(ticker: str) -> Dict[str, Any]:
        try:
            master_agent = MasterAgent(ticker)
            return master_agent.get_final_verdict()
        except Exception as e:
            return {
                "ticker": ticker,
                "error": f"Analysis failed: {str(e)}",
                "verdict": {"short_term": "Hold", "long_term": "Hold"},
                "rationale": {"technical": [], "fundamental": []}
            }
    
    @staticmethod
    def get_company_info(ticker: str) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information for better LLM understanding
            key_info = {
                "basicInfo": {
                    "companyName": info.get("longName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "country": info.get("country", "N/A"),
                    "currency": info.get("currency", "USD"),
                    "exchange": info.get("exchange", "N/A"),
                    "marketCap": info.get("marketCap"),
                    "employees": info.get("fullTimeEmployees")
                },
                "tradingInfo": {
                    "currentPrice": info.get("currentPrice"),
                    "previousClose": info.get("previousClose"),
                    "dayLow": info.get("dayLow"),
                    "dayHigh": info.get("dayHigh"),
                    "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                    "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                    "volume": info.get("volume"),
                    "averageVolume": info.get("averageVolume")
                },
                "financialMetrics": {
                    "trailingPE": info.get("trailingPE"),
                    "forwardPE": info.get("forwardPE"),
                    "priceToBook": info.get("priceToBook"),
                    "enterpriseValue": info.get("enterpriseValue"),
                    "profitMargins": info.get("profitMargins"),
                    "returnOnEquity": info.get("returnOnEquity"),
                    "returnOnAssets": info.get("returnOnAssets"),
                    "currentRatio": info.get("currentRatio"),
                    "debtToEquity": info.get("debtToEquity"),
                    "totalRevenue": info.get("totalRevenue"),
                    "revenueGrowth": info.get("revenueGrowth"),
                    "earningsGrowth": info.get("earningsQuarterlyGrowth"),
                    "freeCashflow": info.get("freeCashflow")
                },
                "dividendInfo": {
                    "dividendRate": info.get("dividendRate"),
                    "dividendYield": info.get("dividendYield"),
                    "payoutRatio": info.get("payoutRatio"),
                    "exDividendDate": info.get("exDividendDate")
                },
                "analystInfo": {
                    "targetHighPrice": info.get("targetHighPrice"),
                    "targetLowPrice": info.get("targetLowPrice"),
                    "targetMeanPrice": info.get("targetMeanPrice"),
                    "recommendationMean": info.get("recommendationMean"),
                    "recommendationKey": info.get("recommendationKey"),
                    "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions")
                },
                "businessSummary": info.get("longBusinessSummary", "")[:500] + "..." if info.get("longBusinessSummary") else "N/A"
            }
            
            return key_info
            
        except Exception as e:
            return {"error": f"Failed to fetch company info: {str(e)}"}
    
    @classmethod
    def collect_all_data(cls, ticker: str) -> StockData:
        analysis_result = cls.get_stock_analysis(ticker)
        company_info = cls.get_company_info(ticker)
        
        return StockData(
            ticker=ticker,
            analysis_result=analysis_result,
            company_info=company_info
        )


class PromptTemplates:
    
    COMPREHENSIVE_ANALYSIS = ChatPromptTemplate.from_messages([
        ("system", """You are a senior financial advisor with expertise in both technical and fundamental analysis. 

Your role is to provide comprehensive investment recommendations by analyzing:
1. Technical analysis results (RSI, MACD, moving averages, etc.)
2. Fundamental analysis results (PE ratios, profit margins, ROE, etc.) 
3. Company information (sector, industry, market cap, analyst opinions, etc.)

Guidelines for your analysis:
- Provide clear, actionable investment advice
- Explain technical and fundamental reasoning
- Consider company context (sector, size, business model)
- Include both short-term and long-term perspectives
- Mention key risks and opportunities
- Reference analyst opinions when available
- Use professional yet accessible language
- Be objective and balanced in assessment

Structure your response with clear sections and avoid overly technical jargon."""),
        
        ("human", """Please provide a comprehensive investment analysis for {ticker}:

TECHNICAL & FUNDAMENTAL ANALYSIS:
{analysis_json}

COMPANY INFORMATION:
{company_info_json}

Please provide:
1. Executive Summary & Overall Recommendation
2. Technical Analysis Insights
3. Fundamental Analysis Insights  
4. Company & Market Context
5. Risk Assessment
6. Investment Timeline Recommendations
7. Key Factors to Monitor""")
    ])
    
    SPECIFIC_QUESTION = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable financial advisor. Answer the user's specific question about the stock using the provided analysis and company data. Be direct, informative, and base your answer on the available data."""),
        
        ("human", """Stock Data for {ticker}:

ANALYSIS RESULTS:
{analysis_json}

COMPANY INFORMATION:
{company_info_json}

USER QUESTION: {question}

Please provide a direct, helpful answer based on the data above.""")
    ])
    
    QUICK_SUMMARY = ChatPromptTemplate.from_messages([
        ("system", """Provide a concise investment summary in 2-3 paragraphs. Focus on the most important insights and clear recommendation."""),
        
        ("human", """Summarize the investment case for {ticker}:

Analysis: {analysis_json}
Company Info: {company_info_json}""")
    ])


class FinancialBot:
    
    def __init__(self, gemini_api_key: str, model: str = "gemini-1.5-flash"):
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=gemini_api_key,
            temperature=0.3,
            max_tokens=2000
        )
        
        self.data_collector = StockDataCollector()
        self.prompts = PromptTemplates()
        
        # Create chains for different use cases
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup langchain chains for different operations"""
        self.comprehensive_chain = (
            RunnablePassthrough() 
            | self.prompts.COMPREHENSIVE_ANALYSIS 
            | self.llm 
            | StrOutputParser()
        )
        
        self.qa_chain = (
            self.prompts.SPECIFIC_QUESTION 
            | self.llm 
            | StrOutputParser()
        )
        
        self.summary_chain = (
            self.prompts.QUICK_SUMMARY 
            | self.llm 
            | StrOutputParser()
        )
    
    def get_comprehensive_analysis(self, ticker: str) -> str:
        """
        Get comprehensive investment analysis
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Detailed investment analysis
        """
        try:
            # Collect all data
            stock_data = self.data_collector.collect_all_data(ticker)
            
            # Prepare data for LLM
            analysis_json = json.dumps(stock_data.analysis_result, indent=2)
            company_info_json = json.dumps(stock_data.company_info, indent=2)
            
            # Generate comprehensive analysis
            response = self.comprehensive_chain.invoke({
                "ticker": ticker.upper(),
                "analysis_json": analysis_json,
                "company_info_json": company_info_json
            })
            
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error analyzing {ticker}: {str(e)}"
    
    def answer_question(self, ticker: str, question: str) -> str:
        """
        Answer specific questions about a stock
        
        Args:
            ticker: Stock ticker symbol
            question: User's question
            
        Returns:
            Answer to the user's question
        """
        try:
            stock_data = self.data_collector.collect_all_data(ticker)
            
            analysis_json = json.dumps(stock_data.analysis_result, indent=2)
            company_info_json = json.dumps(stock_data.company_info, indent=2)
            
            response = self.qa_chain.invoke({
                "ticker": ticker.upper(),
                "analysis_json": analysis_json,
                "company_info_json": company_info_json,
                "question": question
            })
            
            return response
            
        except Exception as e:
            return f"Sorry, I couldn't answer your question about {ticker}: {str(e)}"
    
    def get_quick_summary(self, ticker: str) -> str:
        """
        Get a quick investment summary
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Brief investment summary
        """
        try:
            stock_data = self.data_collector.collect_all_data(ticker)
            
            analysis_json = json.dumps(stock_data.analysis_result, indent=2)
            company_info_json = json.dumps(stock_data.company_info, indent=2)
            
            response = self.summary_chain.invoke({
                "ticker": ticker.upper(),
                "analysis_json": analysis_json,
                "company_info_json": company_info_json
            })
            
            return response
            
        except Exception as e:
            return f"Sorry, I couldn't generate a summary for {ticker}: {str(e)}"
    
    def get_raw_data(self, ticker: str) -> StockData:
        """Get raw stock data for custom processing"""
        return self.data_collector.collect_all_data(ticker)


class FinancialBotInterface:
    """User-friendly interface for the financial bot"""
    
    def __init__(self, gemini_api_key: str):
        self.bot = FinancialBot(gemini_api_key)
    
    def analyze(self, ticker: str, analysis_type: str = "comprehensive") -> str:
        """
        Main analysis method
        
        Args:
            ticker: Stock ticker
            analysis_type: 'comprehensive', 'quick', or 'summary'
        """
        ticker = ticker.upper()
        
        if analysis_type == "comprehensive":
            return self.bot.get_comprehensive_analysis(ticker)
        elif analysis_type == "quick" or analysis_type == "summary":
            return self.bot.get_quick_summary(ticker)
        else:
            return self.bot.get_comprehensive_analysis(ticker)
    
    def ask(self, ticker: str, question: str) -> str:
        """Ask a specific question about a stock"""
        return self.bot.answer_question(ticker.upper(), question)
    
    def compare_stocks(self, tickers: list) -> str:
        """Compare multiple stocks (basic implementation)"""
        if len(tickers) > 3:
            return "Please limit comparison to 3 stocks maximum."
        
        comparisons = []
        for ticker in tickers:
            summary = self.bot.get_quick_summary(ticker)
            comparisons.append(f"=== {ticker.upper()} ===\n{summary}\n")
        
        return "\n".join(comparisons)


# Example usage and testing
def main():
    """Example usage of the enhanced financial bot"""
    
    # Setup (you need to set your API key)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        print("Please set your GEMINI_API_KEY environment variable")
        return
    
    # Initialize bot
    bot_interface = FinancialBotInterface(GEMINI_API_KEY)
    
    # Example 1: Comprehensive analysis
    print("=== COMPREHENSIVE ANALYSIS ===")
    ticker = "LMT"
    analysis = bot_interface.analyze(ticker, "comprehensive")
    print(f"\nAnalysis for {ticker}:")
    print(analysis)
    print("\n" + "="*70 + "\n")
    
    # Example 2: Quick summary
    print("=== QUICK SUMMARY ===")
    summary = bot_interface.analyze(ticker, "quick")
    print(f"\nQuick summary for {ticker}:")
    print(summary)
    print("\n" + "="*70 + "\n")
    
    # Example 3: Q&A
    print("=== Q&A EXAMPLES ===")
    questions = [
        "What's the current valuation like?",
        "Should I hold this for retirement?",
        "What are the main risks?",
        "How does this compare to sector peers?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        answer = bot_interface.ask(ticker, question)
        print(f"A: {answer}\n")
    
    # Example 4: Stock comparison
    # print("=== STOCK COMPARISON ===")
    # comparison = bot_interface.compare_stocks(["AAPL", "GOOGL"])
    # print(comparison)


if __name__ == "__main__":
    main()


# Utility functions for integration
class BotUtils:
    """Utility functions for bot integration"""
    
    @staticmethod
    def format_for_web(response: str) -> Dict[str, Any]:
        """Format response for web API"""
        return {
            "status": "success",
            "response": response,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Basic ticker validation"""
        return isinstance(ticker, str) and len(ticker.strip()) > 0
    
    @staticmethod
    def get_supported_analysis_types() -> list:
        """Get list of supported analysis types"""
        return ["comprehensive", "quick", "summary"]


# FastAPI integration example
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Financial Bot API")
bot_interface = FinancialBotInterface(os.getenv("GEMINI_API_KEY"))

class AnalysisRequest(BaseModel):
    ticker: str
    analysis_type: str = "comprehensive"

class QuestionRequest(BaseModel):
    ticker: str
    question: str

@app.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    try:
        if not BotUtils.validate_ticker(request.ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker")
        
        response = bot_interface.analyze(request.ticker, request.analysis_type)
        return BotUtils.format_for_web(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        if not BotUtils.validate_ticker(request.ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker")
        
        response = bot_interface.ask(request.ticker, request.question)
        return BotUtils.format_for_web(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""