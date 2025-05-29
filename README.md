# Stock Analysis Tool

A web-based tool for analyzing stock financial data with an intuitive and visually appealing interface.

## Features

- Historical financial data analysis for the last 10 years
- Key financial metrics including:
  - Market Cap
  - Revenue and Revenue Growth
  - Net Income and Net Income Growth
  - EPS (Earnings Per Share)
  - Shares Outstanding and Growth
  - Free Cash Flow
  - Stock Repurchases
  - Total Debt
  - Debt/Equity Ratio
  - Various Price Ratios (P/S, P/E, P/B, P/CF, P/FCF)
  - ROE (Return on Equity)
- Interactive charts and visualizations
- Clean, modern user interface

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory and add your Financial Modeling Prep API key:
```
FMP_API_KEY=your_api_key_here
```
You can get an API key by signing up at [Financial Modeling Prep](https://site.financialmodelingprep.com/)

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL) in the input field

4. View the financial data in either table format or interactive charts

## Requirements

- Python 3.7+
- Streamlit
- pandas
- plotly
- requests
- python-dotenv
- Financial Modeling Prep API key

## Note

This tool uses the Financial Modeling Prep API to fetch stock data. The data availability and accuracy depend on the FMP database. Make sure you have a valid API key and sufficient API credits for your usage. 