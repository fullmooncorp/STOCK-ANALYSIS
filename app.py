import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Stock Analysis Tool")
st.markdown("Enter a stock ticker symbol to analyze its historical financial data.")

# Search box
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL):", "").upper()

# Add period selection
period = st.radio("Select Period:", ["Annual", "Quarterly"], horizontal=True)

# FMP API configuration
FMP_API_KEY = os.getenv('FMP_API_KEY')
if not FMP_API_KEY:
    st.error("Please set your FMP_API_KEY in the .env file")
    st.stop()

BASE_URL = "https://financialmodelingprep.com/api/v3"

def format_number(value):
    """Format number to K, M, B format"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    value = float(value)
    if abs(value) >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.1f}"

def format_percentage(value):
    """Format percentage values"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.1f}%"

def format_ratio(value):
    """Format ratio values"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.2f}"

def safe_get(data, key, default=None):
    """Safely get a value from a dictionary with a default if not found"""
    return data.get(key, default)

def fetch_company_profile(symbol):
    url = f"{BASE_URL}/profile/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching profile: {response.status_code} - {response.text}")
        return None
    data = response.json()
    if not data:
        st.error(f"No profile data found for {symbol}")
        return None
    return data[0]

def fetch_financial_statements(symbol, period_type="annual"):
    limit = 40 if period_type == "quarterly" else 10
    
    # Fetch income statement
    income_url = f"{BASE_URL}/income-statement/{symbol}?limit={limit}&period={period_type}&apikey={FMP_API_KEY}"
    income_response = requests.get(income_url)
    if income_response.status_code != 200:
        st.error(f"Error fetching income statement: {income_response.status_code} - {income_response.text}")
        return [], [], []
    income_data = income_response.json()
    if not income_data:
        st.error(f"No income statement data found for {symbol}")
        return [], [], []

    # Fetch balance sheet
    balance_url = f"{BASE_URL}/balance-sheet-statement/{symbol}?limit={limit}&period={period_type}&apikey={FMP_API_KEY}"
    balance_response = requests.get(balance_url)
    if balance_response.status_code != 200:
        st.error(f"Error fetching balance sheet: {balance_response.status_code} - {balance_response.text}")
        return [], [], []
    balance_data = balance_response.json()
    if not balance_data:
        st.error(f"No balance sheet data found for {symbol}")
        return [], [], []

    # Fetch cash flow
    cash_flow_url = f"{BASE_URL}/cash-flow-statement/{symbol}?limit={limit}&period={period_type}&apikey={FMP_API_KEY}"
    cash_flow_response = requests.get(cash_flow_url)
    if cash_flow_response.status_code != 200:
        st.error(f"Error fetching cash flow: {cash_flow_response.status_code} - {cash_flow_response.text}")
        return [], [], []
    cash_flow_data = cash_flow_response.json()
    if not cash_flow_data:
        st.error(f"No cash flow data found for {symbol}")
        return [], [], []

    return income_data, balance_data, cash_flow_data

def fetch_ratios(symbol, period_type="annual"):
    limit = 40 if period_type == "quarterly" else 10
    url = f"{BASE_URL}/ratios/{symbol}?limit={limit}&period={period_type}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching ratios: {response.status_code} - {response.text}")
        return []
    data = response.json()
    if not data:
        st.error(f"No ratio data found for {symbol}")
        return []
    return data

def fetch_dividend_data(symbol):
    """Fetch dividend data for the given symbol"""
    url = f"{BASE_URL}/historical-price-full/stock_dividend/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching dividend data: {response.status_code} - {response.text}")
        return None
    data = response.json()
    if not data or 'historical' not in data:
        st.warning(f"No dividend data found for {symbol}")
        return None
    return data['historical']

def fetch_analyst_estimates(symbol):
    """Fetch analyst estimates for the given symbol"""
    url = f"{BASE_URL}/analyst-estimates/{symbol}?limit=10&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching analyst estimates: {response.status_code} - {response.text}")
        return None
    data = response.json()
    if not data:
        st.warning(f"Could not fetch analyst estimates for {symbol}")
        return None
    return data

def fetch_insider_trading(symbol):
    """Fetch insider trading data for the given symbol"""
    url = f"{BASE_URL}/insider-trading/{symbol}?limit=50&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching insider trading data: {response.status_code} - {response.text}")
        return None
    data = response.json()
    if not data:
        st.warning(f"No insider trading data found for {symbol}")
        return None
    return data

def fetch_institutional_ownership(symbol):
    """Fetch institutional ownership data for the given symbol"""
    url = f"{BASE_URL}/institutional-holder/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching institutional ownership data: {response.status_code} - {response.text}")
        return None
    data = response.json()
    if not data:
        st.warning(f"No institutional ownership data found for {symbol}")
        return None
    return data

if ticker:
    try:
        # Fetch all required data
        profile = fetch_company_profile(ticker)
        if not profile:
            st.error(f"Could not fetch company profile for {ticker}")
            st.stop()

        # Add Company Overview Section
        st.subheader("Company Overview")
        
        # Add Company Name
        company_name = profile.get('companyName', ticker)
        st.markdown(f"## {company_name} ({ticker})")
        
        # Calculate current dividend yield
        current_price = profile.get('price', 0)
        dividend_data = fetch_dividend_data(ticker)
        current_dividend_yield = "N/A"
        
        if dividend_data and current_price > 0:
            # Get the most recent dividend
            latest_dividend = dividend_data[0].get('dividend', 0)
            # Calculate annual yield (assuming quarterly dividends)
            annual_dividend = latest_dividend * 4
            current_dividend_yield = f"{(annual_dividend / current_price * 100):.2f}%"
        
        # Create two columns for the overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Company Description
            if 'description' in profile:
                st.markdown(f"**Description:** {profile['description']}")
            
            # Key Information
            st.markdown("**Key Information:**")
            key_info = {
                'Industry': profile.get('industry', 'N/A'),
                'Sector': profile.get('sector', 'N/A'),
                'CEO': profile.get('ceo', 'N/A'),
                'Website': profile.get('website', 'N/A'),
                'Exchange': profile.get('exchange', 'N/A'),
                'Country': profile.get('country', 'N/A')
            }
            
            for key, value in key_info.items():
                if value != 'N/A':
                    st.markdown(f"- **{key}:** {value}")
        
        with col2:
            # Market Data
            st.markdown("**Market Data:**")
            market_data = {
                'Market Cap': format_number(profile.get('mktCap')),
                '52 Week High': f"${profile.get('range', '0-0').split('-')[1] if profile.get('range') else '0'}",
                '52 Week Low': f"${profile.get('range', '0-0').split('-')[0] if profile.get('range') else '0'}",
                'Beta': f"{profile.get('beta', 0):.2f}",
                'Dividend Yield': current_dividend_yield,
                'P/E Ratio': f"{profile.get('pe', 0):.2f}"
            }
            
            for key, value in market_data.items():
                if value != 'N/A':
                    st.markdown(f"- **{key}:** {value}")
        
        st.markdown("---")  # Add a separator line

        period_type = "quarterly" if period == "Quarterly" else "annual"
        income_data, balance_data, cash_flow_data = fetch_financial_statements(ticker, period_type)
        if not income_data or not balance_data or not cash_flow_data:
            st.error(f"Could not fetch financial statements for {ticker}")
            st.stop()

        ratios_data = fetch_ratios(ticker, period_type)
        if not ratios_data:
            st.warning(f"Could not fetch ratios for {ticker}. Some metrics may be missing.")

        # Fetch analyst estimates
        analyst_estimates = fetch_analyst_estimates(ticker)
        if not analyst_estimates:
            st.warning(f"Could not fetch analyst estimates for {ticker}")

        # Fetch dividend data
        dividend_data = fetch_dividend_data(ticker)
        if not dividend_data:
            st.warning(f"Could not fetch dividend data for {ticker}")

        # Create a DataFrame for our analysis
        data = []
        
        for i in range(len(income_data)):
            try:
                # Get period label
                if period_type == "quarterly":
                    period_label = f"{income_data[i]['calendarYear']} Q{income_data[i]['period']}"
                else:
                    period_label = income_data[i]['calendarYear']
                
                year_data = {
                    'Period': period_label,
                    'Market Cap': safe_get(profile, 'mktCap'),
                    'Revenue': safe_get(income_data[i], 'revenue'),
                    'Net Income': safe_get(income_data[i], 'netIncome'),
                    'EPS': safe_get(income_data[i], 'eps'),
                    'Basic Avg Shares': safe_get(income_data[i], 'weightedAverageShsOut'),
                    'FCF': safe_get(cash_flow_data[i], 'freeCashFlow'),
                    'Total Debt': safe_get(balance_data[i], 'totalDebt'),
                    'Stock Repurch': safe_get(cash_flow_data[i], 'commonStockRepurchased'),
                }
                
                # Calculate growth rates
                if i < len(income_data) - 1:
                    prev_rev = safe_get(income_data[i + 1], 'revenue', 0)
                    curr_rev = safe_get(income_data[i], 'revenue', 0)
                    year_data['Rev Growth'] = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev != 0 else None
                    
                    prev_ni = safe_get(income_data[i + 1], 'netIncome', 0)
                    curr_ni = safe_get(income_data[i], 'netIncome', 0)
                    year_data['NI Growth'] = ((curr_ni - prev_ni) / prev_ni * 100) if prev_ni != 0 else None
                    
                    prev_shares = safe_get(income_data[i + 1], 'weightedAverageShsOut', 0)
                    curr_shares = safe_get(income_data[i], 'weightedAverageShsOut', 0)
                    year_data['Shares Growth'] = ((curr_shares - prev_shares) / prev_shares * 100) if prev_shares != 0 else None
                
                # Add ratios
                if i < len(ratios_data):
                    year_data['P/S'] = safe_get(ratios_data[i], 'priceToSalesRatio')
                    year_data['P/E'] = safe_get(ratios_data[i], 'priceEarningsRatio')
                    year_data['P/B'] = safe_get(ratios_data[i], 'priceToBookRatio')
                    year_data['P/CF'] = safe_get(ratios_data[i], 'priceCashFlowRatio')
                    year_data['P/FCF'] = safe_get(ratios_data[i], 'priceToFreeCashFlowsRatio')
                    roe = safe_get(ratios_data[i], 'returnOnEquity')
                    year_data['ROE'] = roe * 100 if roe is not None else None
                    year_data['Debt/Equity'] = safe_get(ratios_data[i], 'debtEquityRatio')
                
                data.append(year_data)
            except Exception as e:
                st.error(f"Error processing data for period {income_data[i]['calendarYear']}: {str(e)}")
                continue
        
        if not data:
            st.error("No data could be processed. Please check the API response.")
            st.stop()

        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Sort DataFrame by Period in ascending order
        df = df.sort_values('Period')
        
        # Reorder columns in a logical grouping
        column_order = [
            # Basic Info
            'Period',
            'Market Cap',
            'Basic Avg Shares',
            
            # Revenue & Growth
            'Revenue',
            'Rev Growth',
            
            # Profitability
            'Net Income',
            'NI Growth',
            'EPS',
            'ROE',
            
            # Cash Flow
            'FCF',
            
            # Valuation Ratios
            'P/E',
            'P/S',
            'P/B',
            'P/CF',
            'P/FCF',
            
            # Financial Health
            'Total Debt',
            'Debt/Equity',
            'Stock Repurch'
        ]
        
        # Reorder columns
        df = df[column_order]
        
        # Display the data
        st.subheader(f"Financial Analysis for {ticker} ({period})")
        
        # Display the formatted table
        display_df = df.copy()
        
        # Define formatting rules for different columns
        monetary_columns = ['Market Cap', 'Revenue', 'Net Income', 'FCF', 'Total Debt', 'Stock Repurch']
        percentage_columns = ['Rev Growth', 'NI Growth', 'Shares Growth']
        ratio_columns = ['P/S', 'P/E', 'P/B', 'P/CF', 'P/FCF', 'Debt/Equity']
        
        # Apply formatting
        for col in display_df.columns:
            if col == 'Period':
                continue
            elif col == 'Basic Avg Shares':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            elif col in monetary_columns:
                display_df[col] = display_df[col].apply(format_number)
            elif col in percentage_columns:
                display_df[col] = display_df[col].apply(format_percentage)
            elif col == 'ROE':
                display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
            elif col in ratio_columns:
                display_df[col] = display_df[col].apply(format_ratio)
        
        # Configure column tooltips
        column_config = {
            'P/E': st.column_config.NumberColumn(
                'P/E',
                help="The P/E ratio, or Price-to-Earnings Ratio, is a valuation metric that compares a company's stock price to its earnings per share (EPS). It essentially tells you how much investors are willing to pay for each dollar of a company's earnings. A higher P/E ratio might indicate that a stock is overvalued, while a lower P/E ratio could suggest it's undervalued"
            ),
            'P/S': st.column_config.NumberColumn(
                'P/S',
                help="The P/S ratio (Price-to-Sales ratio) is a valuation metric used to assess a company's stock price in relation to its total revenue. It helps investors gauge how much they're paying for each dollar of sales, often used as an alternative to the P/E ratio when earnings are negative or volatile"
            ),
            'P/B': st.column_config.NumberColumn(
                'P/B',
                help="The P/B ratio (Price-to-Book ratio) is a financial metric used to evaluate a company's stock price in relation to its book value per share. It's calculated by dividing the market price per share by the book value per share. A lower P/B ratio may suggest a company is undervalued, while a higher ratio could indicate that investors are paying a premium for the company's assets"
            ),
            'P/CF': st.column_config.NumberColumn(
                'P/CF',
                help="The Price-to-Cash Flow (P/CF) ratio is a financial metric that compares a company's market value (share price) to its operating cash flow per share. It helps investors assess whether a stock is undervalued or overvalued by considering the cash a company actually generates from its operations"
            ),
            'P/FCF': st.column_config.NumberColumn(
                'P/FCF',
                help="The Price to Free Cash Flow (P/FCF) ratio is a valuation metric that compares a company's market capitalization to its free cash flow. It essentially tells investors how much they're willing to pay for each dollar of free cash flow the company generates"
            ),
            'Debt/Equity': st.column_config.NumberColumn(
                'Debt/Equity',
                help="Debt to Equity Ratio - Measures financial leverage. Ideal: 0.5-2.0"
            ),
            'ROE': st.column_config.NumberColumn(
                'ROE',
                help="Return on Equity - Measures profitability relative to shareholder equity. IdeaL Higher the better"
            )
        }
        
        st.dataframe(display_df, use_container_width=True, column_config=column_config)
        
        # Update chart titles to reflect period type
        period_suffix = " (Quarterly)" if period == "Quarterly" else " (Annual)"
        
        # EPS Over Time Chart
        fig_eps = go.Figure()
        fig_eps.add_trace(go.Bar(
            x=df['Period'],
            y=df['EPS'],
            name='EPS',
            text=df['EPS'].apply(lambda x: f"{x:.2f}"),
            textposition='outside',
            marker_color='#00CC96'
        ))
        
        fig_eps.update_layout(
            title=f"EPS Over Time{period_suffix}",
            xaxis_title="Period",
            yaxis_title="EPS",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_eps, use_container_width=True)
        
        # Revenue Over Time Chart
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Bar(
            x=df['Period'],
            y=df['Revenue'],
            name='Revenue',
            text=df['Revenue'].apply(lambda x: format_number(x)),
            textposition='outside'
        ))
        
        fig_revenue.update_layout(
            title=f"Revenue Over Time{period_suffix}",
            xaxis_title="Period",
            yaxis_title="Revenue",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Quarterly Sales Pattern Chart (only for quarterly view)
        if period == "Quarterly":
            # Create a pivot table to organize data by year and quarter
            df['Year'] = df['Period'].str[:4]
            df['Quarter'] = df['Period'].str[-2:]
            pivot_df = df.pivot(index='Year', columns='Quarter', values='Revenue')
            
            fig_quarterly = go.Figure()
            
            # Add a bar for each quarter
            for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                if quarter in pivot_df.columns:
                    fig_quarterly.add_trace(go.Bar(
                        name=quarter,
                        x=pivot_df.index,
                        y=pivot_df[quarter],
                        text=pivot_df[quarter].apply(lambda x: format_number(x)),
                        textposition='outside'
                    ))
            
            fig_quarterly.update_layout(
                title=f"Quarterly Sales Pattern{period_suffix}",
                xaxis_title="Year",
                yaxis_title="Revenue",
                template="plotly_white",
                height=500,
                barmode='group',
                showlegend=True
            )
            
            st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Revenue Growth Rate Chart
        fig_rev_growth = go.Figure()
        fig_rev_growth.add_trace(go.Scatter(
            x=df['Period'],
            y=df['Rev Growth'],
            name='Revenue Growth Rate',
            mode='lines+markers+text',
            text=df['Rev Growth'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"),
            textposition='top center'
        ))
        
        fig_rev_growth.update_layout(
            title=f"Revenue Growth Rate Over Time{period_suffix}",
            xaxis_title="Period",
            yaxis_title="Growth Rate (%)",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_rev_growth, use_container_width=True)
        
        # Net Income Growth Rate Chart
        fig_ni_growth = go.Figure()
        fig_ni_growth.add_trace(go.Scatter(
            x=df['Period'],
            y=df['NI Growth'],
            name='Net Income Growth Rate',
            mode='lines+markers+text',
            text=df['NI Growth'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"),
            textposition='top center'
        ))
        
        fig_ni_growth.update_layout(
            title=f"Net Income Growth Rate Over Time{period_suffix}",
            xaxis_title="Period",
            yaxis_title="Growth Rate (%)",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_ni_growth, use_container_width=True)
        
        # Shares Outstanding Chart
        fig_shares = go.Figure()
        fig_shares.add_trace(go.Bar(
            x=df['Period'],
            y=df['Basic Avg Shares'],
            name='Basic Average Shares',
            text=df['Basic Avg Shares'].apply(lambda x: format_number(x)),
            textposition='outside',
            marker_color='#9B59B6'
        ))
        
        fig_shares.update_layout(
            title=f"Basic Average Shares Over Time{period_suffix}",
            xaxis_title="Period",
            yaxis_title="Basic Average Shares",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_shares, use_container_width=True)
        
        # Stock Repurchases Chart
        fig_repurch = go.Figure()
        fig_repurch.add_trace(go.Bar(
            x=df['Period'],
            y=df['Stock Repurch'],
            name='Stock Repurchases',
            text=df['Stock Repurch'].apply(lambda x: format_number(x)),
            textposition='outside'
        ))
        
        fig_repurch.update_layout(
            title=f"Stock Repurchases Over Time{period_suffix}",
            xaxis_title="Period",
            yaxis_title="Stock Repurchases",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_repurch, use_container_width=True)
        
        # Total Debt Chart
        fig_debt = go.Figure()
        fig_debt.add_trace(go.Bar(
            x=df['Period'],
            y=df['Total Debt'],
            name='Total Debt',
            text=df['Total Debt'].apply(lambda x: format_number(x)),
            textposition='outside'
        ))
        
        fig_debt.update_layout(
            title=f"Total Debt Over Time{period_suffix}",
            xaxis_title="Period",
            yaxis_title="Total Debt",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_debt, use_container_width=True)
        
        # After the Total Debt chart and before the Dividend History section, add Cash Flow Analysis
        st.subheader("Cash Flow Analysis")
        
        # Create a DataFrame for cash flow data
        cash_flow_df = pd.DataFrame(cash_flow_data)
        cash_flow_df['date'] = pd.to_datetime(cash_flow_df['date'])
        cash_flow_df = cash_flow_df.sort_values('date', ascending=False)
        
        # Select key cash flow metrics
        cash_flow_metrics = {
            'Operating Cash Flow': 'operatingCashFlow',
            'Free Cash Flow': 'freeCashFlow',
            'Capital Expenditure': 'capitalExpenditure',
            'Dividend Payments': 'dividendsPaid',
            'Net Income': 'netIncome',
            'Cash Flow from Investing': 'netCashUsedForInvestingActivites',
            'Cash Flow from Financing': 'netCashUsedProvidedByFinancingActivities'
        }
        
        # Create a DataFrame for the selected metrics
        display_cash_flow = cash_flow_df[['date'] + list(cash_flow_metrics.values())].copy()
        display_cash_flow.columns = ['Date'] + list(cash_flow_metrics.keys())
        
        # Format the values
        for col in display_cash_flow.columns:
            if col != 'Date':
                display_cash_flow[col] = display_cash_flow[col].apply(lambda x: format_number(x) if pd.notnull(x) else "N/A")
        
        # Display the table
        st.dataframe(display_cash_flow, use_container_width=True)
        
        # Create cash flow charts
        # 1. Operating Cash Flow vs Net Income
        fig_ocf_ni = go.Figure()
        
        fig_ocf_ni.add_trace(go.Bar(
            x=cash_flow_df['date'],
            y=cash_flow_df['operatingCashFlow'],
            name='Operating Cash Flow',
            marker_color='#00CC96'
        ))
        
        fig_ocf_ni.add_trace(go.Bar(
            x=cash_flow_df['date'],
            y=cash_flow_df['netIncome'],
            name='Net Income',
            marker_color='#636EFA'
        ))
        
        fig_ocf_ni.update_layout(
            title='Operating Cash Flow vs Net Income',
            xaxis_title='Date',
            yaxis_title='Amount',
            template='plotly_white',
            height=500,
            barmode='group',
            showlegend=True
        )
        
        st.plotly_chart(fig_ocf_ni, use_container_width=True)
        
        # 2. Free Cash Flow and Capital Expenditure
        fig_fcf_capex = go.Figure()
        
        fig_fcf_capex.add_trace(go.Bar(
            x=cash_flow_df['date'],
            y=cash_flow_df['freeCashFlow'],
            name='Free Cash Flow',
            marker_color='#00CC96'
        ))
        
        fig_fcf_capex.add_trace(go.Bar(
            x=cash_flow_df['date'],
            y=cash_flow_df['capitalExpenditure'],
            name='Capital Expenditure',
            marker_color='#EF553B'
        ))
        
        fig_fcf_capex.update_layout(
            title='Free Cash Flow and Capital Expenditure',
            xaxis_title='Date',
            yaxis_title='Amount',
            template='plotly_white',
            height=500,
            barmode='group',
            showlegend=True
        )
        
        st.plotly_chart(fig_fcf_capex, use_container_width=True)
        
        # 3. Cash Flow Components
        fig_cf_components = go.Figure()
        
        fig_cf_components.add_trace(go.Bar(
            x=cash_flow_df['date'],
            y=cash_flow_df['operatingCashFlow'],
            name='Operating',
            marker_color='#00CC96'
        ))
        
        fig_cf_components.add_trace(go.Bar(
            x=cash_flow_df['date'],
            y=cash_flow_df['netCashUsedForInvestingActivites'],
            name='Investing',
            marker_color='#636EFA'
        ))
        
        fig_cf_components.add_trace(go.Bar(
            x=cash_flow_df['date'],
            y=cash_flow_df['netCashUsedProvidedByFinancingActivities'],
            name='Financing',
            marker_color='#EF553B'
        ))
        
        fig_cf_components.update_layout(
            title='Cash Flow Components',
            xaxis_title='Date',
            yaxis_title='Amount',
            template='plotly_white',
            height=500,
            barmode='group',
            showlegend=True
        )
        
        st.plotly_chart(fig_cf_components, use_container_width=True)
        
        # 4. Dividend Payments
        fig_dividends = go.Figure()
        
        fig_dividends.add_trace(go.Bar(
            x=cash_flow_df['date'],
            y=cash_flow_df['dividendsPaid'].abs(),  # Convert to positive for better visualization
            name='Dividend Payments',
            marker_color='#9B59B6'
        ))
        
        fig_dividends.update_layout(
            title='Dividend Payments',
            xaxis_title='Date',
            yaxis_title='Amount',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_dividends, use_container_width=True)
        
        # After the Cash Flow Analysis section, add the Dividend History section
        if dividend_data:
            st.subheader("Dividend History")
            
            # Create a DataFrame for dividend data
            dividend_df = pd.DataFrame(dividend_data)
            dividend_df['date'] = pd.to_datetime(dividend_df['date'])
            dividend_df = dividend_df.sort_values('date', ascending=False)  # Most recent first
            
            # Calculate dividend yield (dividend / price * 100)
            # We need to fetch the stock price for each dividend date
            price_url = f"{BASE_URL}/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
            price_response = requests.get(price_url)
            if price_response.status_code == 200:
                price_data = price_response.json()
                if 'historical' in price_data:
                    price_df = pd.DataFrame(price_data['historical'])
                    price_df['date'] = pd.to_datetime(price_df['date'])
                    
                    # Merge dividend data with price data
                    merged_df = pd.merge(dividend_df, price_df[['date', 'adjClose']], on='date', how='left')
                    
                    # Calculate annual dividend yield
                    # First, get the annual dividend amount
                    merged_df['year'] = pd.to_datetime(merged_df['date']).dt.year
                    
                    # Filter for last 10 years
                    current_year = pd.Timestamp.now().year
                    start_year = current_year - 10
                    merged_df = merged_df[merged_df['year'] >= start_year]
                    
                    annual_dividends = merged_df.groupby('year')['dividend'].sum().reset_index()
                    
                    # Get the year-end price for each year
                    year_end_prices = price_df.copy()
                    year_end_prices['year'] = pd.to_datetime(year_end_prices['date']).dt.year
                    year_end_prices = year_end_prices[year_end_prices['year'] >= start_year]
                    year_end_prices = year_end_prices.sort_values('date').groupby('year').last()[['adjClose']].reset_index()
                    
                    # Merge annual dividends with year-end prices
                    annual_data = pd.merge(annual_dividends, year_end_prices, on='year', how='left')
                    
                    # Calculate dividend yield using year-end price
                    annual_data['dividendYield'] = (annual_data['dividend'] / annual_data['adjClose'] * 100)
                    
                    # Sort by year in descending order
                    annual_data = annual_data.sort_values('year', ascending=False)
                    
                    # Create the combined dividend chart
                    fig_dividend = go.Figure()
                    
                    # Add the dividend amount bars
                    fig_dividend.add_trace(go.Bar(
                        x=annual_data['year'],
                        y=annual_data['dividend'],
                        name='Annual Dividend Amount',
                        text=annual_data['dividend'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A"),
                        textposition='outside',
                        marker_color='#808080',
                        yaxis='y'
                    ))
                    
                    # Add the dividend yield line
                    fig_dividend.add_trace(go.Scatter(
                        x=annual_data['year'],
                        y=annual_data['dividendYield'],
                        name='Dividend Yield',
                        line=dict(color='#636EFA', width=2),
                        mode='lines+markers+text',
                        text=annual_data['dividendYield'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"),
                        textposition='top center',
                        yaxis='y2'
                    ))
                    
                    # Update layout with two y-axes
                    fig_dividend.update_layout(
                        title=f'Dividend Analysis ({annual_data['year'].min()}-{annual_data['year'].max()})',
                        xaxis_title='Year',
                        yaxis=dict(
                            title='Annual Dividend Amount ($)',
                            side='left',
                            showgrid=False
                        ),
                        yaxis2=dict(
                            title='Average Dividend Yield (%)',
                            side='right',
                            overlaying='y',
                            showgrid=False
                        ),
                        template='plotly_white',
                        height=500,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig_dividend, use_container_width=True)

        # After the Dividend History section, add Insider Trading and Institutional Ownership sections
        # Insider Trading Section
        st.subheader("Insider Trading")
        insider_data = fetch_insider_trading(ticker)
        
        if insider_data:
            # Create a DataFrame for insider trading
            insider_df = pd.DataFrame(insider_data)
            insider_df['transactionDate'] = pd.to_datetime(insider_df['transactionDate'])
            insider_df = insider_df.sort_values('transactionDate', ascending=False)
            
            # Format the data for display
            display_insider_df = insider_df[['transactionDate', 'transactionType', 'name', 'transactionPrice', 'sharesTransacted', 'value']].copy()
            display_insider_df.columns = ['Date', 'Type', 'Name', 'Price', 'Shares', 'Value']
            
            # Format the values
            display_insider_df['Price'] = display_insider_df['Price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
            display_insider_df['Shares'] = display_insider_df['Shares'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            display_insider_df['Value'] = display_insider_df['Value'].apply(lambda x: format_number(x) if pd.notnull(x) else "N/A")
            
            # Display the table
            st.dataframe(display_insider_df, use_container_width=True)
            
            # Create a chart for insider transactions
            fig_insider = go.Figure()
            
            # Add buy transactions
            buys = insider_df[insider_df['transactionType'].str.contains('Buy', case=False, na=False)]
            if not buys.empty:
                fig_insider.add_trace(go.Bar(
                    x=buys['transactionDate'],
                    y=buys['sharesTransacted'],
                    name='Buy',
                    marker_color='#00CC96'
                ))
            
            # Add sell transactions
            sells = insider_df[insider_df['transactionType'].str.contains('Sell', case=False, na=False)]
            if not sells.empty:
                fig_insider.add_trace(go.Bar(
                    x=sells['transactionDate'],
                    y=sells['sharesTransacted'].abs(),
                    name='Sell',
                    marker_color='#EF553B'
                ))
            
            fig_insider.update_layout(
                title='Insider Trading Activity',
                xaxis_title='Date',
                yaxis_title='Number of Shares',
                template='plotly_white',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_insider, use_container_width=True)
        
        # Institutional Ownership Section
        st.subheader("Institutional Ownership")
        inst_data = fetch_institutional_ownership(ticker)
        
        if inst_data:
            # Create a DataFrame for institutional ownership
            inst_df = pd.DataFrame(inst_data)
            
            # Format the data for display
            display_inst_df = inst_df[['holder', 'shares', 'dateReported', 'change']].copy()
            display_inst_df.columns = ['Institution', 'Shares Held', 'Date Reported', 'Change']
            
            # Format the values
            display_inst_df['Shares Held'] = display_inst_df['Shares Held'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            display_inst_df['Change'] = display_inst_df['Change'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            display_inst_df['Date Reported'] = pd.to_datetime(display_inst_df['Date Reported']).dt.strftime('%Y-%m-%d')
            
            # Sort by shares held
            display_inst_df = display_inst_df.sort_values('Shares Held', ascending=False)
            
            # Display the table
            st.dataframe(display_inst_df, use_container_width=True)
            
            # Create a bar chart for top 10 institutional holders
            top_10 = inst_df.head(10)
            fig_inst = go.Figure(data=[go.Bar(
                x=top_10['holder'],
                y=top_10['shares'],
                text=top_10['shares'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                marker_color='#636EFA'
            )])
            
            fig_inst.update_layout(
                title='Top 10 Institutional Holders',
                xaxis_title='Institution',
                yaxis_title='Shares Held',
                template='plotly_white',
                height=700,
                showlegend=False,
                xaxis_tickangle=-45,
                margin=dict(t=100, b=100),
                yaxis=dict(
                    tickformat=",.0f",
                    gridcolor='lightgray'
                )
            )
            
            st.plotly_chart(fig_inst, use_container_width=True)
        
        # Add AI Analysis Section at the end
        st.subheader("🤖 AI Analysis Summary")
        
        # Create a summary of key metrics
        summary_points = []
        
        # Company Overview Summary
        if profile:
            summary_points.append(f"**Company Overview:** {company_name} is a {profile.get('industry', 'N/A')} company in the {profile.get('sector', 'N/A')} sector with a market cap of {format_number(profile.get('mktCap'))}.")
        
        # Financial Performance Summary with Historical Trends
        if not df.empty:
            latest_rev = df['Revenue'].iloc[0]
            latest_rev_growth = df['Rev Growth'].iloc[0]
            latest_ni = df['Net Income'].iloc[0]
            latest_ni_growth = df['NI Growth'].iloc[0]
            
            # Calculate 3-year average growth rates
            if len(df) >= 3:
                avg_rev_growth = df['Rev Growth'].head(3).mean()
                avg_ni_growth = df['NI Growth'].head(3).mean()
                summary_points.append(f"**Financial Performance:** The company reported revenue of {format_number(latest_rev)} with a growth rate of {format_percentage(latest_rev_growth)}. Net income was {format_number(latest_ni)} with a growth rate of {format_percentage(latest_ni_growth)}. Over the past 3 years, revenue has grown at an average rate of {format_percentage(avg_rev_growth)} and net income at {format_percentage(avg_ni_growth)}.")
            else:
                summary_points.append(f"**Financial Performance:** The company reported revenue of {format_number(latest_rev)} with a growth rate of {format_percentage(latest_rev_growth)}. Net income was {format_number(latest_ni)} with a growth rate of {format_percentage(latest_ni_growth)}.")
            
            # Analyze profitability trends
            if 'ROE' in df.columns:
                latest_roe = df['ROE'].iloc[0]
                avg_roe = df['ROE'].head(3).mean()
                summary_points.append(f"**Profitability:** The company's Return on Equity (ROE) is {format_percentage(latest_roe)}, with a 3-year average of {format_percentage(avg_roe)}.")
        
        # Cash Flow Analysis with Trends
        if not cash_flow_df.empty:
            latest_ocf = cash_flow_df['operatingCashFlow'].iloc[0]
            latest_fcf = cash_flow_df['freeCashFlow'].iloc[0]
            
            # Calculate cash flow trends
            if len(cash_flow_df) >= 3:
                avg_ocf = cash_flow_df['operatingCashFlow'].head(3).mean()
                avg_fcf = cash_flow_df['freeCashFlow'].head(3).mean()
                ocf_trend = "increasing" if latest_ocf > avg_ocf else "decreasing"
                fcf_trend = "increasing" if latest_fcf > avg_fcf else "decreasing"
                
                summary_points.append(f"**Cash Flow:** Operating cash flow was {format_number(latest_ocf)} with free cash flow of {format_number(latest_fcf)}. Both metrics show a {ocf_trend} trend compared to their 3-year averages of {format_number(avg_ocf)} and {format_number(avg_fcf)} respectively.")
            else:
                summary_points.append(f"**Cash Flow:** Operating cash flow was {format_number(latest_ocf)} with free cash flow of {format_number(latest_fcf)}.")
            
            # Analyze cash flow quality
            if 'netIncome' in cash_flow_df.columns:
                latest_ni = cash_flow_df['netIncome'].iloc[0]
                ocf_to_ni = latest_ocf / latest_ni if latest_ni != 0 else 0
                summary_points.append(f"**Cash Flow Quality:** The company's operating cash flow to net income ratio is {format_ratio(ocf_to_ni)}, indicating {'strong' if ocf_to_ni > 1 else 'weak'} cash flow quality.")
        
        # Dividend Analysis with Historical Context
        if dividend_data:
            latest_dividend = dividend_data[0].get('dividend', 0)
            
            # Calculate dividend growth rate
            if len(dividend_data) >= 4:  # At least 4 quarters
                prev_year_dividend = sum(d['dividend'] for d in dividend_data[4:8])
                current_year_dividend = sum(d['dividend'] for d in dividend_data[0:4])
                div_growth = ((current_year_dividend - prev_year_dividend) / prev_year_dividend * 100) if prev_year_dividend != 0 else 0
                
                summary_points.append(f"**Dividend:** The company pays a quarterly dividend of ${latest_dividend:.2f} per share, resulting in a current yield of {current_dividend_yield}. The annual dividend has {'increased' if div_growth > 0 else 'decreased'} by {format_percentage(div_growth)} compared to the previous year.")
            else:
                summary_points.append(f"**Dividend:** The company pays a quarterly dividend of ${latest_dividend:.2f} per share, resulting in a current yield of {current_dividend_yield}.")
        
        # Insider Trading Analysis with Context
        if insider_data:
            recent_buys = sum(1 for x in insider_data if 'Buy' in x.get('transactionType', ''))
            recent_sells = sum(1 for x in insider_data if 'Sell' in x.get('transactionType', ''))
            
            # Calculate total value of insider transactions
            buy_value = sum(float(x.get('value', 0)) for x in insider_data if 'Buy' in x.get('transactionType', ''))
            sell_value = sum(float(x.get('value', 0)) for x in insider_data if 'Sell' in x.get('transactionType', ''))
            
            summary_points.append(f"**Insider Activity:** In recent transactions, there were {recent_buys} insider buys (${format_number(buy_value)}) and {recent_sells} insider sells (${format_number(sell_value)}). This suggests {'positive' if buy_value > sell_value else 'negative'} insider sentiment.")
        
        # Institutional Ownership Analysis with Trends
        if inst_data:
            top_holder = inst_data[0].get('holder', 'N/A')
            top_shares = inst_data[0].get('shares', 0)
            
            # Calculate total institutional ownership
            total_inst_shares = sum(float(x.get('shares', 0)) for x in inst_data)
            total_shares = float(profile.get('sharesOutstanding', 0)) if profile else 0
            inst_ownership_pct = (total_inst_shares / total_shares * 100) if total_shares > 0 else 0
            
            summary_points.append(f"**Institutional Ownership:** The largest institutional holder is {top_holder} with {format_number(top_shares)} shares. Institutional investors own approximately {format_percentage(inst_ownership_pct)} of the company's shares.")
        
        # Valuation Analysis with Historical Context
        if not df.empty:
            latest_pe = df['P/E'].iloc[0]
            latest_ps = df['P/S'].iloc[0]
            latest_pb = df['P/B'].iloc[0]
            
            # Calculate historical averages
            if len(df) >= 3:
                avg_pe = df['P/E'].head(3).mean()
                avg_ps = df['P/S'].head(3).mean()
                avg_pb = df['P/B'].head(3).mean()
                
                pe_status = "overvalued" if latest_pe > avg_pe else "undervalued"
                ps_status = "overvalued" if latest_ps > avg_ps else "undervalued"
                pb_status = "overvalued" if latest_pb > avg_pb else "undervalued"
                
                summary_points.append(f"**Valuation:** The company trades at a P/E ratio of {format_ratio(latest_pe)} (vs. 3-year avg of {format_ratio(avg_pe)}), P/S ratio of {format_ratio(latest_ps)} (vs. {format_ratio(avg_ps)}), and P/B ratio of {format_ratio(latest_pb)} (vs. {format_ratio(avg_pb)}). Based on historical averages, the stock appears {pe_status} on a P/E basis, {ps_status} on a P/S basis, and {pb_status} on a P/B basis.")
            else:
                summary_points.append(f"**Valuation:** The company trades at a P/E ratio of {format_ratio(latest_pe)}, P/S ratio of {format_ratio(latest_ps)}, and P/B ratio of {format_ratio(latest_pb)}.")
        
        # Display the summary
        for point in summary_points:
            st.markdown(point)
        
        # Add a disclaimer
        st.markdown("""
        ---
        *This analysis is generated automatically based on the available data and should not be considered as financial advice. 
        Always do your own research before making investment decisions.*
        """)

    except Exception as e:
        st.error(f"Error fetching data for {ticker}. Please check if the ticker symbol is correct.")
        st.error(f"Detailed error: {str(e)}") 
