import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Stock Analysis Tool")
st.markdown("Enter a stock ticker symbol to analyze its historical financial data.")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL):", "").upper()

# Add period selection
period = st.radio("Select Period:", ["Annual", "Quarterly"], horizontal=True)

# FMP API configuration
FMP_API_KEY = os.getenv('FMP_API_KEY')
if not FMP_API_KEY:
    st.error("Please set your FMP_API_KEY in the .env file")
    st.stop()

BASE_URL = "https://financialmodelingprep.com/api/v3"

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
        
        # After the Total Debt chart and before the Analyst Estimates section, add the Dividend Yield chart
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
        
        # After all the existing charts, add the Analyst Estimates section
        if analyst_estimates:
            st.subheader("Analyst Estimates")
            
            # Create a DataFrame for analyst estimates
            estimates_data = []
            for estimate in analyst_estimates:
                estimates_data.append({
                    'Period': f"{estimate['date']}",
                    'Revenue Est': estimate.get('estimatedRevenueAvg', None),
                    'Revenue High': estimate.get('estimatedRevenueHigh', None),
                    'Revenue Low': estimate.get('estimatedRevenueLow', None),
                    'EPS Est': estimate.get('estimatedEpsAvg', None),
                    'EPS High': estimate.get('estimatedEpsHigh', None),
                    'EPS Low': estimate.get('estimatedEpsLow', None),
                    'Net Income Est': estimate.get('estimatedNetIncomeAvg', None),
                    'Net Income High': estimate.get('estimatedNetIncomeHigh', None),
                    'Net Income Low': estimate.get('estimatedNetIncomeLow', None),
                    'EBITDA Est': estimate.get('estimatedEbitdaAvg', None),
                    'EBITDA High': estimate.get('estimatedEbitdaHigh', None),
                    'EBITDA Low': estimate.get('estimatedEbitdaLow', None),
                    'Revenue Analysts': estimate.get('numberAnalystEstimatedRevenue', None),
                    'EPS Analysts': estimate.get('numberAnalystsEstimatedEps', None)
                })
            
            estimates_df = pd.DataFrame(estimates_data)
            estimates_df = estimates_df.sort_values('Period', ascending=False)  # Most recent first
            
            # Create a display DataFrame for the table
            display_estimates_df = estimates_df.copy()
            
            # Format the monetary columns using format_number for display
            monetary_columns = ['Revenue Est', 'Revenue High', 'Revenue Low', 
                              'Net Income Est', 'Net Income High', 'Net Income Low',
                              'EBITDA Est', 'EBITDA High', 'EBITDA Low']
            
            for col in monetary_columns:
                display_estimates_df[col] = display_estimates_df[col].apply(lambda x: format_number(x) if pd.notnull(x) else "N/A")
            
            # Display the estimates table
            st.dataframe(
                display_estimates_df,
                use_container_width=True,
                column_config={
                    'Period': st.column_config.TextColumn('Period'),
                    'Revenue Est': st.column_config.TextColumn('Revenue Est'),
                    'Revenue High': st.column_config.TextColumn('Revenue High'),
                    'Revenue Low': st.column_config.TextColumn('Revenue Low'),
                    'EPS Est': st.column_config.NumberColumn('EPS Est', format="$%.2f"),
                    'EPS High': st.column_config.NumberColumn('EPS High', format="$%.2f"),
                    'EPS Low': st.column_config.NumberColumn('EPS Low', format="$%.2f"),
                    'Net Income Est': st.column_config.TextColumn('Net Income Est'),
                    'Net Income High': st.column_config.TextColumn('Net Income High'),
                    'Net Income Low': st.column_config.TextColumn('Net Income Low'),
                    'EBITDA Est': st.column_config.TextColumn('EBITDA Est'),
                    'EBITDA High': st.column_config.TextColumn('EBITDA High'),
                    'EBITDA Low': st.column_config.TextColumn('EBITDA Low'),
                    'Revenue Analysts': st.column_config.NumberColumn('Revenue Analysts', format="%d"),
                    'EPS Analysts': st.column_config.NumberColumn('EPS Analysts', format="%d")
                }
            )
            
            # Create a chart for EPS estimates
            fig_eps_est = go.Figure()
            
            # Add the average estimate line
            fig_eps_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['EPS Est'],
                name='Average Estimate',
                line=dict(color='#00CC96', width=2),
                mode='lines+markers+text',
                text=estimates_df['EPS Est'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A"),
                textposition='top center'
            ))
            
            # Add the high and low range
            fig_eps_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['EPS High'],
                name='High Estimate',
                line=dict(color='rgba(0,204,150,0.2)', width=0),
                showlegend=False
            ))
            
            fig_eps_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['EPS Low'],
                name='Low Estimate',
                line=dict(color='rgba(0,204,150,0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(0,204,150,0.1)',
                showlegend=False
            ))
            
            fig_eps_est.update_layout(
                title='EPS Estimates Over Time',
                xaxis_title='Period',
                yaxis_title='EPS',
                template='plotly_white',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_eps_est, use_container_width=True)
            
            # Create a chart for Revenue estimates
            fig_rev_est = go.Figure()
            
            # Add the average estimate line
            fig_rev_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['Revenue Est'],
                name='Average Estimate',
                line=dict(color='#9B59B6', width=2),
                mode='lines+markers+text',
                text=estimates_df['Revenue Est'].apply(lambda x: format_number(x) if pd.notnull(x) else "N/A"),
                textposition='top center'
            ))
            
            # Add the high and low range
            fig_rev_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['Revenue High'],
                name='High Estimate',
                line=dict(color='rgba(155,89,182,0.2)', width=0),
                showlegend=False
            ))
            
            fig_rev_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['Revenue Low'],
                name='Low Estimate',
                line=dict(color='rgba(155,89,182,0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(155,89,182,0.1)',
                showlegend=False
            ))
            
            fig_rev_est.update_layout(
                title='Revenue Estimates Over Time',
                xaxis_title='Period',
                yaxis_title='Revenue',
                template='plotly_white',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_rev_est, use_container_width=True)
            
            # Create a chart for Net Income estimates
            fig_ni_est = go.Figure()
            
            # Add the average estimate line
            fig_ni_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['Net Income Est'],
                name='Average Estimate',
                line=dict(color='#FFA15A', width=2),
                mode='lines+markers+text',
                text=estimates_df['Net Income Est'].apply(lambda x: format_number(x) if pd.notnull(x) else "N/A"),
                textposition='top center'
            ))
            
            # Add the high and low range
            fig_ni_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['Net Income High'],
                name='High Estimate',
                line=dict(color='rgba(255,161,90,0.2)', width=0),
                showlegend=False
            ))
            
            fig_ni_est.add_trace(go.Scatter(
                x=estimates_df['Period'],
                y=estimates_df['Net Income Low'],
                name='Low Estimate',
                line=dict(color='rgba(255,161,90,0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(255,161,90,0.1)',
                showlegend=False
            ))
            
            fig_ni_est.update_layout(
                title='Net Income Estimates Over Time',
                xaxis_title='Period',
                yaxis_title='Net Income',
                template='plotly_white',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_ni_est, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}. Please check if the ticker symbol is correct.")
        st.error(f"Detailed error: {str(e)}") 
