import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Portfolio Dashboard v7",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .positive-return {
        color: #28a745;
        font-weight: bold;
    }
    .negative-return {
        color: #dc3545;
        font-weight: bold;
    }
    .holdings-table {
        font-size: 0.9rem;
    }
    .benchmark-table {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class PortfolioTracker:
    def __init__(self):
        self.holdings = pd.DataFrame()
        self.transactions = pd.DataFrame()
        self.market_data = {}
        
        # Load stock split data
        with open("stock_splits.json") as f:
            self.stock_splits = json.load(f).get("known_splits", {})
        
        # Market suffix mappings for yfinance
        self.market_suffixes = {
            'USD': '', 'SGD': '.SI', 'EUR': '.PA', 'GBP': '.L', 
            'CAD': '.TO', 'AUD': '.AX', 'HKD': '.HK', 'JPY': '.T',
            'BRL': '.SA', 'KRW': '.KS'
        }
        
        # Country mappings based on currency
        self.currency_to_country = {
            'USD': 'United States', 'SGD': 'Singapore', 'EUR': 'Europe',
            'GBP': 'United Kingdom', 'CAD': 'Canada', 'AUD': 'Australia',
            'HKD': 'Hong Kong', 'JPY': 'Japan', 'BRL': 'Brazil', 'KRW': 'South Korea'
        }
        
        # FX pairs for currency conversion
        self.fx_pairs_to_sgd = {
            'USD': 'USDSGD=X', 'EUR': 'EURSGD=X', 'GBP': 'GBPSGD=X',
            'JPY': 'JPYSGD=X', 'CAD': 'CADSGD=X', 'AUD': 'AUDSGD=X',
            'HKD': 'HKDSGD=X', 'SGD': None
        }
        
        # FX pairs for USD conversion
        self.fx_pairs_to_usd = {
            'SGD': 'SGDUSD=X', 'EUR': 'EURUSD=X', 'GBP': 'GBPUSD=X',
            'JPY': 'JPYUSD=X', 'CAD': 'CADUSD=X', 'AUD': 'AUDUSD=X',
            'HKD': 'HKDUSD=X', 'USD': None
        }
        
        # Benchmark indices
        self.benchmarks = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'NYSE': '^NYA',
            'STI (Singapore)': '^STI',
            'MSCI World': 'URTH',
            'VTI (Total Stock)': 'VTI'
        }

    @st.cache_data
    def load_csv_data(_self, uploaded_file):
        """Load and process CSV data"""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Column mapping
            column_mapping = {
                'Date': 'date', 'Ticker': 'symbol', 'Action': 'type',
                'Shares': 'quantity', 'Price': 'price', 'Commission': 'fees',
                'Currency': 'currency', 'Note': 'note'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Data cleaning
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['type'] = df['type'].astype(str).str.lower().str.strip()
            df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
            df['currency'] = df['currency'].astype(str).str.upper().str.strip()
            df['note'] = df['note'].astype(str).fillna('')
            
            # Clean numeric fields
            for col in ['quantity', 'price']:
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['fees'] = pd.to_numeric(df['fees'], errors='coerce').fillna(0)
            df['total_value'] = df['quantity'] * df['price'] + df['fees']
            
            # Filter valid data
            df = df.dropna(subset=['date', 'symbol', 'quantity', 'price'])
            df = df[(df['quantity'] > 0) & (df['price'] > 0)]
            
            # Standardize actions
            action_mapping = {
                'buy': 'buy', 'purchase': 'buy', 'bought': 'buy',
                'sell': 'sell', 'sale': 'sell', 'sold': 'sell'
            }
            df['type'] = df['type'].map(action_mapping).fillna(df['type'])
            
            # Add country info
            df['country'] = df['currency'].map(_self.currency_to_country).fillna('Other')

            # --- Apply stock splits ---
            def apply_stock_splits(row):
                splits = _self.stock_splits.get(row['symbol'], [])
                for split in splits:
                    split_date = pd.to_datetime(split['date'])
                    if row['date'] < split_date:
                        ratio = split['ratio']
                        row['quantity'] *= ratio
                        row['price'] /= ratio
                return row

            df = df.apply(apply_stock_splits, axis=1)
            # --------------------------
            
            return df
            
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return pd.DataFrame()


    def calculate_holdings(self, transactions_df):
        """Calculate current holdings from transactions"""
        if transactions_df.empty:
            return pd.DataFrame()
        
        holdings_list = []
        
        for symbol in transactions_df['symbol'].unique():
            if pd.isna(symbol):
                continue
                
            symbol_txns = transactions_df[transactions_df['symbol'] == symbol]
            total_quantity = 0
            total_cost = 0
            
            for _, txn in symbol_txns.iterrows():
                if pd.isna(txn['quantity']) or pd.isna(txn['price']):
                    continue
                    
                if txn['type'] == 'buy':
                    total_quantity += txn['quantity']
                    total_cost += txn['total_value']
                elif txn['type'] == 'sell':
                    if total_quantity > 0:
                        avg_cost = total_cost / total_quantity
                        sold_cost = txn['quantity'] * avg_cost
                        total_cost -= sold_cost
                    total_quantity -= txn['quantity']
            
            if total_quantity > 0:
                currency = symbol_txns.iloc[-1]['currency']
                country = symbol_txns.iloc[-1]['country']
                avg_price = total_cost / total_quantity
                
                holdings_list.append({
                    'symbol': str(symbol).upper().strip(),
                    'quantity': total_quantity,
                    'avg_cost': avg_price,
                    'currency': currency if not pd.isna(currency) else 'USD',
                    'country': country if not pd.isna(country) else 'Other',
                    'total_cost': total_cost
                })
        
        return pd.DataFrame(holdings_list)

    @st.cache_data(ttl=300)
    def fetch_exchange_rates_to_sgd(_self):
        """Get exchange rates to SGD"""
        rates = {'SGD': 1.0}
        
        for currency, fx_symbol in _self.fx_pairs_to_sgd.items():
            if fx_symbol is None:
                continue
            try:
                ticker = yf.Ticker(fx_symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    rates[currency] = hist['Close'].iloc[-1]
            except:
                rates[currency] = 1.0
        
        return rates

    @st.cache_data(ttl=300)
    def fetch_exchange_rates_to_usd(_self):
        """Get exchange rates to USD"""
        rates = {'USD': 1.0}
        
        for currency, fx_symbol in _self.fx_pairs_to_usd.items():
            if fx_symbol is None:
                continue
            try:
                ticker = yf.Ticker(fx_symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    rates[currency] = hist['Close'].iloc[-1]
            except:
                rates[currency] = 1.0
        
        return rates

    @st.cache_data(ttl=300)
    def fetch_benchmark_data(_self, period='1y'):
        """Fetch benchmark performance data"""
        benchmark_data = {}
        
        for name, symbol in _self.benchmarks.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    
                    returns = {}
                    periods = {'1M': 21, '3M': 63, '6M': 126, 'YTD': None, '1Y': 252}
                    
                    for period_name, days in periods.items():
                        if period_name == 'YTD':
                            year_start = hist[hist.index.year == hist.index[-1].year].iloc[0]['Close']
                            returns[period_name] = ((current_price / year_start) - 1) * 100
                        elif days and len(hist) > days:
                            past_price = hist['Close'].iloc[-days]
                            returns[period_name] = ((current_price / past_price) - 1) * 100
                        else:
                            returns[period_name] = 0
                    
                    benchmark_data[name] = returns
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}")
        
        return benchmark_data

    @st.cache_data(ttl=300)
    def fetch_stock_prices(_self, symbols_currencies):
        """Fetch stock prices with performance data"""
        prices = {}
        
        for symbol, currency in symbols_currencies:
            try:
                # Convert to yfinance symbol
                yf_symbol = symbol
                if currency in _self.market_suffixes and not '.' in symbol:
                    yf_symbol = symbol + _self.market_suffixes[currency]
                
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(period='1y')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    
                    # Calculate performance metrics
                    returns = {}
                    periods = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252}
                    
                    for period_name, days in periods.items():
                        if len(hist) > days:
                            past_price = hist['Close'].iloc[-days]
                            returns[period_name] = ((current_price / past_price) - 1) * 100
                        else:
                            returns[period_name] = 0
                    
                    # YTD calculation
                    year_start_data = hist[hist.index.year == hist.index[-1].year]
                    if not year_start_data.empty:
                        year_start_price = year_start_data.iloc[0]['Close']
                        returns['YTD'] = ((current_price / year_start_price) - 1) * 100
                    else:
                        returns['YTD'] = 0
                    
                    try:
                        info = ticker.info
                        sector = info.get('sector', 'Unknown')
                        long_name = info.get('longName', symbol)
                    except:
                        sector = 'Unknown'
                        long_name = symbol
                    
                    prices[symbol] = {
                        'price': float(current_price),
                        'currency': currency,
                        'sector': sector,
                        'long_name': long_name,
                        'yf_symbol': yf_symbol,
                        'returns': returns,
                        'price_history': hist
                    }
                    
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                prices[symbol] = {
                    'price': 0, 'currency': currency, 'sector': 'Unknown',
                    'long_name': symbol, 'yf_symbol': symbol,
                    'returns': {p: 0 for p in ['1M', '3M', '6M', 'YTD', '1Y']},
                    'price_history': pd.DataFrame()
                }
        
        return prices

    def convert_currency(self, amount, from_currency, to_currency, sgd_rates, usd_rates):
        """Convert between currencies"""
        if from_currency == to_currency:
            return amount
        
        if to_currency == 'SGD':
            if from_currency == 'USD':
                return amount * sgd_rates.get('USD', 1.0)
            else:
                rate = sgd_rates.get(from_currency, 1.0)
                return amount * rate
        
        elif to_currency == 'USD':
            if from_currency == 'SGD':
                return amount * usd_rates.get('SGD', 1.0)
            else:
                rate = usd_rates.get(from_currency, 1.0)
                return amount * rate
        
        return amount

    def calculate_portfolio_metrics(self, holdings_df, transactions_df, base_currency='SGD'):
        """Calculate comprehensive portfolio metrics"""
        if holdings_df.empty:
            return None
        
        # Get exchange rates for both currencies
        sgd_rates = self.fetch_exchange_rates_to_sgd()
        usd_rates = self.fetch_exchange_rates_to_usd()
        
        symbols_currencies = list(zip(holdings_df['symbol'], holdings_df['currency']))
        prices = self.fetch_stock_prices(symbols_currencies)
        
        portfolio_data = []
        total_value = 0
        total_cost = 0
        
        for _, holding in holdings_df.iterrows():
            symbol = holding['symbol']
            quantity = holding['quantity']
            avg_cost = holding['avg_cost']
            currency = holding['currency']
            country = holding['country']
            
            if symbol in prices and prices[symbol]['price'] > 0:
                current_price = prices[symbol]['price']
                
                # Convert to base currency
                current_price_base = self.convert_currency(current_price, currency, base_currency, sgd_rates, usd_rates)
                avg_cost_base = self.convert_currency(avg_cost, currency, base_currency, sgd_rates, usd_rates)
                
                current_value = quantity * current_price_base
                cost_basis = quantity * avg_cost_base
                
                gain_loss = current_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                
                portfolio_data.append({
                    'symbol': symbol,
                    'long_name': prices[symbol]['long_name'],
                    'quantity': quantity,
                    'avg_cost': avg_cost_base,
                    'current_price': current_price_base,
                    'current_value': current_value,
                    'cost_basis': cost_basis,
                    'gain_loss': gain_loss,
                    'gain_loss_pct': gain_loss_pct,
                    'country': country,
                    'currency': currency,
                    'sector': prices[symbol]['sector'],
                    'yf_symbol': prices[symbol]['yf_symbol'],
                    'returns_1M': prices[symbol]['returns'].get('1M', 0),
                    'returns_3M': prices[symbol]['returns'].get('3M', 0),
                    'returns_6M': prices[symbol]['returns'].get('6M', 0),
                    'returns_YTD': prices[symbol]['returns'].get('YTD', 0),
                    'returns_1Y': prices[symbol]['returns'].get('1Y', 0)
                })
                
                total_value += current_value
                total_cost += cost_basis
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        if not portfolio_df.empty:
            portfolio_df['weight_pct'] = (portfolio_df['current_value'] / total_value) * 100
        
        # Calculate portfolio-level returns
        portfolio_returns = self.calculate_portfolio_returns(portfolio_df)
        
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'portfolio_df': portfolio_df,
            'total_value': total_value,
            'total_cost': total_cost,
            'total_gain_loss': total_gain_loss,
            'total_gain_loss_pct': total_gain_loss_pct,
            'num_holdings': len(portfolio_df),
            'sgd_rates': sgd_rates,
            'usd_rates': usd_rates,
            'portfolio_returns': portfolio_returns,
            'base_currency': base_currency
        }

    def calculate_portfolio_returns(self, portfolio_df):
        """Calculate time-weighted portfolio returns"""
        if portfolio_df.empty:
            return {p: 0 for p in ['1M', '3M', '6M', 'YTD', '1Y']}
        
        total_value = portfolio_df['current_value'].sum()
        
        returns = {}
        for period in ['1M', '3M', '6M', 'YTD', '1Y']:
            weighted_return = 0
            for _, holding in portfolio_df.iterrows():
                weight = holding['current_value'] / total_value
                holding_return = holding[f'returns_{period}']
                weighted_return += weight * holding_return
            returns[period] = weighted_return
        
        return returns

def create_allocation_pie_chart(portfolio_df, group_by='country'):
    """Create enhanced allocation pie chart"""
    
    allocation = portfolio_df.groupby(group_by)['current_value'].sum().reset_index()
    allocation['percentage'] = (allocation['current_value'] / allocation['current_value'].sum()) * 100
    allocation = allocation.sort_values('percentage', ascending=False)
    
    fig = go.Figure(data=[go.Pie(
        labels=allocation[group_by],
        values=allocation['current_value'],
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12),
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        ),
        pull=[0.1 if i == 0 else 0 for i in range(len(allocation))]
    )])
    
    fig.update_layout(
        title=f'Portfolio Allocation by {group_by.title()}',
        font=dict(size=14),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(t=50, b=50, l=50, r=150)
    )
    
    return fig

def create_benchmark_comparison_table(portfolio_returns, benchmark_data):
    """Create benchmark comparison table"""
    
    comparison_data = []
    
    # Add portfolio data
    portfolio_row = ['Portfolio']
    for period in ['1M', '3M', '6M', 'YTD', '1Y']:
        value = portfolio_returns.get(period, 0)
        portfolio_row.append(f"{value:+.2f}%")
    comparison_data.append(portfolio_row)
    
    # Add benchmark data
    for benchmark_name, returns in benchmark_data.items():
        row = [benchmark_name]
        for period in ['1M', '3M', '6M', 'YTD', '1Y']:
            value = returns.get(period, 0)
            row.append(f"{value:+.2f}%")
        comparison_data.append(row)
    
    columns = ['Benchmark', '1M', '3M', '6M', 'YTD', '1Y']
    df = pd.DataFrame(comparison_data, columns=columns)
    
    return df
def create_holdings_table(portfolio_df, base_currency='SGD', sort_by='current_value', ascending=False):
    """Create sortable detailed holdings table"""
    
    if portfolio_df.empty:
        return pd.DataFrame()
    
    # 1) Sort by the raw numeric column first
    df_sorted = portfolio_df.sort_values(sort_by, ascending=ascending)
    
    # 2) Then select & format only after sorting
    currency_symbol = 'S$' if base_currency == 'SGD' else '$'
    display_df = df_sorted[['symbol', 'long_name', 'current_value', 'weight_pct', 
                            'gain_loss', 'gain_loss_pct', 'quantity', 'current_price', 
                            'avg_cost', 'country', 'sector']].copy()
    
    # 3) Now format the numbers for display
    
    display_df['current_value'] = display_df['current_value'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
    display_df['weight_pct'] = display_df['weight_pct'].apply(lambda x: f"{x:.2f}%")
    display_df['gain_loss'] = display_df['gain_loss'].apply(lambda x: f"{currency_symbol}{x:+,.2f}")
    display_df['gain_loss_pct'] = display_df['gain_loss_pct'].apply(lambda x: f"{x:+.2f}%")
    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"{currency_symbol}{x:.3f}")
    display_df['avg_cost'] = display_df['avg_cost'].apply(lambda x: f"{currency_symbol}{x:.3f}")
    display_df['quantity'] = display_df['quantity'].apply(lambda x: f"{x:,.0f}")
    
    display_df.columns = ['Symbol', 'Name', f'Value ({base_currency})', 'Weight', 
                         f'Gain/Loss ({base_currency})', 'Return (%)', 'Quantity', 
                         f'Price ({base_currency})', f'Avg Cost ({base_currency})', 
                         'Country', 'Sector']
    
    return display_df

def create_aggregation_page(portfolio_df, base_currency='SGD'):
    """Create aggregation analysis page"""
    
    if portfolio_df.empty:
        st.warning("No portfolio data available for aggregation.")
        return

    currency_symbol = 'S$' if base_currency == 'SGD' else '$'

    st.header("📊 Portfolio Aggregation Analysis")
    
    # Aggregation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        agg_by = st.selectbox(
            "Aggregate by:",
            ['Country', 'Sector', 'Currency', 'Symbol'],
            key='agg_by'
        )
    
    with col2:
        metric = st.selectbox(
            "Show metric:",
            ['Portfolio Value', 'Gain/Loss Amount', 'Gain/Loss %', 'Weight %'],
            key='agg_metric'
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart type:",
            ['Bar Chart', 'Pie Chart', 'Treemap'],
            key='chart_type'
        )
    
    # Map selections to dataframe columns
    agg_mapping = {
        'Country': 'country',
        'Sector': 'sector', 
        'Currency': 'currency',
        'Symbol': 'symbol'
    }
    
    metric_mapping = {
        'Portfolio Value': 'current_value',
        'Gain/Loss Amount': 'gain_loss',
        'Gain/Loss %': 'gain_loss_pct',
        'Weight %': 'weight_pct'
    }
    
    agg_col = agg_mapping[agg_by]
    metric_col = metric_mapping[metric]
    
    # Create aggregated data
    if metric in ['Portfolio Value', 'Gain/Loss Amount']:
        agg_data = portfolio_df.groupby(agg_col)[metric_col].sum().reset_index()
    else:
        # For percentages, use weighted average
        agg_data = portfolio_df.groupby(agg_col).apply(
            lambda x: pd.Series({
                metric_col: (x[metric_col] * x['current_value']).sum() / x['current_value'].sum()
            })
        ).reset_index()
    
    agg_data = agg_data.sort_values(metric_col, ascending=False)
    
    # Create visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if chart_type == 'Bar Chart':
            fig = px.bar(
                agg_data,
                x=agg_col,
                y=metric_col,
                title=f'{metric} by {agg_by}',
                color=metric_col,
                color_continuous_scale='viridis'
            )
            
            if metric in ['Portfolio Value', 'Gain/Loss Amount']:
                fig.update_traces(text=[f'{currency_symbol}{x:,.0f}' for x in agg_data[metric_col]], 
                                textposition='outside')
            else:
                fig.update_traces(text=[f'{x:.1f}%' for x in agg_data[metric_col]], 
                                textposition='outside')
            
            fig.update_layout(xaxis_tickangle=45)
            
        elif chart_type == 'Pie Chart':
            if metric in ['Portfolio Value', 'Weight %']:
                values = agg_data[metric_col]
                
                fig = px.pie(
                    agg_data,
                    values=values,
                    names=agg_col,
                    title=f'{metric} by {agg_by}'
                )
            else:
                st.warning("Pie chart only available for Portfolio Value and Weight %")
                fig = None
                
        elif chart_type == 'Treemap':
            if metric == 'Portfolio Value':
                fig = px.treemap(
                    agg_data,
                    path=[agg_col],
                    values=metric_col,
                    title=f'{metric} by {agg_by}'
                )
            else:
                st.warning("Treemap only available for Portfolio Value")
                fig = None
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Summary Table")
        
        # Format the summary table
        summary_df = agg_data.copy()
        
        if metric in ['Portfolio Value', 'Gain/Loss Amount']:
            summary_df[metric_col] = summary_df[metric_col].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        else:
            summary_df[metric_col] = summary_df[metric_col].apply(lambda x: f"{x:.2f}%")
        
        summary_df.columns = [agg_by, metric]
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        # Additional statistics
        st.subheader("Statistics")
        
        if metric in ['Portfolio Value', 'Gain/Loss Amount']:
            raw_values = agg_data[metric_col]
            st.metric("Total", f"{currency_symbol}{raw_values.sum():,.2f}")
            st.metric("Average", f"{currency_symbol}{raw_values.mean():,.2f}")
            st.metric("Largest", f"{currency_symbol}{raw_values.max():,.2f}")
            st.metric("Count", len(raw_values))
        else:
            raw_values = agg_data[metric_col]
            st.metric("Weighted Average", f"{raw_values.mean():.2f}%")
            st.metric("Highest", f"{raw_values.max():.2f}%")
            st.metric("Lowest", f"{raw_values.min():.2f}%")

def main():
    # Main title
    st.markdown('<h1 class="main-header">Portfolio Dashboard v7</h1>', unsafe_allow_html=True)
    
    # Initialize tracker
    if 'tracker' not in st.session_state:
        st.session_state.tracker = PortfolioTracker()
    
    tracker = st.session_state.tracker
    
    # Navigation
    page = st.sidebar.selectbox(
        "📋 Navigate to:",
        ["📊 Dashboard", "📈 Aggregation Analysis"],
        key='main_nav'
    )
    
    # Sidebar
    st.sidebar.header("📁 Portfolio Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    # Currency toggle
    st.sidebar.header("💱 Display Currency")
    base_currency = st.sidebar.radio(
        "Choose base currency:",
        ["SGD", "USD"],
        index=0,
        key='base_currency'
    )
    
    # Date filter
    st.sidebar.header("📅 Filters")
    filter_date = st.sidebar.date_input("As of Date", value=datetime.now().date())
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading portfolio data..."):
            transactions_df = tracker.load_csv_data(uploaded_file)
            holdings_df = tracker.calculate_holdings(transactions_df)
        
        if not holdings_df.empty:
            # Calculate metrics with selected currency
            with st.spinner(f"Fetching market data and calculating returns in {base_currency}..."):
                metrics = tracker.calculate_portfolio_metrics(holdings_df, transactions_df, base_currency)
                benchmark_data = tracker.fetch_benchmark_data()
            
            if metrics:
                portfolio_df = metrics['portfolio_df']
                currency_symbol = 'S$' if base_currency == 'SGD' else '$'
                
                if page == "📊 Dashboard":
                    # Group by options
                    group_by_options = ['Country', 'Sector', 'Currency', 'Do not group']
                    group_by = st.sidebar.selectbox("Group by", group_by_options, index=0)
                    
                    # Sorting options for holdings table
                    st.sidebar.header("📊 Table Settings")
                    sort_options = {
                        'Portfolio Value': 'current_value',
                        'Gain/Loss Amount': 'gain_loss',
                        'Gain/Loss %': 'gain_loss_pct',
                        'Weight %': 'weight_pct',
                        'Symbol': 'symbol'
                    }
                    
                    sort_by_display = st.sidebar.selectbox(
                        "Sort holdings by:",
                        list(sort_options.keys()),
                        index=0,
                        key='sort_by'
                    )
                    
                    sort_order = st.sidebar.radio(
                        "Sort order:",
                        ["Descending (High to Low)", "Ascending (Low to High)"],
                        index=0,
                        key='sort_order'
                    )
                    
                    sort_column = sort_options[sort_by_display]
                    ascending = sort_order == "Ascending (Low to High)"
                    
                    # Main metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Value", f"{currency_symbol}{metrics['total_value']:,.2f}")
                    
                    with col2:
                        gain_loss_color = "normal" if metrics['total_gain_loss'] >= 0 else "inverse"
                        st.metric("Total Gain/Loss", 
                                 f"{currency_symbol}{metrics['total_gain_loss']:+,.2f}",
                                 f"{metrics['total_gain_loss_pct']:+.2f}%",
                                 delta_color=gain_loss_color)
                    
                    with col3:
                        st.metric("Holdings", metrics['num_holdings'])
                    
                    with col4:
                        st.metric("Countries", portfolio_df['country'].nunique())
                    
                    # Exchange rates info
                    with st.expander(f"💱 Current Exchange Rates to {base_currency}"):
                        if base_currency == 'SGD':
                            rates_data = metrics['sgd_rates']
                        else:
                            rates_data = metrics['usd_rates']
                        
                        rates_df = pd.DataFrame(list(rates_data.items()), columns=['Currency', f'Rate to {base_currency}'])
                        rates_df[f'Rate to {base_currency}'] = rates_df[f'Rate to {base_currency}'].round(4)
                        st.dataframe(rates_df, hide_index=True)
                    
                    # Create two-column layout
                    left_col, right_col = st.columns([1, 1])
                    
                    with left_col:
                        # Portfolio allocation pie chart
                        st.subheader("Portfolio Allocation")
                        
                        group_mapping = {
                            'Country': 'country',
                            'Sector': 'sector', 
                            'Currency': 'currency',
                            'Do not group': 'symbol'
                        }
                        
                        group_field = group_mapping.get(group_by, 'country')
                        pie_chart = create_allocation_pie_chart(portfolio_df, group_field)
                        st.plotly_chart(pie_chart, use_container_width=True)
                        
                        # Holdings table with sorting
                        st.subheader(f"All Holdings (Sorted by {sort_by_display})")
                        holdings_table = create_holdings_table(portfolio_df, base_currency, sort_column, ascending)
                        if not holdings_table.empty:
                            # Add styling for positive/negative values
                            def highlight_performance(val):
                                if isinstance(val, str) and '%' in val:
                                    try:
                                        num_val = float(val.replace('%', '').replace('+', ''))
                                        if num_val > 0:
                                            return 'color: #28a745; font-weight: bold'
                                        elif num_val < 0:
                                            return 'color: #dc3545; font-weight: bold'
                                    except:
                                        pass
                                elif isinstance(val, str) and (currency_symbol in val):
                                    if '+' in val:
                                        return 'color: #28a745; font-weight: bold'
                                    elif val.count('-') > val.count(currency_symbol):
                                        return 'color: #dc3545; font-weight: bold'
                                return ''
                            
                            styled_table = holdings_table.style.applymap(highlight_performance)
                            st.dataframe(styled_table, use_container_width=True, hide_index=True)
                    
                    with right_col:
                        # Benchmark comparison
                        st.subheader("Total Return vs Benchmarks")
                        
                        if benchmark_data:
                            comparison_df = create_benchmark_comparison_table(
                                metrics['portfolio_returns'], benchmark_data
                            )
                            
                            # Style the comparison table
                            def highlight_performance(val):
                                if '%' in str(val) and val != 'Benchmark':
                                    try:
                                        num_val = float(val.replace('%', '').replace('+', ''))
                                        if num_val > 0:
                                            return 'color: #28a745; font-weight: bold'
                                        elif num_val < 0:
                                            return 'color: #dc3545; font-weight: bold'
                                    except:
                                        pass
                                return ''
                            
                            styled_comparison = comparison_df.style.applymap(highlight_performance)
                            st.dataframe(styled_comparison, use_container_width=True, hide_index=True)
                        
                        # Historical performance chart
                        st.subheader("Historical Return")
                        
                        if not portfolio_df.empty:
                            periods = ['1M', '3M', '6M', 'YTD', '1Y']
                            returns = [metrics['portfolio_returns'].get(p, 0) for p in periods]
                            
                            fig = go.Figure()
                            colors = ['green' if x >= 0 else 'red' for x in returns]
                            
                            fig.add_trace(go.Bar(
                                x=periods,
                                y=returns,
                                marker_color=colors,
                                text=[f"{x:.1f}%" for x in returns],
                                textposition='outside'
                            ))
                            
                            fig.update_layout(
                                title="Portfolio Returns by Period",
                                yaxis_title="Return (%)",
                                xaxis_title="Period",
                                showlegend=False,
                                yaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=2)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional analysis tabs
                    st.header("Detailed Analysis")
                    
                    tab1, tab2, tab3 = st.tabs(["📊 Performance", "🌍 Geography", "🏢 Sectors"])
                    
                    with tab1:
                        # Detailed performance metrics
                        st.subheader("Individual Holdings Performance")
                        
                        perf_df = portfolio_df[['symbol', 'long_name', 'current_value', 'weight_pct',
                                               'returns_1M', 'returns_3M', 'returns_6M', 'returns_YTD', 'returns_1Y']].copy()
                        
                        # Format performance data
                        for col in ['returns_1M', 'returns_3M', 'returns_6M', 'returns_YTD', 'returns_1Y']:
                            perf_df[col] = perf_df[col].apply(lambda x: f"{x:+.2f}%")
                        
                        perf_df['current_value'] = perf_df['current_value'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                        perf_df['weight_pct'] = perf_df['weight_pct'].apply(lambda x: f"{x:.2f}%")
                        
                        perf_df.columns = ['Symbol', 'Name', f'Value ({base_currency})', 'Weight', '1M', '3M', '6M', 'YTD', '1Y']
                        
                        st.dataframe(perf_df, use_container_width=True, hide_index=True)
                    
                    with tab2:
                        # Geographic analysis
                        country_summary = portfolio_df.groupby('country').agg({
                            'current_value': 'sum',
                            'gain_loss': 'sum',
                            'weight_pct': 'sum'
                        }).round(2)
                        
                        country_summary['gain_loss_pct'] = (country_summary['gain_loss'] / 
                                                           (country_summary['current_value'] - country_summary['gain_loss']) * 100)
                        
                        # Format columns
                        country_summary['current_value'] = country_summary['current_value'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                        country_summary['gain_loss'] = country_summary['gain_loss'].apply(lambda x: f"{currency_symbol}{x:+,.2f}")
                        country_summary['weight_pct'] = country_summary['weight_pct'].apply(lambda x: f"{x:.2f}%")
                        country_summary['gain_loss_pct'] = country_summary['gain_loss_pct'].apply(lambda x: f"{x:+.2f}%")
                        
                        country_summary.columns = [f'Total Value ({base_currency})', f'Total Gain/Loss ({base_currency})', 
                                                  'Weight %', 'Return %']
                        
                        st.dataframe(country_summary, use_container_width=True)
                    
                    with tab3:
                        # Sector analysis
                        sector_summary = portfolio_df.groupby('sector').agg({
                            'current_value': 'sum',
                            'gain_loss': 'sum',
                            'weight_pct': 'sum'
                        }).round(2)
                        
                        # Format columns
                        sector_summary['current_value'] = sector_summary['current_value'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                        sector_summary['gain_loss'] = sector_summary['gain_loss'].apply(lambda x: f"{currency_symbol}{x:+,.2f}")
                        sector_summary['weight_pct'] = sector_summary['weight_pct'].apply(lambda x: f"{x:.2f}%")
                        
                        sector_summary.columns = [f'Total Value ({base_currency})', f'Total Gain/Loss ({base_currency})', 'Weight %']
                        
                        st.dataframe(sector_summary, use_container_width=True)
                
                elif page == "📈 Aggregation Analysis":
                    # Show aggregation page
                    create_aggregation_page(portfolio_df, base_currency)
                    
            else:
                st.error("Unable to calculate portfolio metrics. Please check your data.")
        else:
            st.warning("No valid holdings found in the uploaded data.")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Upload your portfolio CSV file to get started")
        
        st.subheader("📋 Expected CSV Format")
        sample_data = {
            'Date': ['2024-01-15', '2024-01-20', '2024-02-01'],
            'Ticker': ['AAPL', 'N2IU', 'MSFT'],
            'Action': ['Buy', 'Buy', 'Buy'],
            'Shares': [100, 1000, 50],
            'Price': [185.0, 2.85, 410.0],
            'Commission': [1.0, 25.0, 1.0],
            'Currency': ['USD', 'SGD', 'USD'],
            'Note': ['Tech stock', 'SG tech', 'Cloud software']
        }
        st.dataframe(pd.DataFrame(sample_data), hide_index=True)
        
        st.subheader("🌟 Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Dashboard Features:**
            - Real-time portfolio valuation
            - Multi-currency support (SGD/USD)
            - Sortable holdings table
            - Performance vs benchmarks
            - Geographic & sector allocation
            - Gain/loss tracking with color coding
            """)
        
        with col2:
            st.markdown("""
            **📈 Aggregation Analysis:**
            - Group by Country, Sector, Currency, Symbol
            - Multiple chart types (Bar, Pie, Treemap)
            - Interactive filtering and drill-down
            - Summary statistics
            - Customizable metrics display
            """)
        
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Upload CSV**: Use the file uploader in the sidebar
        2. **Choose Currency**: Toggle between SGD and USD display
        3. **Explore Dashboard**: View real-time portfolio metrics and charts
        4. **Analyze Data**: Use the Aggregation Analysis page for detailed insights
        5. **Sort & Filter**: Customize how your holdings are displayed
        """)

if __name__ == "__main__":
    main()
