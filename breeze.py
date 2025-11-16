"""
Kite Connect F&O Trading Dashboard
Streamlit application for trading with Zerodha Kite Connect API
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from kiteconnect import KiteConnect
import time

# Page config
st.set_page_config(
    page_title="F&O Dashboard - Kite Connect",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------
# Configuration
# --------------------------
API_KEY = "aj0gv6rpjm11ecac"
ACCESS_TOKEN = "SmCnbRkg9WhWv7FnF3cXpjEGBJkWqihw"

# --------------------------
# FNO Stocks List
# --------------------------
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "ITC", "SBIN", "HCLTECH", "AXISBANK",
    "KOTAKBANK", "LT", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO", "TATAMOTORS",
    "ADANIPORTS", "ADANIENT", "TECHM", "POWERGRID", "NTPC",
    "COALINDIA", "TATASTEEL", "BAJAJFINSV", "HEROMOTOCO", "INDUSINDBK",
    "M&M", "GRASIM", "HINDALCO", "JSWSTEEL", "SBILIFE",
    "ICICIGI", "BAJAJ-AUTO", "HDFCLIFE", "ADANIGREEN", "SHREECEM",
    "EICHERMOT", "UPL", "TATACONSUM", "BRITANNIA", "NESTLEIND",
    "HINDUNILVR", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP"
]

# RSS Feeds
FINANCIAL_RSS_FEEDS = [
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

# --------------------------
# Initialize session state
# --------------------------
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'kite_connected' not in st.session_state:
    st.session_state.kite_connected = False
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'instruments_df' not in st.session_state:
    st.session_state.instruments_df = None

# --------------------------
# Kite Connection
# --------------------------
@st.cache_resource
def init_kite():
    """Initialize Kite Connect"""
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(ACCESS_TOKEN)
        
        # Test connection
        profile = kite.profile()
        
        return kite, True, profile
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None, False, None

if not st.session_state.kite_connected:
    kite, connected, profile = init_kite()
    st.session_state.kite = kite
    st.session_state.kite_connected = connected
    st.session_state.profile = profile

# --------------------------
# Helper Functions
# --------------------------
@st.cache_data(ttl=300)
def get_instruments():
    """Get and cache instruments list"""
    try:
        instruments = st.session_state.kite.instruments("NSE")
        df = pd.DataFrame(instruments)
        return df
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return None

def get_instrument_token(symbol):
    """Get instrument token for a symbol"""
    if st.session_state.instruments_df is None:
        st.session_state.instruments_df = get_instruments()
    
    if st.session_state.instruments_df is not None:
        result = st.session_state.instruments_df[
            st.session_state.instruments_df['tradingsymbol'] == symbol
        ]
        if not result.empty:
            return result.iloc[0]['instrument_token']
    return None

def fetch_historical_data(symbol, days=30, interval="day"):
    """Fetch historical data from Kite"""
    try:
        kite = st.session_state.kite
        if not kite:
            return None
        
        instrument_token = get_instrument_token(symbol)
        if not instrument_token:
            return None
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # For intraday data, adjust the date range
        if interval in ["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute"]:
            # For intraday, get today's data only if it's a trading day
            # Otherwise get last trading day
            if to_date.weekday() >= 5:  # Saturday or Sunday
                days_back = to_date.weekday() - 4  # Go back to Friday
                to_date = to_date - timedelta(days=days_back)
                from_date = to_date.replace(hour=9, minute=0, second=0)
            else:
                from_date = to_date.replace(hour=9, minute=0, second=0)
        
        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if data:
                df = pd.DataFrame(data)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    return df
            
            return None
        except Exception as e:
            # If today's data fails, try yesterday
            if interval in ["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute"]:
                to_date = datetime.now() - timedelta(days=1)
                from_date = to_date.replace(hour=9, minute=0, second=0)
                
                data = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval
                )
                
                if data:
                    df = pd.DataFrame(data)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        return df
            
            return None
        
    except Exception as e:
        return None

# --------------------------
# Technical Indicators
# --------------------------
def calculate_sma(data, period):
    """Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    """Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """RSI Indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD Indicator"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# --------------------------
# Sentiment Analysis
# --------------------------
def analyze_sentiment(text):
    """Keyword-based sentiment analysis"""
    POSITIVE = ['surge', 'rally', 'gain', 'profit', 'growth', 'rise', 'bullish', 
                'strong', 'beats', 'outperform', 'jumps', 'soars', 'upgrade']
    
    NEGATIVE = ['fall', 'drop', 'loss', 'decline', 'weak', 'crash', 'bearish',
                'concern', 'risk', 'plunge', 'slump', 'miss', 'downgrade']
    
    text_lower = text.lower()
    pos_count = sum(1 for w in POSITIVE if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE if w in text_lower)
    
    if pos_count > neg_count:
        return "positive", min(0.6 + pos_count * 0.1, 0.95)
    elif neg_count > pos_count:
        return "negative", min(0.6 + neg_count * 0.1, 0.95)
    else:
        return "neutral", 0.5

def fetch_news(num_articles=12, specific_stock=None):
    """Fetch news articles"""
    all_articles = []
    seen_titles = set()
    
    for feed_url, source_name in FINANCIAL_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = getattr(entry, 'title', '')
                if not title or title in seen_titles:
                    continue
                
                if specific_stock and specific_stock != "All Stocks":
                    if specific_stock.upper() not in title.upper():
                        continue
                
                sentiment, score = analyze_sentiment(title)
                
                all_articles.append({
                    "Title": title,
                    "Source": source_name,
                    "Sentiment": sentiment,
                    "Score": score,
                    "Link": entry.link,
                    "Published": getattr(entry, 'published', 'Recent')
                })
                seen_titles.add(title)
                
                if len(all_articles) >= num_articles:
                    break
        except:
            continue
        
        if len(all_articles) >= num_articles:
            break
    
    return all_articles[:num_articles]

def get_live_quotes(symbols):
    """Get live quotes for symbols"""
    try:
        kite = st.session_state.kite
        if not kite:
            return None
        
        # Format symbols for Kite
        formatted_symbols = [f"NSE:{symbol}" for symbol in symbols]
        quotes = kite.quote(formatted_symbols)
        return quotes
    except Exception as e:
        return None

# --------------------------
# Streamlit App
# --------------------------

st.title("📈 F&O Dashboard - Kite Connect")

# Connection Status
if st.session_state.kite_connected:
    profile = st.session_state.profile
    st.success(f"✅ Connected to Kite API | User: {profile.get('user_name', 'N/A')}")
else:
    st.error("❌ Not connected to Kite API")
    st.info("Please check your API key and access token")
    st.stop()

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📰 News", "💹 Charts & Indicators", "⚡ Intraday Monitor", "📊 Portfolio"])

# TAB 1: NEWS
with tab1:
    st.header("Market News & Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stock_filter = st.selectbox(
            "Filter by Stock",
            ["All Stocks"] + FNO_STOCKS,
            key="news_filter"
        )
    
    with col2:
        if st.button("🔄 Refresh News", key="refresh_news"):
            st.session_state.news_articles = fetch_news(12, stock_filter)
            st.success("News refreshed!")
    
    if not st.session_state.news_articles:
        with st.spinner("Loading news..."):
            st.session_state.news_articles = fetch_news(12, stock_filter)
    
    if st.session_state.news_articles:
        df_news = pd.DataFrame(st.session_state.news_articles)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", len(df_news))
        with col2:
            st.metric("🟢 Positive", len(df_news[df_news['Sentiment'] == 'positive']))
        with col3:
            st.metric("⚪ Neutral", len(df_news[df_news['Sentiment'] == 'neutral']))
        with col4:
            st.metric("🔴 Negative", len(df_news[df_news['Sentiment'] == 'negative']))
        
        st.markdown("---")
        
        # Display articles
        for article in st.session_state.news_articles:
            sentiment_colors = {"positive": "#28a745", "neutral": "#6c757d", "negative": "#dc3545"}
            sentiment_emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
            
            st.markdown(f"**[{article['Title']}]({article['Link']})**")
            st.markdown(
                f"<span style='background-color: {sentiment_colors[article['Sentiment']]}; "
                f"color: white; padding: 3px 10px; border-radius: 4px; font-size: 11px;'>"
                f"{sentiment_emoji[article['Sentiment']]} {article['Sentiment'].upper()}</span>",
                unsafe_allow_html=True
            )
            st.caption(f"Source: {article['Source']} | {article['Published']}")
            st.markdown("---")

# TAB 2: CHARTS WITH INDICATORS
with tab2:
    st.header("Stock Charts with Technical Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stock = st.selectbox(
            "Select Stock",
            FNO_STOCKS,
            key="chart_stock"
        )
    
    with col2:
        period = st.selectbox(
            "Period",
            ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months"],
            key="chart_period"
        )
        days_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90, "6 Months": 180}
        days = days_map[period]
    
    with col3:
        interval = st.selectbox(
            "Interval",
            ["day", "60minute", "30minute", "15minute", "5minute"],
            format_func=lambda x: {"day": "Daily", "60minute": "60 Min", "30minute": "30 Min", 
                                   "15minute": "15 Min", "5minute": "5 Min"}[x],
            key="chart_interval"
        )
    
    # Intraday data limited to fewer days
    if interval != "day" and days > 30:
        st.info("ℹ️ Intraday data limited to 30 days")
        days = 30
    
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = fetch_historical_data(selected_stock, days, interval)
    
    if df is not None and not df.empty and len(df) > 0:
        # Calculate indicators
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['SMA_50'] = calculate_sma(df['close'], 50)
        df['EMA_12'] = calculate_ema(df['close'], 12)
        df['EMA_26'] = calculate_ema(df['close'], 26)
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['close'])
        
        # Metrics
        current = df['close'].iloc[-1]
        prev = df['close'].iloc[0]
        change = current - prev
        change_pct = (change / prev) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"₹{current:.2f}")
        with col2:
            st.metric("Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
        with col3:
            st.metric("High", f"₹{df['high'].max():.2f}")
        with col4:
            st.metric("Low", f"₹{df['low'].min():.2f}")
        
        st.markdown("---")
        
        # Candlestick chart with moving averages
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='blue', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='orange', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_12'],
            mode='lines',
            name='EMA 12',
            line=dict(color='green', width=1.5, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_26'],
            mode='lines',
            name='EMA 26',
            line=dict(color='red', width=1.5, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{selected_stock} - Price Chart with SMA & EMA",
            xaxis_title="Date/Time",
            yaxis_title="Price (₹)",
            height=500,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI Chart
        df_rsi = df.dropna(subset=['RSI'])
        if len(df_rsi) > 0:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df_rsi.index,
                y=df_rsi['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(
                title="RSI Indicator",
                height=250,
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        # MACD Chart
        df_macd = df.dropna(subset=['MACD', 'MACD_Signal'])
        if len(df_macd) > 0:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df_macd.index,
                y=df_macd['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ))
            fig_macd.add_trace(go.Scatter(
                x=df_macd.index,
                y=df_macd['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ))
            fig_macd.add_trace(go.Bar(
                x=df_macd.index,
                y=df_macd['MACD_Hist'],
                name='Histogram',
                marker_color='gray'
            ))
            fig_macd.update_layout(
                title="MACD Indicator",
                height=250
            )
            st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.error(f"Could not fetch data for {selected_stock}")

# TAB 3: INTRADAY MONITOR
with tab3:
    st.header("⚡ Intraday Multi-Stock Monitor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        watchlist = st.multiselect(
            "Select Stocks (max 6)",
            FNO_STOCKS,
            default=["RELIANCE", "TCS", "HDFCBANK", "INFY"],
            max_selections=6,
            key="intraday_stocks"
        )
    
    with col2:
        intraday_interval = st.selectbox(
            "Interval",
            ["5minute", "15minute", "30minute", "60minute"],
            format_func=lambda x: {"5minute": "5 Min", "15minute": "15 Min", 
                                   "30minute": "30 Min", "60minute": "60 Min"}[x],
            key="intraday_interval"
        )
    
    with col3:
        if st.button("🔄 Refresh", key="intraday_refresh"):
            st.rerun()
    
    if watchlist:
        # Try to get live quotes first
        live_quotes = get_live_quotes(watchlist)
        
        # Display current market status
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if market_open <= now <= market_close and now.weekday() < 5:
            st.info("🟢 Market is OPEN - Showing live intraday data")
        else:
            st.warning("⚪ Market is CLOSED - Showing last trading day data")
        
        st.markdown("---")
        
        num_cols = 2 if len(watchlist) <= 4 else 3
        num_rows = (len(watchlist) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, col in enumerate(cols):
                stock_idx = row * num_cols + col_idx
                
                if stock_idx < len(watchlist):
                    stock_symbol = watchlist[stock_idx]
                    
                    with col:
                        # Try to get live quote first
                        if live_quotes and f"NSE:{stock_symbol}" in live_quotes:
                            quote_data = live_quotes[f"NSE:{stock_symbol}"]
                            current = quote_data.get('last_price', 0)
                            prev = quote_data.get('ohlc', {}).get('close', current)
                            change = current - prev
                            change_pct = (change / prev * 100) if prev > 0 else 0
                            arrow = "🟢" if change_pct >= 0 else "🔴"
                            
                            st.markdown(f"### {arrow} {stock_symbol}")
                            st.metric("LTP", f"₹{current:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                        
                        # Fetch historical intraday data for chart
                        with st.spinner(f"Loading {stock_symbol}..."):
                            df = fetch_historical_data(stock_symbol, 1, intraday_interval)
                        
                        if df is not None and not df.empty and len(df) >= 2:
                            # If we don't have live quote, use historical data
                            if not live_quotes or f"NSE:{stock_symbol}" not in live_quotes:
                                current = df['close'].iloc[-1]
                                prev = df['close'].iloc[0]
                                change = current - prev
                                change_pct = (change / prev) * 100
                                arrow = "🟢" if change_pct >= 0 else "🔴"
                                
                                st.markdown(f"### {arrow} {stock_symbol}")
                                st.metric("Price", f"₹{current:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                            
                            # Mini candlestick chart
                            fig = go.Figure(data=[go.Candlestick(
                                x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name=stock_symbol
                            )])
                            fig.update_layout(
                                height=250,
                                margin=dict(l=10, r=10, t=10, b=10),
                                showlegend=False,
                                xaxis_rangeslider_visible=False
                            )
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                            
                            # Show data range
                            st.caption(f"📊 {len(df)} candles | {df.index[0].strftime('%H:%M')} - {df.index[-1].strftime('%H:%M')}")
                        else:
                            st.warning(f"No intraday data available for {stock_symbol}")
                            st.caption("Try: Market hours (9:15 AM - 3:30 PM) on trading days")
    else:
        st.info("👆 Select stocks to monitor")

# TAB 4: PORTFOLIO
with tab4:
    st.header("📊 Your Portfolio")
    
    try:
        kite = st.session_state.kite
        
        # Holdings
        st.subheader("Holdings")
        holdings = kite.holdings()
        
        if holdings:
            df_holdings = pd.DataFrame(holdings)
            
            # Calculate totals
            total_investment = sum(h.get('average_price', 0) * h.get('quantity', 0) for h in holdings)
            total_current = sum(h.get('last_price', 0) * h.get('quantity', 0) for h in holdings)
            total_pnl = total_current - total_investment
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Holdings", len(holdings))
            with col2:
                st.metric("Investment", f"₹{total_investment:,.2f}")
            with col3:
                st.metric("Current Value", f"₹{total_current:,.2f}")
            with col4:
                pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
                st.metric("P&L", f"₹{total_pnl:,.2f}", f"{pnl_pct:.2f}%")
            
            # Display holdings table
            display_cols = ['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']
            if all(col in df_holdings.columns for col in display_cols):
                st.dataframe(df_holdings[display_cols], use_container_width=True)
        else:
            st.info("No holdings found")
        
        st.markdown("---")
        
        # Positions
        st.subheader("Positions")
        positions = kite.positions()
        
        if positions:
            net_positions = positions.get('net', [])
            if net_positions:
                df_positions = pd.DataFrame(net_positions)
                st.dataframe(df_positions, use_container_width=True)
            else:
                st.info("No open positions")
        
        st.markdown("---")
        
        # Orders
        st.subheader("Today's Orders")
        orders = kite.orders()
        
        if orders:
            df_orders = pd.DataFrame(orders)
            display_cols = ['order_timestamp', 'tradingsymbol', 'transaction_type', 
                           'quantity', 'price', 'status']
            available_cols = [col for col in display_cols if col in df_orders.columns]
            if available_cols:
                st.dataframe(df_orders[available_cols], use_container_width=True)
        else:
            st.info("No orders today")
        
    except Exception as e:
        st.error(f"Error fetching portfolio data: {e}")

# Footer
st.markdown("---")
st.caption("💡 F&O Dashboard powered by Zerodha Kite Connect API")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
