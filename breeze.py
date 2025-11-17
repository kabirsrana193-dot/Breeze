import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import feedparser
from kiteconnect import KiteConnect, KiteTicker
import time
import threading
import pytz

# Page config
st.set_page_config(
    page_title="F&O Dashboard - Kite Connect (Live)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------
# Configuration
# --------------------------
API_KEY = "aj0gv6rpjm11ecac"
ACCESS_TOKEN = "Xm8xsdCJZCREksR68JTnk5JWMXWtWnUG"

# Indian market hours
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"
IST = pytz.timezone('Asia/Kolkata')

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
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'ticker_active' not in st.session_state:
    st.session_state.ticker_active = False
if 'kws' not in st.session_state:
    st.session_state.kws = None

# --------------------------
# Kite Connection
# --------------------------
@st.cache_resource
def init_kite():
    """Initialize Kite Connect"""
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(ACCESS_TOKEN)
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

def get_instrument_tokens(symbols):
    """Get instrument tokens for multiple symbols"""
    tokens = {}
    for symbol in symbols:
        token = get_instrument_token(symbol)
        if token:
            tokens[symbol] = token
    return tokens

def filter_market_hours(df):
    """
    Filter dataframe to only include market hours (9:15 AM - 3:30 PM IST)
    Removes after-hours and pre-market data to avoid gaps in charts
    """
    if df is None or df.empty:
        return df
    
    try:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Convert to IST if not already
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        elif df.index.tz != IST:
            df.index = df.index.tz_convert(IST)
        
        # Filter by time (9:15 AM to 3:30 PM)
        df_filtered = df.between_time('09:15', '15:30')
        
        return df_filtered
    except Exception as e:
        st.warning(f"Time filtering error: {e}")
        return df

def fetch_historical_data(symbol, days=30, interval="day"):
    """Fetch historical data from Kite with market hours filtering"""
    try:
        kite = st.session_state.kite
        if not kite:
            return None
        
        instrument_token = get_instrument_token(symbol)
        if not instrument_token:
            return None
        
        to_date = datetime.now(IST)
        
        # Adjust date range based on interval
        if interval in ["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute"]:
            # For intraday, limit to recent days
            if days > 30:
                days = 30
            
            # If weekend, go back to last trading day
            if to_date.weekday() >= 5:  # Saturday or Sunday
                days_back = to_date.weekday() - 4
                to_date = to_date - timedelta(days=days_back)
            
            # Set to market close time
            to_date = to_date.replace(hour=15, minute=30, second=0, microsecond=0)
            from_date = to_date - timedelta(days=days)
            from_date = from_date.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            # For daily data
            from_date = to_date - timedelta(days=days)
        
        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date.replace(tzinfo=None),
                to_date=to_date.replace(tzinfo=None),
                interval=interval
            )
            
            if data:
                df = pd.DataFrame(data)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    
                    # Filter to market hours for intraday data
                    if interval != "day":
                        df = filter_market_hours(df)
                    
                    # Remove any rows with all zeros (no trading)
                    df = df[(df[['open', 'high', 'low', 'close']] != 0).all(axis=1)]
                    
                    return df
            return None
        except Exception as e:
            st.warning(f"Data fetch error: {e}")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --------------------------
# WebSocket Functions
# --------------------------
def start_websocket(symbols):
    """Start WebSocket connection for live data"""
    try:
        tokens_map = get_instrument_tokens(symbols)
        if not tokens_map:
            st.error("Could not get instrument tokens")
            return
        
        tokens = list(tokens_map.values())
        kws = KiteTicker(API_KEY, ACCESS_TOKEN)
        
        def on_ticks(ws, ticks):
            for tick in ticks:
                token = tick['instrument_token']
                symbol = None
                for sym, tok in tokens_map.items():
                    if tok == token:
                        symbol = sym
                        break
                
                if symbol:
                    st.session_state.live_data[symbol] = {
                        'ltp': tick.get('last_price', 0),
                        'change': tick.get('change', 0),
                        'volume': tick.get('volume', 0),
                        'oi': tick.get('oi', 0),
                        'timestamp': datetime.now(IST),
                        'ohlc': tick.get('ohlc', {}),
                        'high': tick.get('ohlc', {}).get('high', 0),
                        'low': tick.get('ohlc', {}).get('low', 0),
                        'open': tick.get('ohlc', {}).get('open', 0),
                        'close': tick.get('ohlc', {}).get('close', 0)
                    }
        
        def on_connect(ws, response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            st.session_state.ticker_active = True
        
        def on_close(ws, code, reason):
            st.session_state.ticker_active = False
        
        def on_error(ws, code, reason):
            st.error(f"WebSocket Error: {reason}")
        
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error
        
        st.session_state.kws = kws
        
        def run_websocket():
            kws.connect(threaded=True)
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        return True
    except Exception as e:
        st.error(f"WebSocket Error: {e}")
        return False

def stop_websocket():
    """Stop WebSocket connection"""
    try:
        if st.session_state.kws:
            st.session_state.kws.close()
            st.session_state.ticker_active = False
            st.session_state.kws = None
    except:
        pass

# --------------------------
# Technical Indicators
# --------------------------
def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    if len(data) < period:
        return pd.Series([None] * len(data), index=data.index)
    return data.rolling(window=period, min_periods=1).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return pd.Series([None] * len(data), index=data.index)
    return data.ewm(span=period, adjust=False, min_periods=1).mean()

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate Supertrend"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate basic upper and lower bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize supertrend
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                
            if direction.iloc[i] == 1 and supertrend.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
            if direction.iloc[i] == -1 and supertrend.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
    
    return supertrend, direction

# --------------------------
# Sentiment Analysis
# --------------------------
def analyze_sentiment(text):
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

# --------------------------
# Streamlit App
# --------------------------

st.title("📈 F&O Dashboard - Kite Connect (🔴 LIVE)")

# Connection Status
if st.session_state.kite_connected:
    profile = st.session_state.profile
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ Connected to Kite API | User: {profile.get('user_name', 'N/A')}")
    with col2:
        if st.session_state.ticker_active:
            st.success("🔴 WebSocket LIVE")
        else:
            st.warning("⚪ WebSocket OFF")
else:
    st.error("❌ Not connected to Kite API")
    st.stop()

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📰 News", "💹 Charts & Indicators", "🔴 LIVE Monitor", "📊 Portfolio"])

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
        
        for article in st.session_state.news_articles:
            sentiment_colors = {"positive": "#28a745", "neutral": "#6c757d", "negative": "#dc3545"}
            sentiment_emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
            
            st.markdown(f"[{article['Title']}]({article['Link']})")
            st.markdown(
                f"<span style='background-color: {sentiment_colors[article['Sentiment']]}; "
                f"color: white; padding: 3px 10px; border-radius: 4px; font-size: 11px;'>"
                f"{sentiment_emoji[article['Sentiment']]} {article['Sentiment'].upper()}</span>",
                unsafe_allow_html=True
            )
            st.caption(f"Source: {article['Source']} | {article['Published']}")
            st.markdown("---")

# TAB 2: CHARTS - COMBINED MA CHART
with tab2:
    st.header("Stock Charts with Technical Indicators")
    st.caption("📊 EMA: 9, 21, 50 | SMA: 20, 50, 200 | BB: 20,2 | Supertrend: 10,3 | Market Hours: 9:15 AM - 3:30 PM IST")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stock = st.selectbox("Select Stock", FNO_STOCKS, key="chart_stock")
    
    with col2:
        period = st.selectbox("Period", ["1 Week", "2 Weeks", "1 Month", "3 Months"], key="chart_period")
        days_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90}
        days = days_map[period]
    
    with col3:
        interval = st.selectbox(
            "Interval",
            ["day", "60minute", "30minute", "15minute", "5minute"],
            format_func=lambda x: {
                "day": "Daily", 
                "60minute": "60 Min", 
                "30minute": "30 Min", 
                "15minute": "15 Min",
                "5minute": "5 Min"
            }[x],
            key="chart_interval"
        )
    
    # Limit days for intraday intervals
    if interval != "day" and days > 30:
        days = 7
        st.info("📌 Intraday data limited to 1 week for better performance")
    
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = fetch_historical_data(selected_stock, days, interval)
    
    if df is not None and not df.empty:
        # Calculate all indicators
        df['EMA_9'] = calculate_ema(df['close'], 9)
        df['EMA_21'] = calculate_ema(df['close'], 21)
        df['EMA_50'] = calculate_ema(df['close'], 50)
        
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['SMA_50'] = calculate_sma(df['close'], 50)
        df['SMA_200'] = calculate_sma(df['close'], 200)
        
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['close'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['close'])
        df['Supertrend'], df['ST_direction'] = calculate_supertrend(df)
        
        # Current metrics
        current = df['close'].iloc[-1]
        prev = df['close'].iloc[0]
        change = current - prev
        change_pct = (change / prev) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current", f"₹{current:.2f}")
        with col2:
            st.metric("Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
        with col3:
            st.metric("High", f"₹{df['high'].max():.2f}")
        with col4:
            st.metric("Low", f"₹{df['low'].min():.2f}")
        
        st.markdown("---")
        
        # 1. CANDLESTICK CHART WITH EMA & SMA - BIGGER SIZE
        st.subheader(f"📊 {selected_stock} - Price Chart with Moving Averages")
        
        # Format datetime index as strings to remove gaps
        if interval != 'day':
            df_plot = df.copy()
            df_plot.index = df_plot.index.strftime('%d %b %H:%M')
            x_data = df_plot.index
            xaxis_type = 'category'
            tickformat = None
        else:
            df_plot = df.copy()
            x_data = df.index
            xaxis_type = 'date'
            tickformat = '%d %b %Y'
        
        fig_candle = go.Figure()
        
        # Candlesticks
        fig_candle.add_trace(go.Candlestick(
            x=x_data,
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # EMA Lines
        fig_candle.add_trace(go.Scatter(
            x=x_data, y=df_plot['EMA_9'],
            name='EMA 9',
            line=dict(color='#4CAF50', width=1.5),
            mode='lines'
        ))
        
        fig_candle.add_trace(go.Scatter(
            x=x_data, y=df_plot['EMA_21'],
            name='EMA 21',
            line=dict(color='#FF9800', width=1.5),
            mode='lines'
        ))
        
        fig_candle.add_trace(go.Scatter(
            x=x_data, y=df_plot['EMA_50'],
            name='EMA 50',
            line=dict(color='#9C27B0', width=1.5),
            mode='lines'
        ))
        
        # SMA Lines
        fig_candle.add_trace(go.Scatter(
            x=x_data, y=df_plot['SMA_20'],
            name='SMA 20',
            line=dict(color='#FF5722', width=1.5, dash='dash'),
            mode='lines'
        ))
        
        fig_candle.add_trace(go.Scatter(
            x=x_data, y=df_plot['SMA_50'],
            name='SMA 50',
            line=dict(color='#FFC107', width=1.5, dash='dash'),
            mode='lines'
        ))
        
        # Only show SMA 200 if we have enough data
        if len(df_plot) >= 200:
            fig_candle.add_trace(go.Scatter(
                x=x_data, y=df_plot['SMA_200'],
                name='SMA 200',
                line=dict(color='#795548', width=2, dash='dash'),
                mode='lines'
            ))
        
        fig_candle.update_layout(
            title=f"{selected_stock} - {interval.upper()} Chart with Moving Averages",
            yaxis_title="Price (₹)",
            xaxis_title="Time (IST)",
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            xaxis=dict(
                type=xaxis_type,
                tickformat=tickformat,
                tickangle=-45,
                nticks=15,
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            )
        )
        
        st.plotly_chart(fig_candle, use_container_width=True)
        st.info("💡 **Tip:** Click legend items to show/hide individual moving averages. EMA lines are solid, SMA lines are dashed.")
        
        # 2. BOLLINGER BANDS
        st.subheader("📊 Bollinger Bands (20, 2)")
        
        fig_bb = go.Figure()
        
        fig_bb.add_trace(go.Scatter(
            x=x_data, y=df_plot['close'],
            name='Close Price',
            line=dict(color='#FFC107', width=2),
            mode='lines',
            visible=True
        ))
        
        fig_bb.update_layout(
            title="Bollinger Bands Analysis",
            yaxis_title="Price (₹)",
            xaxis_title="Time (IST)",
            height=450,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            xaxis=dict(
                type=xaxis_type,
                tickformat=tickformat,
                tickangle=-45,
                nticks=15
            ),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )
        
        st.plotly_chart(fig_bb, use_container_width=True)
        st.info("💡 **Tip:** Click legend items to show/hide bands. Price touching upper band = potential resistance, lower band = potential support.")
        
        # 3. SUPERTREND
        st.subheader("🎯 Supertrend (10, 3)")
        
        fig_st = go.Figure()
        
        # Price candlesticks
        fig_st.add_trace(go.Candlestick(
            x=x_data,
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            showlegend=False
        ))
        
        # Supertrend line with color based on direction
        if 'ST_direction' in df_plot.columns:
            i = 0
            while i < len(df_plot):
                if pd.notna(df_plot['ST_direction'].iloc[i]) and df_plot['ST_direction'].iloc[i] == 1:
                    # Find continuous green segment
                    start_idx = i
                    while i < len(df_plot) and pd.notna(df_plot['ST_direction'].iloc[i]) and df_plot['ST_direction'].iloc[i] == 1:
                        i += 1
                    end_idx = i
                    
                    # Plot green segment
                    fig_st.add_trace(go.Scatter(
                        x=x_data[start_idx:end_idx],
                        y=df_plot['Supertrend'].iloc[start_idx:end_idx],
                        name='Buy Signal',
                        line=dict(color='#4CAF50', width=2),
                        mode='lines',
                        showlegend=(start_idx == 0)
                    ))
                elif pd.notna(df_plot['ST_direction'].iloc[i]) and df_plot['ST_direction'].iloc[i] == -1:
                    # Find continuous red segment
                    start_idx = i
                    while i < len(df_plot) and pd.notna(df_plot['ST_direction'].iloc[i]) and df_plot['ST_direction'].iloc[i] == -1:
                        i += 1
                    end_idx = i
                    
                    # Plot red segment
                    fig_st.add_trace(go.Scatter(
                        x=x_data[start_idx:end_idx],
                        y=df_plot['Supertrend'].iloc[start_idx:end_idx],
                        name='Sell Signal',
                        line=dict(color='#ef5350', width=2),
                        mode='lines',
                        showlegend=(start_idx == 0)
                    ))
                else:
                    i += 1
        
        fig_st.update_layout(
            title="Supertrend Indicator",
            yaxis_title="Price (₹)",
            xaxis_title="Time (IST)",
            height=450,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                type=xaxis_type,
                tickformat=tickformat,
                tickangle=-45,
                nticks=15,
                rangeslider_visible=False
            ),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )
        
        st.plotly_chart(fig_st, use_container_width=True)
        
        # 4. RSI
        st.subheader("📊 RSI (Relative Strength Index)")
        
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=x_data, y=df_plot['RSI'],
            name='RSI',
            line=dict(color='#9C27B0', width=2),
            fill='tozeroy',
            fillcolor='rgba(156, 39, 176, 0.1)',
            visible=True
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought (70)", annotation_position="right")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                         annotation_text="Oversold (30)", annotation_position="right")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray",
                         annotation_text="Neutral (50)", annotation_position="right")
        
        fig_rsi.update_layout(
            title="RSI Indicator",
            yaxis_title="RSI Value",
            xaxis_title="Time (IST)",
            height=300,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                type=xaxis_type,
                tickformat=tickformat,
                tickangle=-45,
                nticks=12
            ),
            yaxis=dict(range=[0, 100], showgrid=True)
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # 5. MACD
        st.subheader("📈 MACD (Moving Average Convergence Divergence)")
        
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=x_data, y=df_plot['MACD'],
            name='MACD',
            line=dict(color='#2196F3', width=2)
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=x_data, y=df_plot['MACD_signal'],
            name='Signal',
            line=dict(color='#FF5722', width=2)
        ))
        
        # Histogram
        colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df_plot['MACD_hist']]
        fig_macd.add_trace(go.Bar(
            x=x_data, y=df_plot['MACD_hist'],
            name='Histogram',
            marker_color=colors,
            opacity=0.5
        ))
        
        fig_macd.update_layout(
            title="MACD Indicator",
            yaxis_title="MACD Value",
            xaxis_title="Time (IST)",
            height=300,
            hovermode='x unified',
            xaxis=dict(
                type=xaxis_type,
                tickformat=tickformat,
                tickangle=-45,
                nticks=12
            ),
            yaxis=dict(showgrid=True)
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # Trading Signals Summary
        st.markdown("---")
        st.subheader("🎯 Technical Signals Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**📊 Moving Averages**")
            current_price = df['close'].iloc[-1]
            
            signals = []
            if not pd.isna(df['EMA_9'].iloc[-1]) and current_price > df['EMA_9'].iloc[-1]:
                signals.append("✅ Above EMA 9")
            else:
                signals.append("❌ Below EMA 9")
                
            if not pd.isna(df['EMA_21'].iloc[-1]) and current_price > df['EMA_21'].iloc[-1]:
                signals.append("✅ Above EMA 21")
            else:
                signals.append("❌ Below EMA 21")
                
            if not pd.isna(df['SMA_50'].iloc[-1]) and current_price > df['SMA_50'].iloc[-1]:
                signals.append("✅ Above SMA 50")
            else:
                signals.append("❌ Below SMA 50")
            
            for signal in signals:
                st.caption(signal)
        
        with col2:
            st.markdown("**📈 RSI Signal**")
            current_rsi = df['RSI'].iloc[-1]
            if not pd.isna(current_rsi):
                if current_rsi > 70:
                    st.error(f"🔴 Overbought: {current_rsi:.2f}")
                elif current_rsi < 30:
                    st.success(f"🟢 Oversold: {current_rsi:.2f}")
                else:
                    st.info(f"⚪ Neutral: {current_rsi:.2f}")
            else:
                st.caption("Calculating...")
        
        with col3:
            st.markdown("**📊 MACD Signal**")
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['MACD_signal'].iloc[-1]
            
            if not pd.isna(current_macd) and not pd.isna(current_signal):
                if current_macd > current_signal:
                    st.success("🟢 Bullish Crossover")
                else:
                    st.error("🔴 Bearish Crossover")
                
                st.caption(f"MACD: {current_macd:.2f}")
                st.caption(f"Signal: {current_signal:.2f}")
            else:
                st.caption("Calculating...")
        
        with col4:
            st.markdown("**🎯 Bollinger & Supertrend**")
            
            # Bollinger Bands signal
            bb_upper = df['BB_upper'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]
            
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                if current_price > bb_upper:
                    st.caption("🔴 Above BB Upper")
                elif current_price < bb_lower:
                    st.caption("🟢 Below BB Lower")
                else:
                    st.caption("⚪ Within BB Range")
            
            # Supertrend signal
            st_direction = df['ST_direction'].iloc[-1]
            if not pd.isna(st_direction):
                if st_direction == 1:
                    st.success("🟢 Supertrend BUY")
                else:
                    st.error("🔴 Supertrend SELL")
            else:
                st.caption("Calculating...")
    
    else:
        st.error(f"❌ No data available for {selected_stock}")
        st.info("📌 Try selecting a different time period or interval")

# TAB 3: LIVE MONITOR WITH WEBSOCKET
with tab3:
    st.header("🔴 LIVE Intraday Monitor (WebSocket)")
    st.caption("⏰ Market Hours: 9:15 AM - 3:30 PM IST | Live data updates every second")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        watchlist = st.multiselect(
            "Select Stocks (max 6)",
            FNO_STOCKS,
            default=["RELIANCE", "TCS", "HDFCBANK", "INFY"],
            max_selections=6,
            key="live_stocks"
        )
    
    with col2:
        auto_refresh = st.checkbox("Auto Refresh (2s)", value=True)
    
    with col3:
        if st.button("🔴 Start Live Stream", key="start_live"):
            if watchlist:
                stop_websocket()
                if start_websocket(watchlist):
                    st.success("✅ WebSocket Connected!")
                    time.sleep(1)
                    st.rerun()
        
        if st.button("⏹ Stop Stream", key="stop_live"):
            stop_websocket()
            st.info("WebSocket disconnected")
            time.sleep(1)
            st.rerun()
    
    if watchlist and st.session_state.ticker_active:
        st.success(f"🔴 LIVE: Streaming {len(watchlist)} stocks")
        
        # Auto refresh
        if auto_refresh:
            time.sleep(2)
            st.rerun()
        
        # Display live data in grid
        num_cols = 2 if len(watchlist) <= 4 else 3
        num_rows = (len(watchlist) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, col in enumerate(cols):
                stock_idx = row * num_cols + col_idx
                
                if stock_idx < len(watchlist):
                    symbol = watchlist[stock_idx]
                    
                    with col:
                        if symbol in st.session_state.live_data:
                            data = st.session_state.live_data[symbol]
                            ltp = data['ltp']
                            close = data['close']
                            change = ltp - close if close > 0 else 0
                            change_pct = (change / close * 100) if close > 0 else 0
                            arrow = "🟢" if change >= 0 else "🔴"
                            
                            # Card design
                            st.markdown(f"### {arrow} **{symbol}**")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("LTP", f"₹{ltp:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                            with col_b:
                                st.metric("Volume", f"{data['volume']:,}")
                            
                            # OHLC
                            st.caption(f"**O:** ₹{data['open']:.2f} | **H:** ₹{data['high']:.2f} | **L:** ₹{data['low']:.2f} | **C:** ₹{data['close']:.2f}")
                            
                            # Timestamp
                            st.caption(f"⏱ Updated: {data['timestamp'].strftime('%H:%M:%S IST')}")
                        else:
                            st.info(f"⏳ Waiting for {symbol} data...")
        
        # Refresh button at bottom
        st.markdown("---")
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    
    elif watchlist:
        st.info("👆 Click 'Start Live Stream' to begin receiving live data")
        st.warning("⚠ **Important:** WebSocket streams only work during market hours (9:15 AM - 3:30 PM IST)")
        
        # Check if market is open
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if market_open <= now <= market_close and now.weekday() < 5:
            st.success("✅ Market is currently OPEN")
        else:
            st.error("❌ Market is currently CLOSED")
            if now.weekday() >= 5:
                st.info("📅 Market closed on weekends")
            elif now < market_open:
                st.info(f"⏰ Market opens at 9:15 AM IST")
            else:
                st.info(f"⏰ Market closed at 3:30 PM IST")
    else:
        st.info("👆 Select stocks to monitor")

# TAB 4: PORTFOLIO
with tab4:
    st.header("📊 Your Portfolio")
    
    try:
        kite = st.session_state.kite
        
        # Holdings
        st.subheader("💼 Holdings")
        holdings = kite.holdings()
        
        if holdings:
            df_holdings = pd.DataFrame(holdings)
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
                st.metric("Total P&L", f"₹{total_pnl:,.2f}", f"{pnl_pct:.2f}%")
            
            st.markdown("---")
            
            display_cols = ['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']
            if all(col in df_holdings.columns for col in display_cols):
                st.dataframe(
                    df_holdings[display_cols].style.format({
                        'average_price': '₹{:.2f}',
                        'last_price': '₹{:.2f}',
                        'pnl': '₹{:.2f}'
                    }),
                    use_container_width=True,
                    height=400
                )
        else:
            st.info("📭 No holdings found")
        
        st.markdown("---")
        
        # Positions
        st.subheader("📈 Open Positions")
        positions = kite.positions()
        
        if positions and positions.get('net'):
            df_positions = pd.DataFrame(positions['net'])
            
            # Filter out zero quantity positions
            df_positions = df_positions[df_positions['quantity'] != 0]
            
            if not df_positions.empty:
                total_pos_pnl = df_positions['pnl'].sum() if 'pnl' in df_positions.columns else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Open Positions", len(df_positions))
                with col2:
                    st.metric("Positions P&L", f"₹{total_pos_pnl:,.2f}")
                
                st.markdown("---")
                
                display_cols = ['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']
                available_cols = [col for col in display_cols if col in df_positions.columns]
                
                st.dataframe(
                    df_positions[available_cols].style.format({
                        'average_price': '₹{:.2f}',
                        'last_price': '₹{:.2f}',
                        'pnl': '₹{:.2f}'
                    }),
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("📭 No open positions")
        else:
            st.info("📭 No open positions")
        
        st.markdown("---")
        
        # Orders
        st.subheader("📝 Recent Orders")
        orders = kite.orders()
        
        if orders:
            df_orders = pd.DataFrame(orders)
            
            # Show last 10 orders
            df_orders_recent = df_orders.head(10)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Orders (Today)", len(df_orders))
            with col2:
                pending = len(df_orders[df_orders['status'].isin(['OPEN', 'TRIGGER PENDING'])])
                st.metric("Pending Orders", pending)
            
            st.markdown("---")
            
            display_cols = ['order_timestamp', 'tradingsymbol', 'transaction_type', 
                          'quantity', 'price', 'status']
            available_cols = [col for col in display_cols if col in df_orders_recent.columns]
            
            st.dataframe(
                df_orders_recent[available_cols],
                use_container_width=True,
                height=400
            )
        else:
            st.info("📭 No orders found for today")
        
    except Exception as e:
        st.error(f"❌ Error fetching portfolio data: {e}")
        st.info("💡 Make sure you're logged in and have an active trading session")

# Footer
st.markdown("---")
st.caption("🔴 LIVE Dashboard powered by Zerodha Kite Connect WebSocket API")
st.caption("📊 **Technical Indicators:** EMA (9, 21, 50) | SMA (20, 50, 200) | Bollinger Bands (20, 2) | Supertrend (10, 3) | RSI | MACD")
st.caption("⏰ **Market Hours:** 9:15 AM - 3:30 PM IST (Mon-Fri)")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice. Trade at your own risk.")
st.caption(f"📅 Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
