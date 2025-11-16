"""
Kite Connect F&O Trading Dashboard with WebSocket Live Streaming
Real-time tick data using KiteTicker
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import feedparser
from kiteconnect import KiteConnect, KiteTicker
import time
import threading
import queue

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
            if to_date.weekday() >= 5:  # Saturday or Sunday
                days_back = to_date.weekday() - 4
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
        except:
            return None
    except:
        return None

# --------------------------
# WebSocket Functions
# --------------------------
def start_websocket(symbols):
    """Start WebSocket connection for live data"""
    try:
        # Get instrument tokens
        tokens_map = get_instrument_tokens(symbols)
        if not tokens_map:
            st.error("Could not get instrument tokens")
            return
        
        tokens = list(tokens_map.values())
        
        # Initialize KiteTicker
        kws = KiteTicker(API_KEY, ACCESS_TOKEN)
        
        def on_ticks(ws, ticks):
            """Callback for receiving ticks"""
            for tick in ticks:
                token = tick['instrument_token']
                # Find symbol for this token
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
                        'timestamp': datetime.now(),
                        'ohlc': tick.get('ohlc', {}),
                        'high': tick.get('ohlc', {}).get('high', 0),
                        'low': tick.get('ohlc', {}).get('low', 0),
                        'open': tick.get('ohlc', {}).get('open', 0),
                        'close': tick.get('ohlc', {}).get('close', 0)
                    }
        
        def on_connect(ws, response):
            """Callback on successful connect"""
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            st.session_state.ticker_active = True
        
        def on_close(ws, code, reason):
            """Callback on connection close"""
            st.session_state.ticker_active = False
        
        def on_error(ws, code, reason):
            """Callback on error"""
            st.error(f"WebSocket Error: {reason}")
        
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error
        
        # Start in background thread
        st.session_state.kws = kws
        
        # Run WebSocket in a separate thread
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
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
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

# TAB 1: NEWS (same as before)
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
            
            st.markdown(f"**[{article['Title']}]({article['Link']})**")
            st.markdown(
                f"<span style='background-color: {sentiment_colors[article['Sentiment']]}; "
                f"color: white; padding: 3px 10px; border-radius: 4px; font-size: 11px;'>"
                f"{sentiment_emoji[article['Sentiment']]} {article['Sentiment'].upper()}</span>",
                unsafe_allow_html=True
            )
            st.caption(f"Source: {article['Source']} | {article['Published']}")
            st.markdown("---")

# TAB 2: CHARTS (same as before, keeping it brief)
with tab2:
    st.header("Stock Charts with Technical Indicators")
    
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
            ["day", "60minute", "30minute", "15minute"],
            format_func=lambda x: {"day": "Daily", "60minute": "60 Min", "30minute": "30 Min", "15minute": "15 Min"}[x],
            key="chart_interval"
        )
    
    if interval != "day" and days > 30:
        days = 30
    
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = fetch_historical_data(selected_stock, days, interval)
    
    if df is not None and not df.empty:
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['RSI'] = calculate_rsi(df['close'])
        
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
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], 
            low=df['low'], close=df['close']
        )])
        fig.update_layout(title=f"{selected_stock}", height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: LIVE MONITOR WITH WEBSOCKET
with tab3:
    st.header("🔴 LIVE Intraday Monitor (WebSocket)")
    
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
                stop_websocket()  # Stop any existing connection
                if start_websocket(watchlist):
                    st.success("✅ WebSocket Connected!")
                    time.sleep(1)
                    st.rerun()
        
        if st.button("⏹️ Stop Stream", key="stop_live"):
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
                            st.markdown(f"### {arrow} {symbol}")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("LTP", f"₹{ltp:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                            with col_b:
                                st.metric("Volume", f"{data['volume']:,}")
                            
                            # OHLC
                            st.caption(f"O: ₹{data['open']:.2f} | H: ₹{data['high']:.2f} | L: ₹{data['low']:.2f} | C: ₹{data['close']:.2f}")
                            
                            # Timestamp
                            st.caption(f"⏱️ Updated: {data['timestamp'].strftime('%H:%M:%S')}")
                        else:
                            st.info(f"Waiting for {symbol} data...")
        
        # Refresh button at bottom
        st.markdown("---")
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    
    elif watchlist:
        st.info("👆 Click 'Start Live Stream' to begin receiving live data")
        st.warning("⚠️ WebSocket streams only work during market hours (9:15 AM - 3:30 PM)")
    else:
        st.info("👆 Select stocks to monitor")

# TAB 4: PORTFOLIO (same as before)
with tab4:
    st.header("📊 Your Portfolio")
    
    try:
        kite = st.session_state.kite
        
        st.subheader("Holdings")
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
                st.metric("Current", f"₹{total_current:,.2f}")
            with col4:
                pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
                st.metric("P&L", f"₹{total_pnl:,.2f}", f"{pnl_pct:.2f}%")
            
            display_cols = ['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']
            if all(col in df_holdings.columns for col in display_cols):
                st.dataframe(df_holdings[display_cols], use_container_width=True)
        else:
            st.info("No holdings found")
        
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("🔴 LIVE Dashboard powered by Zerodha Kite Connect WebSocket API")
st.caption("⚠️ **Disclaimer:** For educational purposes only. Not financial advice.")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
