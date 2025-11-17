"""
Kite Connect F&O Trading Dashboard with WebSocket Live Streaming
Complete version with all features working
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from kiteconnect import KiteConnect, KiteTicker
import time
import threading
import pytz

# At the top with other imports
import yfinance as yf
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
API_SECRET = "mgso1jdnxj3xeei228dcciyqqx7axl77"  # ⚠️ REPLACE THIS

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

# --------------------------
# Initialize session state
# --------------------------
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'kite_connected' not in st.session_state:
    st.session_state.kite_connected = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'instruments_nse' not in st.session_state:
    st.session_state.instruments_nse = None
if 'instruments_nfo' not in st.session_state:
    st.session_state.instruments_nfo = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'ticker_active' not in st.session_state:
    st.session_state.ticker_active = False
if 'kws' not in st.session_state:
    st.session_state.kws = None

# --------------------------
# Login Management
# --------------------------
st.title("📈 F&O Dashboard - Kite Connect")

if not st.session_state.kite_connected:
    st.header("🔐 Login to Kite Connect")
    
    st.markdown("""
    ### How to get your Access Token:
    1. Click the login link below
    2. After login, copy the `request_token` from URL
    3. Paste it below and generate access token
    """)
    
    login_url = f"https://kite.zerodha.com/connect/login?api_key={API_KEY}&v=3"
    st.markdown(f"### Step 1: [Click here to Login to Kite]({login_url})")
    
    st.markdown("### Step 2: Enter Request Token")
    request_token = st.text_input("Paste Request Token here:", key="request_token_input")
    
    if st.button("🔑 Generate Access Token", key="generate_token"):
        if request_token and API_SECRET != "YOUR_API_SECRET_HERE":
            try:
                with st.spinner("Generating access token..."):
                    kite = KiteConnect(api_key=API_KEY)
                    data = kite.generate_session(request_token, api_secret=API_SECRET)
                    access_token = data["access_token"]
                    
                    kite.set_access_token(access_token)
                    profile = kite.profile()
                    
                    st.session_state.kite = kite
                    st.session_state.access_token = access_token
                    st.session_state.kite_connected = True
                    st.session_state.profile = profile
                    
                    st.success(f"✅ Connected! Welcome {profile.get('user_name', 'User')}")
                    st.info(f"💾 Save this Access Token: `{access_token}`")
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        elif API_SECRET == "YOUR_API_SECRET_HERE":
            st.error("⚠️ Please set your API_SECRET in the code!")
        else:
            st.warning("⚠️ Please enter the request token")
    
    st.markdown("---")
    st.markdown("### OR Use Existing Access Token")
    manual_token = st.text_input("Paste Access Token:", key="manual_token")
    
    if st.button("🔗 Connect", key="connect_token"):
        if manual_token:
            try:
                with st.spinner("Connecting..."):
                    kite = KiteConnect(api_key=API_KEY)
                    kite.set_access_token(manual_token)
                    profile = kite.profile()
                    
                    st.session_state.kite = kite
                    st.session_state.access_token = manual_token
                    st.session_state.kite_connected = True
                    st.session_state.profile = profile
                    
                    st.success(f"✅ Connected! Welcome {profile.get('user_name', 'User')}")
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter an access token")
    
    st.stop()

# --------------------------
# Helper Functions
# --------------------------
@st.cache_data(ttl=600)
def get_instruments_nfo():
    """Get NFO instruments"""
    try:
        instruments = st.session_state.kite.instruments("NFO")
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Error fetching NFO instruments: {e}")
        return None

@st.cache_data(ttl=600)
def get_instruments_nse():
    """Get NSE instruments"""
    try:
        instruments = st.session_state.kite.instruments("NSE")
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Error fetching NSE instruments: {e}")
        return None

def filter_market_hours(df):
    """Filter dataframe to market hours"""
    if df is None or df.empty:
        return df
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        elif df.index.tz != IST:
            df.index = df.index.tz_convert(IST)
        df_filtered = df.between_time('09:15', '15:30')
        return df_filtered
    except:
        return df

def fetch_historical_data(symbol, days=30, interval="day"):
    """Fetch historical data"""
    try:
        kite = st.session_state.kite
        instruments_nse = get_instruments_nse()
        if instruments_nse is None:
            return None
        
        result = instruments_nse[instruments_nse['tradingsymbol'] == symbol]
        if result.empty:
            return None
        
        instrument_token = result.iloc[0]['instrument_token']
        to_date = datetime.now(IST)
        
        if interval in ["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute"]:
            if days > 30:
                days = 30
            if to_date.weekday() >= 5:
                days_back = to_date.weekday() - 4
                to_date = to_date - timedelta(days=days_back)
            to_date = to_date.replace(hour=15, minute=30, second=0, microsecond=0)
            from_date = to_date - timedelta(days=days)
            from_date = from_date.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            from_date = to_date - timedelta(days=days)
        
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
                if interval != "day":
                    df = filter_market_hours(df)
                df = df[(df[['open', 'high', 'low', 'close']] != 0).all(axis=1)]
                return df
        return None
    except Exception as e:
        st.warning(f"Data fetch error: {e}")
        return None

def get_spot_price(symbol):
    """Get current spot price"""
    try:
        kite = st.session_state.kite
        quote = kite.quote(f"NSE:{symbol}")
        if quote and f"NSE:{symbol}" in quote:
            return quote[f"NSE:{symbol}"]["last_price"]
        return None
    except Exception as e:
        return None

def get_options_chain(symbol, expiry_date):
    """Fetch options chain"""
    try:
        kite = st.session_state.kite
        instruments_nfo = get_instruments_nfo()
        if instruments_nfo is None:
            return None
        
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        
        options_data = instruments_nfo[
            (instruments_nfo['name'] == symbol) & 
            (instruments_nfo['expiry'] == expiry_dt) &
            (instruments_nfo['instrument_type'].isin(['CE', 'PE']))
        ].copy()
        
        if options_data.empty:
            return None
        
        symbols_list = [f"NFO:{ts}" for ts in options_data['tradingsymbol'].tolist()]
        
        chunk_size = 500
        all_quotes = {}
        
        for i in range(0, len(symbols_list), chunk_size):
            chunk = symbols_list[i:i + chunk_size]
            quotes = kite.quote(chunk)
            all_quotes.update(quotes)
        
        options_data['ltp'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('last_price', 0)
        )
        options_data['volume'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('volume', 0)
        )
        options_data['oi'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('oi', 0)
        )
        options_data['bid'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('depth', {}).get('buy', [{}])[0].get('price', 0) if all_quotes.get(f"NFO:{x}", {}).get('depth') else 0
        )
        options_data['ask'] = options_data['tradingsymbol'].apply(
            lambda x: all_quotes.get(f"NFO:{x}", {}).get('depth', {}).get('sell', [{}])[0].get('price', 0) if all_quotes.get(f"NFO:{x}", {}).get('depth') else 0
        )
        
        return options_data
        
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return None

# --------------------------
# WebSocket Functions
# --------------------------
def start_websocket(symbols):
    """Start WebSocket connection"""
    try:
        kite = st.session_state.kite
        instruments_nse = get_instruments_nse()
        if instruments_nse is None:
            return False
        
        tokens_map = {}
        for symbol in symbols:
            result = instruments_nse[instruments_nse['tradingsymbol'] == symbol]
            if not result.empty:
                tokens_map[symbol] = result.iloc[0]['instrument_token']
        
        if not tokens_map:
            return False
        
        tokens = list(tokens_map.values())
        kws = KiteTicker(API_KEY, st.session_state.access_token)
        
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
            pass
        
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
    """Stop WebSocket"""
    try:
        if st.session_state.kws:
            st.session_state.kws.close()
            st.session_state.ticker_active = False
            st.session_state.kws = None
            st.session_state.live_data = {}
    except:
        pass

# --------------------------
# Technical Indicators
# --------------------------
def calculate_sma(data, period):
    if len(data) < period:
        return pd.Series([None] * len(data), index=data.index)
    return data.rolling(window=period, min_periods=1).mean()

def calculate_ema(data, period):
    if len(data) < period:
        return pd.Series([None] * len(data), index=data.index)
    return data.ewm(span=period, adjust=False, min_periods=1).mean()

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

def calculate_bollinger_bands(data, period=20, std_dev=2):
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
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
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

def calculate_ichimoku(df):
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    
    chikou_span = df['close'].shift(-26)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_fibonacci_levels(df, lookback=50):
    recent_data = df.tail(lookback)
    high = recent_data['high'].max()
    low = recent_data['low'].min()
    diff = high - low
    
    levels = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }
    
    return levels, high, low

def calculate_obv(df):
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_cmf(df, period=20):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.fillna(0)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return cmf

def detect_candlestick_patterns(df):
    patterns = []
    
    if len(df) < 3:
        return patterns
    
    current = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    
    body_current = abs(current['close'] - current['open'])
    body_prev1 = abs(prev1['close'] - prev1['open'])
    
    # Doji
    if body_current < (current['high'] - current['low']) * 0.1:
        patterns.append(("Doji", "Neutral", "Indecision"))
    
    # Hammer
    lower_shadow = current['open'] - current['low'] if current['close'] > current['open'] else current['close'] - current['low']
    upper_shadow = current['high'] - current['close'] if current['close'] > current['open'] else current['high'] - current['open']
    
    if lower_shadow > body_current * 2 and upper_shadow < body_current * 0.3:
        if current['close'] < prev1['close']:
            patterns.append(("Hammer", "Bullish", "Reversal at bottom"))
    
    # Shooting Star
    if upper_shadow > body_current * 2 and lower_shadow < body_current * 0.3:
        if current['close'] > prev1['close']:
            patterns.append(("Shooting Star", "Bearish", "Reversal at top"))
    
    # Bullish Engulfing
    if (current['close'] > current['open'] and prev1['close'] < prev1['open'] and
        current['open'] < prev1['close'] and current['close'] > prev1['open']):
        patterns.append(("Bullish Engulfing", "Bullish", "Strong reversal"))
    
    # Bearish Engulfing
    if (current['close'] < current['open'] and prev1['close'] > prev1['open'] and
        current['open'] > prev1['close'] and current['close'] < prev1['open']):
        patterns.append(("Bearish Engulfing", "Bearish", "Strong reversal"))
    
    # Morning Star
    if prev2 is not None:
        body_prev2 = abs(prev2['close'] - prev2['open'])
        if (prev2['close'] < prev2['open'] and body_prev1 < body_prev2 * 0.3 and
            current['close'] > current['open'] and current['close'] > (prev2['open'] + prev2['close']) / 2):
            patterns.append(("Morning Star", "Bullish", "Strong reversal"))
    
    # Evening Star
    if prev2 is not None:
        body_prev2 = abs(prev2['close'] - prev2['open'])
        if (prev2['close'] > prev2['open'] and body_prev1 < body_prev2 * 0.3 and
            current['close'] < current['open'] and current['close'] < (prev2['open'] + prev2['close']) / 2):
            patterns.append(("Evening Star", "Bearish", "Strong reversal"))
    
    return patterns

def detect_chart_patterns(df, window=20):
    patterns = []
    
    if len(df) < window:
        return patterns
    
    recent = df.tail(window)
    highs = recent['high']
    lows = recent['low']
    
    # Head and Shoulders
    if len(recent) >= 5:
        mid_point = len(recent) // 2
        left_shoulder = highs.iloc[:mid_point-1].max()
        head = highs.iloc[mid_point-1:mid_point+2].max()
        right_shoulder = highs.iloc[mid_point+2:].max()
        
        if head > left_shoulder * 1.02 and head > right_shoulder * 1.02:
            if abs(left_shoulder - right_shoulder) / left_shoulder < 0.02:
                patterns.append(("Head & Shoulders", "Bearish", "Major reversal"))
    
    # Double Top
    peaks = []
    for i in range(2, len(recent)-2):
        if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and
            highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
            peaks.append((i, highs.iloc[i]))
    
    if len(peaks) >= 2:
        last_two = peaks[-2:]
        if abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] < 0.02:
            patterns.append(("Double Top", "Bearish", "Reversal pattern"))
    
    # Triangle
    recent_highs = highs.tail(10)
    recent_lows = lows.tail(10)
    
    high_trend = (recent_highs.iloc[-1] - recent_highs.iloc[0]) / recent_highs.iloc[0]
    low_trend = (recent_lows.iloc[-1] - recent_lows.iloc[0]) / recent_lows.iloc[0]
    
    if abs(high_trend) < 0.02 and low_trend > 0.03:
        patterns.append(("Ascending Triangle", "Bullish", "Continuation"))
    elif high_trend < -0.03 and abs(low_trend) < 0.02:
        patterns.append(("Descending Triangle", "Bearish", "Continuation"))
    
    return patterns

# --------------------------
# Main Dashboard
# --------------------------

profile = st.session_state.profile
col1, col2 = st.columns([3, 1])
with col1:
    st.success(f"✅ Connected | User: {profile.get('user_name', 'N/A')}")
with col2:
    if st.button("🔓 Logout", key="logout"):
        stop_websocket()
        st.session_state.kite_connected = False
        st.session_state.kite = None
        st.rerun()

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Options Chain", "💹 Charts & Indicators", "🔴 LIVE Monitor", "📊 Portfolio"])

# TAB 1: OPTIONS CHAIN
with tab1:
    st.header("⚡ Options Chain Analysis")
    st.caption("📊 Real-time Call & Put Options Data | Market Hours: 9:15 AM - 3:30 PM IST")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stock_oc = st.selectbox("Select Stock", FNO_STOCKS, key="options_stock")
    
    with col2:
        today = datetime.now(IST).date()
        
        # Function to get last Tuesday of a month
        def get_last_tuesday(year, month):
            if month == 12:
                last_day = datetime(year, month, 31).date()
            else:
                next_month = datetime(year, month + 1, 1).date()
                last_day = next_month - timedelta(days=1)
            days_to_subtract = (last_day.weekday() - 1) % 7
            return last_day - timedelta(days=days_to_subtract)
        
        # Get next 6 monthly expiries
        expiries = []
        current_year = today.year
        current_month = today.month
        
        for i in range(6):
            month = current_month + i
            year = current_year
            if month > 12:
                month =
