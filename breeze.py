"""
Kite Connect F&O Trading Dashboard - Complete Version
With Live Monitor, Options Chain, Advanced Charts & Pattern Recognition
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
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
if 'live_monitor_running' not in st.session_state:
    st.session_state.live_monitor_running = False

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
        
        # Try exact match first
        options_data = instruments_nfo[
            (instruments_nfo['name'] == symbol) & 
            (instruments_nfo['expiry'] == expiry_dt) &
            (instruments_nfo['instrument_type'].isin(['CE', 'PE']))
        ].copy()
        
        # If no data, try common variations
        if options_data.empty:
            symbol_variations = [
                symbol,
                symbol.replace('-', ''),
                symbol.replace('&', ''),
                symbol + 'E' if symbol.startswith('BAJAJ') else symbol
            ]
            
            for var in symbol_variations:
                options_data = instruments_nfo[
                    (instruments_nfo['name'] == var) & 
                    (instruments_nfo['expiry'] == expiry_dt) &
                    (instruments_nfo['instrument_type'].isin(['CE', 'PE']))
                ].copy()
                if not options_data.empty:
                    break
        
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

def calculate_ichimoku(df):
    """Calculate Ichimoku Cloud"""
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
    """Calculate Fibonacci Retracement"""
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
    """Calculate On-Balance Volume"""
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
    """Calculate Chaikin Money Flow"""
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.fillna(0)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return cmf

def detect_candlestick_patterns(df):
    """Detect candlestick patterns"""
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
    """Detect chart patterns"""
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
        
        # Calculate next Tuesday (weekly expiry) - Tuesday is 1
        days_ahead = 1 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_expiry = today + timedelta(days=days_ahead)
        
        # Calculate monthly expiry (last Tuesday of current month)
        # Get last day of current month
        if today.month == 12:
            last_day = today.replace(day=31)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
        
        # Find last Tuesday
        days_to_subtract = (last_day.weekday() - 1) % 7
        monthly_expiry = last_day - timedelta(days=days_to_subtract)
        
        # Generate expiry list
        expiries = []
        for i in range(5):
            expiry = next_expiry + timedelta(weeks=i)
            expiries.append(expiry.strftime("%Y-%m-%d"))
        
        # Add monthly expiry if not already in list
        monthly_str = monthly_expiry.strftime("%Y-%m-%d")
        if monthly_str not in expiries:
            expiries.append(monthly_str)
        
        # Sort expiries
        expiries = sorted(list(set(expiries)))
        
        selected_expiry = st.selectbox(
            "Expiry Date",
            expiries,
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%d %b %Y"),
            key="options_expiry"
        )
    
    with col3:
        if st.button("🔄 Refresh", key="refresh_options"):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner(f"Loading options chain for {selected_stock_oc}..."):
        spot_price = get_spot_price(selected_stock_oc)
        
        if spot_price:
            st.info(f"💹 **Spot Price:** ₹{spot_price:.2f}")
            
            options_df = get_options_chain(selected_stock_oc, selected_expiry)
            
            if options_df is not None and not options_df.empty:
                ce_data = options_df[options_df['instrument_type'] == 'CE'].copy()
                pe_data = options_df[options_df['instrument_type'] == 'PE'].copy()
                
                ce_data = ce_data.sort_values('strike')
                pe_data = pe_data.sort_values('strike')
                
                strikes = sorted(options_df['strike'].unique())
                spot_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
                start_idx = max(0, spot_idx - 10)
                end_idx = min(len(strikes), spot_idx + 11)
                filtered_strikes = strikes[start_idx:end_idx]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_ce_oi = ce_data[ce_data['strike'].isin(filtered_strikes)]['oi'].sum()
                    st.metric("Total CALL OI", f"{total_ce_oi:,.0f}")
                
                with col2:
                    total_pe_oi = pe_data[pe_data['strike'].isin(filtered_strikes)]['oi'].sum()
                    st.metric("Total PUT OI", f"{total_pe_oi:,.0f}")
                
                with col3:
                    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
                    st.metric("Put-Call Ratio", f"{pcr:.2f}")
                
                with col4:
                    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
                    st.metric("ATM Strike", f"₹{atm_strike:.0f}")
                
                st.markdown("---")
                st.subheader(f"Options Chain - {selected_stock_oc}")
                
                chain_data = []
                for strike in filtered_strikes:
                    ce_row = ce_data[ce_data['strike'] == strike]
                    pe_row = pe_data[pe_data['strike'] == strike]
                    
                    row = {
                        'CE_OI': ce_row['oi'].values[0] if not ce_row.empty else 0,
                        'CE_Volume': ce_row['volume'].values[0] if not ce_row.empty else 0,
                        'CE_LTP': ce_row['ltp'].values[0] if not ce_row.empty else 0,
                        'Strike': strike,
                        'PE_LTP': pe_row['ltp'].values[0] if not pe_row.empty else 0,
                        'PE_Volume': pe_row['volume'].values[0] if not pe_row.empty else 0,
                        'PE_OI': pe_row['oi'].values[0] if not pe_row.empty else 0,
                    }
                    chain_data.append(row)
                
                chain_df = pd.DataFrame(chain_data)
                
                def highlight_atm(row):
                    if row['Strike'] == atm_strike:
                        return ['background-color: #ffffcc'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    chain_df.style.apply(highlight_atm, axis=1).format({
                        'CE_OI': '{:,.0f}',
                        'CE_Volume': '{:,.0f}',
                        'CE_LTP': '₹{:.2f}',
                        'Strike': '₹{:.0f}',
                        'PE_LTP': '₹{:.2f}',
                        'PE_Volume': '{:,.0f}',
                        'PE_OI': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=600
                )
                
                st.caption("💡 **ATM Strike** highlighted in yellow")
                
                # OI Chart
                st.markdown("---")
                st.subheader("📊 Open Interest Distribution")
                
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(
                    x=chain_df['Strike'],
                    y=chain_df['CE_OI'],
                    name='CALL OI',
                    marker_color='#ef5350',
                    opacity=0.7
                ))
                fig_oi.add_trace(go.Bar(
                    x=chain_df['Strike'],
                    y=chain_df['PE_OI'],
                    name='PUT OI',
                    marker_color='#26a69a',
                    opacity=0.7
                ))
                fig_oi.add_vline(
                    x=spot_price,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"Spot: ₹{spot_price:.2f}"
                )
                fig_oi.update_layout(
                    title="Call vs Put OI",
                    xaxis_title="Strike (₹)",
                    yaxis_title="OI",
                    height=400,
                    barmode='group'
                )
                st.plotly_chart(fig_oi, use_container_width=True)
                
            else:
                st.warning(f"❌ No options data for {selected_stock_oc} on {selected_expiry}")
                st.info("💡 Try different stock or expiry date")
        else:
            st.error("❌ Unable to fetch spot price")

# TAB 2: CHARTS
with tab2:
    st.header("Stock Charts with Advanced Indicators")
    st.caption("📊 Advanced charting with pattern recognition")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_stock = st.selectbox("Select Stock", FNO_STOCKS, key="chart_stock")
    
    with col2:
        period = st.selectbox("Period", ["1 Week", "2 Weeks", "1 Month"], key="chart_period")
        days_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}
        days = days_map[period]
    
    with col3:
        interval = st.selectbox(
            "Interval",
            ["day", "60minute", "15minute"],
            format_func=lambda x: {"day": "Daily", "60minute": "60 Min", "15minute": "15 Min"}[x],
            key="chart_interval"
        )
    
    with col4:
        chart_type = st.selectbox(
            "Chart Type",
            ["Candlestick + MA", "Ichimoku Cloud", "Fibonacci", "Volume Analysis"],
            key="chart_type"
        )
    
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = fetch_historical_data(selected_stock, days, interval)
    
    if df is not None and not df.empty:
        # Calculate indicators
        df['EMA_9'] = calculate_ema(df['close'], 9)
        df['EMA_21'] = calculate_ema(df['close'], 21)
        df['EMA_50'] = calculate_ema(df['close'], 50)
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['SMA_50'] = calculate_sma(df['close'], 50)
        df['SMA_200'] = calculate_sma(df['close'], 200)
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['close'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['close'])
        df['Tenkan'], df['Kijun'], df['SpanA'], df['SpanB'], df['Chikou'] = calculate_ichimoku(df)
        df['OBV'] = calculate_obv(df)
        df['CMF'] = calculate_cmf(df)
        
        # Pattern detection
        candlestick_patterns = detect_candlestick_patterns(df)
        chart_patterns = detect_chart_patterns(df)
        
        # Metrics
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
        
        # Pattern Recognition
        if candlestick_patterns or chart_patterns:
            st.markdown("---")
            st.subheader("🎯 Pattern Recognition")
            
            if candlestick_patterns:
                st.markdown("**📊 Candlestick Patterns:**")
                cols = st.columns(len(candlestick_patterns))
                for idx, (pattern_name, signal, description) in enumerate(candlestick_patterns):
                    with cols[idx]:
                        color = "🟢" if signal == "Bullish" else "🔴" if signal == "Bearish" else "⚪"
                        st.markdown(f"{color} **{pattern_name}**")
                        st.caption(f"{signal} - {description}")
            
            if chart_patterns:
                st.markdown("**📈 Chart Patterns:**")
                cols = st.columns(len(chart_patterns))
                for idx, (pattern_name, signal, description) in enumerate(chart_patterns):
                    with cols[idx]:
                        color = "🟢" if signal == "Bullish" else "🔴" if signal == "Bearish" else "⚪"
                        st.markdown(f"{color} **{pattern_name}**")
                        st.caption(f"{signal} - {description}")
        
        st.markdown("---")
        
        # Format data for plotting
        if interval != 'day':
            df_plot = df.copy()
            df_plot.index = df_plot.index.strftime('%d %b %H:%M')
            x_data = df_plot.index
            xaxis_type = 'category'
        else:
            df_plot = df.copy()
            x_data = df.index
            xaxis_type = 'date'
        
        # MAIN CHART
        if chart_type == "Candlestick + MA":
            st.subheader(f"📊 {selected_stock} - Price Chart with Moving Averages")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=x_data,
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Price'
            ))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['EMA_9'], name='EMA 9', 
                                    line=dict(color='#4CAF50', width=1.5)))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['EMA_21'], name='EMA 21', 
                                    line=dict(color='#FF9800', width=1.5)))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['EMA_50'], name='EMA 50', 
                                    line=dict(color='#9C27B0', width=1.5)))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['SMA_20'], name='SMA 20', 
                                    line=dict(color='#FF5722', width=1.5, dash='dash')))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['SMA_50'], name='SMA 50', 
                                    line=dict(color='#FFC107', width=1.5, dash='dash')))
            
            if len(df_plot) >= 200:
                fig.add_trace(go.Scatter(x=x_data, y=df_plot['SMA_200'], name='SMA 200', 
                                        line=dict(color='#795548', width=2, dash='dash')))
            
            fig.update_layout(
                title=f"{selected_stock} - Candlestick + MA",
                yaxis_title="Price (₹)",
                height=650,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                xaxis=dict(type=xaxis_type)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Ichimoku Cloud":
            st.subheader(f"☁️ {selected_stock} - Ichimoku Cloud")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=x_data,
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Price'
            ))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['Tenkan'], name='Tenkan-sen (9)', 
                                    line=dict(color='#FF6B6B', width=1.5)))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['Kijun'], name='Kijun-sen (26)', 
                                    line=dict(color='#4ECDC4', width=1.5)))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['SpanA'], name='Senkou Span A', 
                                    line=dict(color='rgba(0, 255, 0, 0.3)', width=0.5)))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['SpanB'], name='Senkou Span B', 
                                    line=dict(color='rgba(255, 0, 0, 0.3)', width=0.5),
                                    fill='tonexty', fillcolor='rgba(124, 252, 0, 0.2)'))
            fig.add_trace(go.Scatter(x=x_data, y=df_plot['Chikou'], name='Chikou Span', 
                                    line=dict(color='#9B59B6', width=1.5, dash='dot')))
            
            fig.update_layout(
                title=f"{selected_stock} - Ichimoku Cloud",
                yaxis_title="Price (₹)",
                height=650,
                xaxis_rangeslider_visible=False,
                xaxis=dict(type=xaxis_type)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Price above cloud = Bullish | Below cloud = Bearish")
        
        elif chart_type == "Fibonacci":
            st.subheader(f"📐 {selected_stock} - Fibonacci Retracement")
            
            fib_levels, fib_high, fib_low = calculate_fibonacci_levels(df)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=x_data,
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Price'
            ))
            
            colors = ['#FF0000', '#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#00FF00', '#0000FF']
            for idx, (level_name, level_value) in enumerate(fib_levels.items()):
                fig.add_hline(
                    y=level_value,
                    line_dash="dash",
                    line_color=colors[idx % len(colors)],
                    annotation_text=f"Fib {level_name} (₹{level_value:.2f})"
                )
            
            fig.update_layout(
                title=f"{selected_stock} - Fibonacci Levels",
                yaxis_title="Price (₹)",
                height=650,
                xaxis_rangeslider_visible=False,
                xaxis=dict(type=xaxis_type)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Swing High", f"₹{fib_high:.2f}")
            with col2:
                st.metric("Swing Low", f"₹{fib_low:.2f}")
        
        elif chart_type == "Volume Analysis":
            st.subheader(f"📊 {selected_stock} - Volume Analysis")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=x_data,
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Price',
                yaxis='y'
            ))
            
            colors = ['#26a69a' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else '#ef5350' 
                     for i in range(len(df_plot))]
            
            fig.add_trace(go.Bar(
                x=x_data,
                y=df_plot['volume'],
                name='Volume',
                yaxis='y2',
                marker_color=colors,
                opacity=0.5
            ))
            
            fig.update_layout(
                title=f"{selected_stock} - Price & Volume",
                yaxis_title="Price (₹)",
                yaxis2=dict(title="Volume", overlaying='y', side='right'),
                height=450,
                xaxis_rangeslider_visible=False,
                xaxis=dict(type=xaxis_type)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # OBV
            st.markdown("**On-Balance Volume (OBV)**")
            fig_obv = go.Figure()
            fig_obv.add_trace(go.Scatter(
                x=x_data, y=df_plot['OBV'],
                name='OBV',
                line=dict(color='#2196F3', width=2),
                fill='tozeroy'
            ))
            fig_obv.update_layout(title="OBV", height=250, xaxis=dict(type=xaxis_type))
            st.plotly_chart(fig_obv, use_container_width=True)
            
            # CMF
            st.markdown("**Chaikin Money Flow (CMF)**")
            fig_cmf = go.Figure()
            fig_cmf.add_trace(go.Scatter(
                x=x_data, y=df_plot['CMF'],
                name='CMF',
                line=dict(color='#9C27B0', width=2),
                fill='tozeroy'
            ))
            fig_cmf.add_hline(y=0, line_color="gray")
            fig_cmf.add_hline(y=0.2, line_dash="dash", line_color="green")
            fig_cmf.add_hline(y=-0.2, line_dash="dash", line_color="red")
            fig_cmf.update_layout(title="CMF", height=250, xaxis=dict(type=xaxis_type))
            st.plotly_chart(fig_cmf, use_container_width=True)
        
        # RSI & MACD
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 RSI")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=x_data, y=df_plot['RSI'], name='RSI', 
                                        line=dict(color='purple', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=300, yaxis=dict(range=[0, 100]), xaxis=dict(type=xaxis_type))
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            st.subheader("📈 MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=x_data, y=df_plot['MACD'], name='MACD', 
                                         line=dict(color='blue', width=2)))
            fig_macd.add_trace(go.Scatter(x=x_data, y=df_plot['MACD_signal'], name='Signal', 
                                         line=dict(color='red', width=2)))
            colors_macd = ['#26a69a' if val >= 0 else '#ef5350' for val in df_plot['MACD_hist']]
            fig_macd.add_trace(go.Bar(x=x_data, y=df_plot['MACD_hist'], name='Histogram', 
                                     marker_color=colors_macd, opacity=0.5))
            fig_macd.update_layout(height=300, xaxis=dict(type=xaxis_type))
            st.plotly_chart(fig_macd, use_container_width=True)
        
    else:
        st.error(f"❌ No data available for {selected_stock}")

# TAB 3: LIVE MONITOR
with tab3:
    st.header("🔴 LIVE Intraday Monitor (WebSocket)")
    st.caption("⏰ Market Hours: 9:15 AM - 3:30 PM IST | Updates every 3 seconds")
    
    # Check market hours
    now = datetime.now(IST)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_market_open = market_open <= now <= market_close and now.weekday() < 5
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        watchlist = st.multiselect(
            "Select Stocks (max 8)",
            FNO_STOCKS,
            default=["RELIANCE", "TCS", "HDFCBANK", "INFY"],
            max_selections=8,
            key="live_stocks"
        )
    
    with col2:
        if is_market_open:
            st.success("✅ Market is OPEN")
        else:
            st.error("❌ Market is CLOSED")
            if now.weekday() >= 5:
                st.info("📅 Weekend")
            elif now < market_open:
                st.info(f"⏰ Opens at 9:15 AM")
            else:
                st.info(f"⏰ Closed at 3:30 PM")
    
    with col3:
        if st.button("🔴 Start Live Stream", key="start_live_btn"):
            if watchlist and is_market_open:
                stop_websocket()  # Stop any existing connection
                st.session_state.live_data = {}  # Clear old data
                if start_websocket(watchlist):
                    st.success("✅ Connected!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to connect")
            elif not watchlist:
                st.warning("⚠️ Select stocks first")
            else:
                st.warning("⚠️ Market is closed")
        
        if st.button("⏹ Stop Stream", key="stop_live_btn"):
            stop_websocket()
            st.info("⏹ Stopped")
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.ticker_active and watchlist:
        st.success(f"🔴 LIVE: Streaming {len(watchlist)} stocks")
        
        # Auto-refresh placeholder
        placeholder = st.empty()
        
        with placeholder.container():
            # Display live data
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
                                
                                st.markdown(f"### {arrow} **{symbol}**")
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("LTP", f"₹{ltp:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                                with col_b:
                                    st.metric("Volume", f"{data['volume']:,}")
                                
                                st.caption(f"**O:** ₹{data['open']:.2f} | **H:** ₹{data['high']:.2f} | **L:** ₹{data['low']:.2f}")
                                st.caption(f"⏱ {data['timestamp'].strftime('%H:%M:%S IST')}")
                            else:
                                st.info(f"⏳ Connecting {symbol}...")
        
        # Auto-refresh after 3 seconds
        time.sleep(3)
        st.rerun()
    
    elif watchlist and not is_market_open:
        st.warning("⚠️ Live streaming only works during market hours (Mon-Fri, 9:15 AM - 3:30 PM IST)")
        st.info("💡 Click 'Start Live Stream' button during market hours to begin")
    else:
        st.info("👆 Select stocks and click 'Start Live Stream' button")

# TAB 4: PORTFOLIO
with tab4:
    st.header("📊 Your Portfolio")
    
    try:
        kite = st.session_state.kite
        
        st.subheader("💼 Holdings")
        holdings = kite.holdings()
        
        if holdings:
            df_holdings = pd.DataFrame(holdings)
            st.dataframe(
                df_holdings[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']],
                use_container_width=True,
                height=400
            )
        else:
            st.info("📭 No holdings found")
        
        st.markdown("---")
        
        st.subheader("📈 Open Positions")
        positions = kite.positions()
        
        if positions and positions.get('net'):
            df_positions = pd.DataFrame(positions['net'])
            df_positions = df_positions[df_positions['quantity'] != 0]
            
            if not df_positions.empty:
                st.dataframe(
                    df_positions[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']],
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("📭 No open positions")
        else:
            st.info("📭 No open positions")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("🔴 Dashboard powered by Zerodha Kite Connect API")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")
st.caption(f"📅 Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
