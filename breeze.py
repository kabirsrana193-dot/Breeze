import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from breeze_connect import BreezeConnect
import threading
import queue

# Page config
st.set_page_config(
    page_title="Live F&O Dashboard - Breeze",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------
# Breeze Configuration
# --------------------------
app_key = "68`47N89970w1dH7u1s5347j8403f287"
secret_key = "5v9k141093cf4361528$z24Q7(Yv2839"
session_token = "53705299"

# --------------------------
# Config - Top F&O Stocks
# --------------------------
FNO_STOCKS = [
    "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Bharti Airtel", "ITC",
    "State Bank of India", "Hindustan Unilever", "Bajaj Finance", 
    "Kotak Mahindra Bank", "Axis Bank", "Larsen & Toubro", "Asian Paints", 
    "Maruti Suzuki", "Titan", "Sun Pharma", "HCL Tech", "Adani Enterprises",
    "Tata Motors", "Wipro", "NTPC", "Bajaj Finserv", "Tata Steel",
    "Hindalco", "IndusInd Bank", "Mahindra & Mahindra", "Coal India",
    "JSW Steel", "Tata Consumer", "Eicher Motors", "BPCL", "Tech Mahindra",
    "Dr Reddy", "Cipla", "UPL", "Britannia", "Divi's Lab",
    "HDFC Life", "Adani Ports", "ONGC", "IOC", "Vedanta", "Bajaj Auto", 
    "Hero MotoCorp", "GAIL", "UltraTech"
]

# Stock code mapping for Breeze API
STOCK_CODE_MAP = {
    "Reliance": "RELIND", "TCS": "TCS", "HDFC Bank": "HDFCBK",
    "Infosys": "INFY", "ICICI Bank": "ICIBAN", "Bharti Airtel": "BHARTI",
    "ITC": "ITC", "State Bank of India": "SBIN",
    "Hindustan Unilever": "HINUNL", "Bajaj Finance": "BAJFIN", 
    "Kotak Mahindra Bank": "KOTBAN", "Axis Bank": "AXIBNK", 
    "Larsen & Toubro": "LT", "Asian Paints": "ASIPAI", 
    "Maruti Suzuki": "MARUTI", "Titan": "TITAN", 
    "Sun Pharma": "SUNPHA", "HCL Tech": "HCLTECH",
    "Adani Enterprises": "ADAENT", "Tata Motors": "TATAMO",
    "Wipro": "WIPRO", "NTPC": "NTPC", "Bajaj Finserv": "BAJFNS",
    "Tata Steel": "TATSTE", "Hindalco": "HINDAL",
    "IndusInd Bank": "INDBNK", "Mahindra & Mahindra": "M&M",
    "Coal India": "COALIN", "JSW Steel": "JSWSTL",
    "Tata Consumer": "TATCON", "Eicher Motors": "EICMOT",
    "BPCL": "BPCL", "Tech Mahindra": "TECMAH", "Dr Reddy": "DRREDL",
    "Cipla": "CIPLA", "UPL": "UPL", "Britannia": "BRITAI",
    "Divi's Lab": "DIVLAB", "ONGC": "ONGC", "IOC": "IOC",
    "Vedanta": "VEDANT", "Bajaj Auto": "BAJAUT",
    "HDFC Life": "HDFLIFE", "Adani Ports": "ADANIS",
    "UltraTech": "ULTCEM", "Hero MotoCorp": "HEROMO",
    "GAIL": "GAIL"
}

# --------------------------
# Initialize session state
# --------------------------
if 'breeze_client' not in st.session_state:
    st.session_state.breeze_client = None
if 'breeze_connected' not in st.session_state:
    st.session_state.breeze_connected = False
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'subscribed_stocks' not in st.session_state:
    st.session_state.subscribed_stocks = set()
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = ["Reliance", "TCS", "HDFC Bank", "Infosys"]
if 'data_queue' not in st.session_state:
    st.session_state.data_queue = queue.Queue()

# --------------------------
# Breeze Connection
# --------------------------
if not st.session_state.breeze_connected:
    try:
        breeze = BreezeConnect(api_key=app_key)
        breeze.generate_session(api_secret=secret_key, session_token=session_token)
        st.session_state.breeze_client = breeze
        st.session_state.breeze_connected = True
    except Exception as e:
        st.error(f"❌ Failed to connect to Breeze API: {str(e)}")
        st.session_state.breeze_connected = False

# --------------------------
# WebSocket Functions
# --------------------------
def on_ticks(ticks):
    """Callback function for receiving live ticks"""
    try:
        if isinstance(ticks, dict):
            # Extract stock info from ticks
            stock_name = ticks.get('stock_name', '')
            symbol = ticks.get('symbol', '')
            
            # Find the display name
            display_name = None
            for name, code in STOCK_CODE_MAP.items():
                if code in stock_name or code in symbol:
                    display_name = name
                    break
            
            if display_name:
                st.session_state.live_data[display_name] = {
                    'ltp': ticks.get('last', 0),
                    'open': ticks.get('open', 0),
                    'high': ticks.get('high', 0),
                    'low': ticks.get('low', 0),
                    'close': ticks.get('close', 0),
                    'change': ticks.get('change', 0),
                    'volume': ticks.get('ttq', 0),
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'bid_price': ticks.get('bPrice', 0),
                    'ask_price': ticks.get('sPrice', 0),
                    'bid_qty': ticks.get('bQty', 0),
                    'ask_qty': ticks.get('sQty', 0),
                }
    except Exception as e:
        pass

def subscribe_stock(stock_name):
    """Subscribe to a stock for live data"""
    try:
        breeze = st.session_state.breeze_client
        if not breeze:
            return False
        
        stock_code = STOCK_CODE_MAP.get(stock_name)
        if not stock_code:
            return False
        
        # Check if already subscribed
        if stock_name in st.session_state.subscribed_stocks:
            return True
        
        # Subscribe to live feed
        response = breeze.subscribe_feeds(
            exchange_code="NSE",
            stock_code=stock_code,
            product_type="cash",
            get_market_depth=False,
            get_exchange_quotes=True
        )
        
        if response:
            st.session_state.subscribed_stocks.add(stock_name)
            return True
        
        return False
    except Exception as e:
        st.error(f"Error subscribing to {stock_name}: {str(e)}")
        return False

def unsubscribe_stock(stock_name):
    """Unsubscribe from a stock"""
    try:
        breeze = st.session_state.breeze_client
        if not breeze:
            return False
        
        stock_code = STOCK_CODE_MAP.get(stock_name)
        if not stock_code:
            return False
        
        breeze.unsubscribe_feeds(
            exchange_code="NSE",
            stock_code=stock_code,
            product_type="cash",
            get_market_depth=False,
            get_exchange_quotes=True
        )
        
        if stock_name in st.session_state.subscribed_stocks:
            st.session_state.subscribed_stocks.remove(stock_name)
        
        if stock_name in st.session_state.live_data:
            del st.session_state.live_data[stock_name]
        
        return True
    except Exception as e:
        st.error(f"Error unsubscribing from {stock_name}: {str(e)}")
        return False

def start_websocket():
    """Start WebSocket connection"""
    try:
        breeze = st.session_state.breeze_client
        if not breeze:
            return False
        
        # Assign callback
        breeze.on_ticks = on_ticks
        
        # Connect to websocket
        breeze.ws_connect()
        st.session_state.ws_connected = True
        return True
    except Exception as e:
        st.error(f"WebSocket connection error: {str(e)}")
        return False

def stop_websocket():
    """Stop WebSocket connection"""
    try:
        breeze = st.session_state.breeze_client
        if breeze and st.session_state.ws_connected:
            breeze.ws_disconnect()
            st.session_state.ws_connected = False
            st.session_state.subscribed_stocks.clear()
            st.session_state.live_data.clear()
        return True
    except Exception as e:
        st.error(f"Error disconnecting WebSocket: {str(e)}")
        return False

# --------------------------
# Historical Data Functions
# --------------------------
@st.cache_data(ttl=300)
def fetch_stock_data_breeze(stock_code, days=7, interval="1day"):
    """Fetch historical stock data using Breeze API"""
    try:
        breeze = st.session_state.breeze_client
        if not breeze:
            return pd.DataFrame()
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        from_date_str = from_date.strftime("%Y-%m-%d") + "T07:00:00.000Z"
        to_date_str = to_date.strftime("%Y-%m-%d") + "T23:59:59.000Z"
        
        response = breeze.get_historical_data(
            interval=interval,
            from_date=from_date_str,
            to_date=to_date_str,
            stock_code=stock_code,
            exchange_code="NSE",
            product_type="cash"
        )
        
        time.sleep(0.7)
        
        if response and 'Success' in response:
            data = response['Success']
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                
                column_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'datetime' in col_lower:
                        column_mapping[col] = 'Date'
                    elif col_lower == 'open':
                        column_mapping[col] = 'Open'
                    elif col_lower == 'high':
                        column_mapping[col] = 'High'
                    elif col_lower == 'low':
                        column_mapping[col] = 'Low'
                    elif col_lower in ['close', 'ltp']:
                        column_mapping[col] = 'Close'
                    elif col_lower == 'volume':
                        column_mapping[col] = 'Volume'
                
                df = df.rename(columns=column_mapping)
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])
                    df = df.set_index('Date')
                    df = df.sort_index()
                    
                    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna(subset=['Close'])
                    
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    available_cols = [col for col in required_cols if col in df.columns]
                    
                    if 'Close' in available_cols and len(df) > 0:
                        if 'Volume' in df.columns:
                            available_cols.append('Volume')
                        return df[available_cols]
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# --------------------------
# Streamlit App
# --------------------------

st.title("📊 Live F&O Dashboard - Real-Time WebSocket Data")
st.markdown("**Real-time streaming quotes powered by ICICI Breeze WebSocket API**")

# Connection Status
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.breeze_connected:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")

with col2:
    if st.session_state.ws_connected:
        st.success(f"✅ WebSocket Active ({len(st.session_state.subscribed_stocks)} stocks)")
    else:
        st.warning("⚪ WebSocket Inactive")

with col3:
    if st.session_state.ws_connected:
        if st.button("🔴 Stop Live Data", type="secondary", use_container_width=True):
            stop_websocket()
            st.rerun()
    else:
        if st.button("🟢 Start Live Data", type="primary", use_container_width=True):
            if start_websocket():
                st.success("WebSocket connected!")
                time.sleep(1)
                st.rerun()

st.markdown("---")

# Stock Selection
st.subheader("📈 Select Stocks to Monitor")

col1, col2 = st.columns([3, 1])

with col1:
    selected_stocks = st.multiselect(
        "Choose up to 6 stocks for live monitoring",
        options=sorted(FNO_STOCKS),
        default=st.session_state.selected_stocks,
        max_selections=6,
        key="stock_selector"
    )

with col2:
    if st.button("🔄 Update Subscriptions", type="primary", use_container_width=True):
        if st.session_state.ws_connected:
            # Unsubscribe from removed stocks
            for stock in st.session_state.subscribed_stocks.copy():
                if stock not in selected_stocks:
                    unsubscribe_stock(stock)
            
            # Subscribe to new stocks
            for stock in selected_stocks:
                if stock not in st.session_state.subscribed_stocks:
                    subscribe_stock(stock)
            
            st.session_state.selected_stocks = selected_stocks
            st.success(f"Subscribed to {len(selected_stocks)} stocks!")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("⚠️ Please start WebSocket connection first!")

st.markdown("---")

# Display Live Data
if st.session_state.ws_connected and selected_stocks:
    st.subheader("📊 Live Market Data")
    
    # Auto-refresh placeholder
    placeholder = st.empty()
    
    # Create grid layout
    num_cols = 2 if len(selected_stocks) <= 4 else 3
    
    with placeholder.container():
        rows = (len(selected_stocks) + num_cols - 1) // num_cols
        
        for row in range(rows):
            cols = st.columns(num_cols)
            
            for col_idx in range(num_cols):
                stock_idx = row * num_cols + col_idx
                
                if stock_idx < len(selected_stocks):
                    stock_name = selected_stocks[stock_idx]
                    
                    with cols[col_idx]:
                        # Get live data
                        live_data = st.session_state.live_data.get(stock_name, {})
                        
                        if live_data:
                            ltp = live_data.get('ltp', 0)
                            change = live_data.get('change', 0)
                            open_price = live_data.get('open', 0)
                            high = live_data.get('high', 0)
                            low = live_data.get('low', 0)
                            volume = live_data.get('volume', 0)
                            timestamp = live_data.get('timestamp', '')
                            bid_price = live_data.get('bid_price', 0)
                            ask_price = live_data.get('ask_price', 0)
                            
                            # Card styling
                            color = "green" if change >= 0 else "red"
                            arrow = "🟢" if change >= 0 else "🔴"
                            
                            st.markdown(f"### {arrow} **{stock_name}**")
                            st.metric(
                                "Live Price (LTP)",
                                f"₹{ltp:.2f}",
                                f"{change:.2f}%"
                            )
                            
                            # Additional details
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.caption(f"**Open:** ₹{open_price:.2f}")
                                st.caption(f"**High:** ₹{high:.2f}")
                                st.caption(f"**Low:** ₹{low:.2f}")
                            with col_b:
                                st.caption(f"**Bid:** ₹{bid_price:.2f}")
                                st.caption(f"**Ask:** ₹{ask_price:.2f}")
                                st.caption(f"**Volume:** {volume:,}")
                            
                            st.caption(f"🕐 Last updated: {timestamp}")
                            
                            # Mini chart
                            stock_code = STOCK_CODE_MAP.get(stock_name)
                            if stock_code:
                                df = fetch_stock_data_breeze(stock_code, days=7)
                                if not df.empty and 'Close' in df.columns:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=df.index,
                                        y=df['Close'],
                                        mode='lines',
                                        line=dict(color=color, width=2),
                                        fill='tozeroy',
                                        fillcolor=f'rgba({"0,255,0" if color == "green" else "255,0,0"},0.1)'
                                    ))
                                    fig.update_layout(
                                        height=150,
                                        margin=dict(l=0, r=0, t=0, b=0),
                                        showlegend=False,
                                        xaxis=dict(showgrid=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, showticklabels=False)
                                    )
                                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        else:
                            st.markdown(f"### ⏳ **{stock_name}**")
                            st.info("Waiting for live data...")
                            st.caption("Make sure to click 'Update Subscriptions'")
    
    # Auto-refresh every 2 seconds
    st.markdown("---")
    st.caption("🔄 Dashboard auto-refreshes every 2 seconds")
    time.sleep(2)
    st.rerun()

elif not st.session_state.ws_connected:
    st.info("👆 Click **'Start Live Data'** to begin receiving real-time quotes")
else:
    st.info("👆 Select stocks from the dropdown and click **'Update Subscriptions'**")

# Footer
st.markdown("---")
st.caption("💡 Live streaming dashboard powered by ICICI Breeze WebSocket API")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")
st.caption(f"📡 Subscribed Stocks: {', '.join(st.session_state.subscribed_stocks) if st.session_state.subscribed_stocks else 'None'}")
