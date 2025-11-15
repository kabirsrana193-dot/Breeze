import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from breeze_connect import BreezeConnect

# Page config
st.set_page_config(
    page_title="Nifty F&O Dashboard - Breeze",
    page_icon="📈",
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
# Config - Top F&O Stocks (with correct Breeze codes)
# --------------------------
FNO_STOCKS = [
    "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Bharti Airtel", "ITC",
    "State Bank of India", "Hindustan Unilever", "Bajaj Finance", 
    "Kotak Mahindra Bank", "Axis Bank", "Larsen & Toubro", "Asian Paints", 
    "Maruti Suzuki", "Titan", "Sun Pharma", "HCL Tech", "Adani Enterprises",
    "Tata Motors", "Wipro", "NTPC", "Bajaj Finserv", "Tata Steel",
    "Hindalco", "IndusInd Bank", "Mahindra & Mahindra", "Coal India",
    "JSW Steel", "Tata Consumer", "Eicher Motors", "BPCL", "Tech Mahindra",
    "Dr Reddy", "Cipla", "UPL", "Britannia"
]

# Verified Breeze stock codes (use breeze.get_names() to verify)
STOCK_CODE_MAP = {
    "Reliance": "RELIND",
    "TCS": "TCS",
    "HDFC Bank": "HDFCBK",
    "Infosys": "INFOSYSTCH",
    "ICICI Bank": "ICIBK",
    "Bharti Airtel": "BHARTIARTL",
    "ITC": "ITC",
    "State Bank of India": "SBIN",
    "Hindustan Unilever": "HINDUNILVR",
    "Bajaj Finance": "BAJFINANCE",
    "Kotak Mahindra Bank": "KOTAKBANK",
    "Axis Bank": "AXSB",
    "Larsen & Toubro": "LT",
    "Asian Paints": "ASIANPAINT",
    "Maruti Suzuki": "MARUTI",
    "Titan": "TITAN",
    "Sun Pharma": "SUNPHARMA",
    "HCL Tech": "HCLTECH",
    "Adani Enterprises": "ADANIENT",
    "Tata Motors": "TATAMOTORS",
    "Wipro": "WIPRO",
    "NTPC": "NTPC",
    "Bajaj Finserv": "BAJAJFINSV",
    "Tata Steel": "TATASTL",
    "Hindalco": "HINDALCO",
    "IndusInd Bank": "INDUSINDBK",
    "Mahindra & Mahindra": "M&M",
    "Coal India": "COALINDIA",
    "JSW Steel": "JSWSTEEL",
    "Tata Consumer": "TATACONSUM",
    "Eicher Motors": "EICHERMOT",
    "BPCL": "BPCL",
    "Tech Mahindra": "TECHM",
    "Dr Reddy": "DRREDDY",
    "Cipla": "CIPLA",
    "UPL": "UPL",
    "Britannia": "BRITANNIA"
}

FINANCIAL_RSS_FEEDS = [
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

ARTICLES_PER_REFRESH = 12

# --------------------------
# Initialize session state
# --------------------------
if 'breeze_client' not in st.session_state:
    st.session_state.breeze_client = None
if 'breeze_connected' not in st.session_state:
    st.session_state.breeze_connected = False
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "All Stocks"
if 'technical_data' not in st.session_state:
    st.session_state.technical_data = []
if 'watchlist_stocks' not in st.session_state:
    st.session_state.watchlist_stocks = [
        "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank"
    ]
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'api_call_count' not in st.session_state:
    st.session_state.api_call_count = 0

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
# Stock Data Functions
# --------------------------
@st.cache_data(ttl=300)
def fetch_stock_data_breeze(stock_code, days=90, interval="1day"):
    """Fetch historical stock data using Breeze API"""
    try:
        breeze = st.session_state.breeze_client
        if not breeze:
            return pd.DataFrame()
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Breeze API requires ISO 8601 format with .000Z
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
        
        # Rate limiting: 100 calls/min = 0.6s minimum between calls
        st.session_state.api_call_count += 1
        time.sleep(0.7)
        
        if response and 'Success' in response:
            data = response['Success']
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                
                # Map column names
                column_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'datetime' in col_lower or 'date' in col_lower:
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
                    
                    # Convert to numeric
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
# Technical Analysis Functions
# --------------------------
def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return pd.Series(index=data.index)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    try:
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    except:
        return pd.Series(index=data.index), pd.Series(index=data.index)

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    try:
        return data.rolling(window=period).mean()
    except:
        return pd.Series(index=data.index)

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    try:
        return data.ewm(span=period, adjust=False).mean()
    except:
        return pd.Series(index=data.index)

def generate_signal_enhanced(stock_code):
    """Generate buy/sell signal with 0-10 scoring"""
    try:
        df = fetch_stock_data_breeze(stock_code, 90, "1day")
        
        if df.empty or len(df) < 50:
            return None
        
        # Calculate all indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['EMA_12'] = calculate_ema(df['Close'], 12)
        df['EMA_26'] = calculate_ema(df['Close'], 26)
        
        df_clean = df.dropna(subset=['Close', 'RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50'])
        if df_clean.empty:
            return None
        
        current_price = df_clean['Close'].iloc[-1]
        rsi = df_clean['RSI'].iloc[-1]
        macd = df_clean['MACD'].iloc[-1]
        macd_signal = df_clean['MACD_Signal'].iloc[-1]
        sma_20 = df_clean['SMA_20'].iloc[-1]
        sma_50 = df_clean['SMA_50'].iloc[-1]
        ema_12 = df_clean['EMA_12'].iloc[-1]
        ema_26 = df_clean['EMA_26'].iloc[-1]
        
        # Enhanced scoring system (0-10)
        score = 5.0  # Start neutral
        signals = []
        
        # RSI Analysis (±2.5 points)
        if pd.notna(rsi):
            if rsi < 20:
                signals.append("RSI Extremely Oversold")
                score += 2.5
            elif rsi < 30:
                signals.append("RSI Oversold")
                score += 1.5
            elif rsi > 80:
                signals.append("RSI Extremely Overbought")
                score -= 2.5
            elif rsi > 70:
                signals.append("RSI Overbought")
                score -= 1.5
        
        # MACD Analysis (±2 points)
        if pd.notna(macd) and pd.notna(macd_signal):
            macd_diff = macd - macd_signal
            if macd > macd_signal:
                if macd_diff > 0.5:
                    signals.append("MACD Strong Bullish")
                    score += 2.0
                else:
                    signals.append("MACD Bullish")
                    score += 1.0
            else:
                if macd_diff < -0.5:
                    signals.append("MACD Strong Bearish")
                    score -= 2.0
                else:
                    signals.append("MACD Bearish")
                    score -= 1.0
        
        # Moving Average Analysis (±2.5 points)
        ma_score = 0
        if pd.notna(sma_20) and pd.notna(sma_50):
            # Golden Cross / Death Cross
            if sma_20 > sma_50:
                signals.append("Golden Cross (SMA)")
                ma_score += 1.5
            elif sma_20 < sma_50:
                signals.append("Death Cross (SMA)")
                ma_score -= 1.5
            
            # Price vs Moving Averages
            if current_price > sma_20 and current_price > sma_50:
                signals.append("Price Above MAs")
                ma_score += 1.0
            elif current_price < sma_20 and current_price < sma_50:
                signals.append("Price Below MAs")
                ma_score -= 1.0
        
        score += ma_score
        
        # EMA Analysis (±1 point)
        if pd.notna(ema_12) and pd.notna(ema_26):
            if ema_12 > ema_26:
                signals.append("EMA Bullish")
                score += 0.5
            else:
                signals.append("EMA Bearish")
                score -= 0.5
        
        # Momentum Analysis (±2 points)
        if len(df_clean) >= 10:
            price_10_ago = df_clean['Close'].iloc[-10]
            momentum = ((current_price - price_10_ago) / price_10_ago) * 100
            if momentum > 5:
                signals.append(f"Strong Uptrend (+{momentum:.1f}%)")
                score += 2.0
            elif momentum > 2:
                signals.append(f"Uptrend (+{momentum:.1f}%)")
                score += 1.0
            elif momentum < -5:
                signals.append(f"Strong Downtrend ({momentum:.1f}%)")
                score -= 2.0
            elif momentum < -2:
                signals.append(f"Downtrend ({momentum:.1f}%)")
                score -= 1.0
        
        # Clamp score between 0 and 10
        score = max(0, min(10, score))
        
        # Generate recommendation based on score
        if score >= 8:
            recommendation = "🟢 STRONG BUY"
            emoji = "🟢"
        elif score >= 6.5:
            recommendation = "🟡 BUY"
            emoji = "🟡"
        elif score >= 4.5:
            recommendation = "⚪ HOLD"
            emoji = "⚪"
        elif score >= 3:
            recommendation = "🟠 SELL"
            emoji = "🟠"
        else:
            recommendation = "🔴 STRONG SELL"
            emoji = "🔴"
        
        return {
            'price': current_price,
            'score': score,
            'recommendation': recommendation,
            'emoji': emoji,
            'rsi': rsi,
            'macd': macd,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'signals': ' | '.join(signals) if signals else 'No signals'
        }
    except Exception as e:
        return None

# --------------------------
# Sentiment Analysis
# --------------------------
def analyze_sentiment(text):
    """Fast keyword-based sentiment analysis"""
    POSITIVE = ['surge', 'rally', 'gain', 'profit', 'growth', 'rise', 'bullish', 
                'strong', 'beats', 'outperform', 'jumps', 'soars', 'upgrade', 
                'breakthrough', 'record', 'momentum', 'recovery']
    
    NEGATIVE = ['fall', 'drop', 'loss', 'decline', 'weak', 'crash', 'bearish',
                'concern', 'risk', 'plunge', 'slump', 'miss', 'downgrade', 
                'warning', 'crisis', 'tumbles', 'worst']
    
    text_lower = text.lower()
    pos_count = sum(1 for w in POSITIVE if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE if w in text_lower)
    
    if pos_count > neg_count:
        return "positive", min(0.6 + pos_count * 0.1, 0.95)
    elif neg_count > pos_count:
        return "negative", min(0.6 + neg_count * 0.1, 0.95)
    else:
        return "neutral", 0.5

# --------------------------
# News Functions
# --------------------------
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

# --------------------------
# Streamlit App
# --------------------------

# API Call Counter Warning
if st.session_state.api_call_count > 4500:
    st.warning(f"⚠️ API calls today: {st.session_state.api_call_count}/5000. Approaching daily limit!")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📰 News", "💹 Charts & Indicators", "⚡ Intraday Monitor"])

# TAB 1: NEWS
with tab1:
    st.title("📈 F&O News Dashboard (Breeze API)")
    st.markdown(f"Track {len(FNO_STOCKS)} F&O stocks | Powered by ICICI Breeze")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        stock_options = ["All Stocks"] + sorted(FNO_STOCKS[:30])
        selected_stock = st.selectbox(
            "🔍 Filter by Stock",
            options=stock_options,
            index=stock_options.index(st.session_state.selected_stock),
            key="stock_filter"
        )
        st.session_state.selected_stock = selected_stock

    with col2:
        if st.button("🔄 Refresh News", type="primary", use_container_width=True, key="refresh_news_btn"):
            with st.spinner("Fetching updates..."):
                new_articles = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
                st.session_state.news_articles = new_articles
                st.session_state.last_refresh = datetime.now()
                st.success(f"✅ Loaded {len(new_articles)} articles!")

    with col3:
        if st.button("🗑 Clear", use_container_width=True, key="clear_news_btn"):
            st.session_state.news_articles = []
            st.rerun()

    if not st.session_state.news_articles:
        with st.spinner("Loading news..."):
            st.session_state.news_articles = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
            st.session_state.last_refresh = datetime.now()

    if st.session_state.news_articles:
        df_all = pd.DataFrame(st.session_state.news_articles)
        
        st.subheader("📊 Sentiment Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total", len(df_all))
        with col2:
            st.metric("🟢 Positive", len(df_all[df_all['Sentiment'] == 'positive']))
        with col3:
            st.metric("⚪ Neutral", len(df_all[df_all['Sentiment'] == 'neutral']))
        with col4:
            st.metric("🔴 Negative", len(df_all[df_all['Sentiment'] == 'negative']))
        
        st.markdown("---")
        
        sentiment_counts = df_all['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        
        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📰 Latest Articles")
        
        for article in st.session_state.news_articles:
            sentiment_colors = {"positive": "#28a745", "neutral": "#6c757d", "negative": "#dc3545"}
            sentiment_emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
            
            st.markdown(f"**[{article['Title']}]({article['Link']})**")
            st.markdown(
                f"<span style='background-color: {sentiment_colors[article['Sentiment']]}; "
                f"color: white; padding: 3px 10px; border-radius: 4px; font-size: 11px;'>"
                f"{sentiment_emoji[article['Sentiment']]} {article['Sentiment'].upper()} "
                f"({article['Score']:.2f})</span>",
                unsafe_allow_html=True
            )
            st.caption(f"Source: {article['Source']} | {article['Published']}")
            st.markdown("---")
    else:
        st.info("No articles found. Click 'Refresh News' to load.")

# TAB 2: STOCK CHARTS WITH SMA/EMA
with tab2:
    st.title("💹 Stock Charts with SMA & EMA")
    st.markdown("Candlestick charts with technical indicators")
    
    if not st.session_state.breeze_connected:
        st.error("⚠️ Breeze API not connected.")
    else:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_chart_stock = st.selectbox(
                "📊 Select Stock",
                options=sorted(FNO_STOCKS),
                key="chart_stock"
            )
        
        with col2:
            period_options = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90}
            period_label = st.selectbox("📅 Period", options=list(period_options.keys()), index=3, key="chart_period")
            period_days = period_options[period_label]
        
        col3, col4 = st.columns(2)
        with col3:
            interval_options = {
                "Daily": "1day",
                "30 Minutes": "30minute", 
                "5 Minutes": "5minute",
                "1 Minute": "1minute"
            }
            interval_label = st.selectbox("⏱️ Interval", options=list(interval_options.keys()), index=0, key="chart_interval")
            interval = interval_options[interval_label]
        
        if interval != "1day" and period_days > 5:
            st.info("ℹ️ Intraday data limited to 5 days.")
            period_days = 5
        
        stock_code = STOCK_CODE_MAP.get(selected_chart_stock)
        
        if stock_code:
            with st.spinner(f"Loading data for {selected_chart_stock}..."):
                df = fetch_stock_data_breeze(stock_code, period_days, interval)
            
            if not df.empty and 'Close' in df.columns and len(df) > 0:
                # Calculate all indicators
                df['RSI'] = calculate_rsi(df['Close'])
                df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
                df['SMA_20'] = calculate_sma(df['Close'], 20)
                df['SMA_50'] = calculate_sma(df['Close'], 50)
                df['EMA_12'] = calculate_ema(df['Close'], 12)
                df['EMA_26'] = calculate_ema(df['Close'], 26)
                
                df_clean = df.dropna(subset=['Close'])
                
                if len(df_clean) >= 2:
                    current_price = df_clean['Close'].iloc[-1]
                    price_change = df_clean['Close'].iloc[-1] - df_clean['Close'].iloc[0]
                    price_change_pct = (price_change / df_clean['Close'].iloc[0]) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current", f"₹{current_price:.2f}")
                    with col2:
                        st.metric("Change", f"₹{price_change:.2f}", f"{price_change_pct:.2f}%")
                    with col3:
                        st.metric("High", f"₹{df_clean['High'].max():.2f}" if 'High' in df_clean.columns else "N/A")
                    with col4:
                        st.metric("Low", f"₹{df_clean['Low'].min():.2f}" if 'Low' in df_clean.columns else "N/A")
                    
                    st.markdown("---")
                    
                    # Candlestick chart with SMA/EMA
                    if all(col in df_clean.columns for col in ['Open', 'High', 'Low', 'Close']):
                        fig = go.Figure()
                        
                        # Add candlestick
                        fig.add_trace(go.Candlestick(
                            x=df_clean.index,
                            open=df_clean['Open'],
                            high=df_clean['High'],
                            low=df_clean['Low'],
                            close=df_clean['Close'],
                            name='Price'
                        ))
                        
                        # Add SMA lines
                        if 'SMA_20' in df_clean.columns:
                            df_sma20 = df_clean.dropna(subset=['SMA_20'])
                            fig.add_trace(go.Scatter(
                                x=df_sma20.index,
                                y=df_sma20['SMA_20'],
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='blue', width=1.5)
                            ))
                        
                        if 'SMA_50' in df_clean.columns:
                            df_sma50 = df_clean.dropna(subset=['SMA_50'])
                            fig.add_trace(go.Scatter(
                                x=df_sma50.index,
                                y=df_sma50['SMA_50'],
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='orange', width=1.5)
                            ))
                        
                        # Add EMA lines
                        if 'EMA_12' in df_clean.columns:
                            df_ema12 = df_clean.dropna(subset=['EMA_12'])
                            fig.add_trace(go.Scatter(
                                x=df_ema12.index,
                                y=df_ema12['EMA_12'],
                                mode='lines',
                                name='EMA 12',
                                line=dict(color='green', width=1.5, dash='dash')
                            ))
                        
                        if 'EMA_26' in df_clean.columns:
                            df_ema26 = df_clean.dropna(subset=['EMA_26'])
                            fig.add_trace(go.Scatter(
                                x=df_ema26.index,
                                y=df_ema26['EMA_26'],
                                mode='lines',
                                name='EMA 26',
                                line=dict(color='red', width=1.5, dash='dash')
                            ))
                        
                        fig.update_layout(
                            title=f"{selected_chart_stock} - {interval_label} Chart with SMA & EMA",
                            xaxis_title="Date/Time",
                            yaxis_title="Price (₹)",
                            height=500,
                            xaxis_rangeslider_visible=False,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # RSI chart
                    df_rsi = df_clean.dropna(subset=['RSI'])
                    if len(df_rsi) > 0:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(
                            x=df_rsi.index, 
                            y=df_rsi['RSI'], 
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                        fig_rsi.update_layout(
                            title="RSI Indicator", 
                            height=250, 
                            yaxis_range=[0, 100],
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # MACD chart
                    df_macd = df_clean.dropna(subset=['MACD', 'MACD_Signal'])
                    if len(df_macd) > 0:
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(
                            x=df_macd.index, 
                            y=df_macd['MACD'], 
                            name='MACD', 
                            line=dict(color='blue', width=2)
                        ))
                        fig_macd.add_trace(go.Scatter(
                            x=df_macd.index, 
                            y=df_macd['MACD_Signal'], 
                            name='Signal', 
                            line=dict(color='red', width=2)
                        ))
                        
                        # Add histogram (MACD - Signal)
                        histogram = df_macd['MACD'] - df_macd['MACD_Signal']
                        colors = ['green' if val >= 0 else 'red' for val in histogram]
                        fig_macd.add_trace(go.Bar(
                            x=df_macd.index,
                            y=histogram,
                            name='Histogram',
                            marker_color=colors,
                            opacity=0.3
                        ))
                        
                        fig_macd.update_layout(
                            title="MACD Indicator", 
                            height=250,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_macd, use_container_width=True)
                else:
                    st.warning(f"⚠️ Insufficient data for {selected_chart_stock}")
            else:
                st.error(f"⚠️ Could not fetch data for {selected_chart_stock}")
                st.info("💡 This may be due to incorrect stock code. Try using breeze.get_names() to verify the stock code.")
        else:
            st.error(f"⚠️ Stock code not found for {selected_chart_stock}")

# TAB 3: INTRADAY MULTI-CHART MONITOR
with tab3:
    st.title("⚡ Intraday Multi-Chart Monitor")
    st.markdown("Real-time intraday candlestick charts for active trading")
    
    if not st.session_state.breeze_connected:
        st.error("⚠️ Breeze API not connected.")
    else:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_watchlist = st.multiselect(
                "Select stocks for intraday monitoring",
                options=sorted(FNO_STOCKS),
                default=st.session_state.watchlist_stocks[:4],
                max_selections=6,
                key="multi_watchlist"
            )
        
        with col2:
            interval_options_intraday = {
                "1 Minute": "1minute",
                "5 Minutes": "5minute",
                "30 Minutes": "30minute"
            }
            intraday_interval_label = st.selectbox(
                "Chart Interval", 
                options=list(interval_options_intraday.keys()), 
                index=1, 
                key="intraday_interval"
            )
            intraday_interval = interval_options_intraday[intraday_interval_label]
        
        with col3:
            if st.button("🔄 Refresh Charts", type="primary", use_container_width=True, key="multi_refresh_btn"):
                st.cache_data.clear()
                st.rerun()
        
        if selected_watchlist:
            st.info(f"📊 Displaying {len(selected_watchlist)} stocks with {intraday_interval_label} candles")
            
            # Always use 1 day for intraday
            intraday_days = 1
            
            num_cols = 2 if len(selected_watchlist) <= 4 else 3
            num_rows = (len(selected_watchlist) + num_cols - 1) // num_cols
            
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col_idx, col in enumerate(cols):
                    stock_idx = row * num_cols + col_idx
                    
                    if stock_idx < len(selected_watchlist):
                        stock_name = selected_watchlist[stock_idx]
                        stock_code = STOCK_CODE_MAP.get(stock_name)
                        
                        with col:
                            if stock_code:
                                with st.spinner(f"Loading {stock_name}..."):
                                    df = fetch_stock_data_breeze(stock_code, intraday_days, intraday_interval)
                                
                                if not df.empty and len(df) >= 2 and 'Close' in df.columns:
                                    df_clean = df.dropna(subset=['Close'])
                                    
                                    if len(df_clean) >= 2:
                                        current = df_clean['Close'].iloc[-1]
                                        prev = df_clean['Close'].iloc[0]
                                        
                                        if pd.notna(current) and pd.notna(prev) and prev != 0:
                                            change = current - prev
                                            change_pct = (change / prev) * 100
                                            arrow = "🟢" if change_pct >= 0 else "🔴"
                                            
                                            st.markdown(f"### {arrow} {stock_name}")
                                            st.metric(
                                                "Price", 
                                                f"₹{current:.2f}", 
                                                f"{change:.2f} ({change_pct:.2f}%)"
                                            )
                                            
                                            # Create candlestick chart
                                            if all(col in df_clean.columns for col in ['Open', 'High', 'Low', 'Close']):
                                                fig = go.Figure(data=[go.Candlestick(
                                                    x=df_clean.index,
                                                    open=df_clean['Open'],
                                                    high=df_clean['High'],
                                                    low=df_clean['Low'],
                                                    close=df_clean['Close'],
                                                    increasing_line_color='green',
                                                    decreasing_line_color='red'
                                                )])
                                                
                                                fig.update_layout(
                                                    height=300,
                                                    margin=dict(l=10, r=10, t=30, b=10),
                                                    showlegend=False,
                                                    xaxis=dict(
                                                        showgrid=True, 
                                                        gridcolor='lightgray',
                                                        type='date'
                                                    ),
                                                    yaxis=dict(
                                                        showgrid=True, 
                                                        gridcolor='lightgray'
                                                    ),
                                                    xaxis_rangeslider_visible=False,
                                                    hovermode='x'
                                                )
                                                
                                                st.plotly_chart(
                                                    fig, 
                                                    use_container_width=True, 
                                                    config={'displayModeBar': False}
                                                )
                                                
                                                # Show volume if available
                                                if 'Volume' in df_clean.columns:
                                                    total_vol = df_clean['Volume'].sum()
                                                    st.caption(f"📊 Volume: {total_vol:,.0f}")
                                            else:
                                                # Fallback to line chart
                                                fig = go.Figure()
                                                fig.add_trace(go.Scatter(
                                                    x=df_clean.index,
                                                    y=df_clean['Close'],
                                                    mode='lines',
                                                    line=dict(
                                                        color='green' if change_pct >= 0 else 'red', 
                                                        width=2
                                                    ),
                                                    fill='tozeroy',
                                                    fillcolor=f'rgba({"0,255,0" if change_pct >= 0 else "255,0,0"},0.1)'
                                                ))
                                                fig.update_layout(
                                                    height=300,
                                                    margin=dict(l=10, r=10, t=10, b=10),
                                                    showlegend=False,
                                                    xaxis=dict(showgrid=False),
                                                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                                                )
                                                st.plotly_chart(
                                                    fig, 
                                                    use_container_width=True, 
                                                    config={'displayModeBar': False}
                                                )
                                        else:
                                            st.warning(f"⚠️ Invalid data for {stock_name}")
                                    else:
                                        st.warning(f"⚠️ Insufficient data for {stock_name}")
                                else:
                                    st.warning(f"⚠️ No intraday data for {stock_name}")
                                    st.caption("Try a different stock or check if market is open")
                            else:
                                st.error(f"⚠️ Stock code not found for {stock_name}")
        else:
            st.info("👆 Select stocks from the dropdown to monitor intraday movements")
            st.markdown("""
            ### 💡 Intraday Trading Features:
            - **Real-time candlestick charts** with 1min, 5min, or 30min intervals
            - **Live price updates** with percentage change
            - **Multiple stocks** monitoring (up to 6 simultaneously)
            - **Volume indicators** for each stock
            - Perfect for day trading and scalping strategies
            """)

# Footer
st.markdown("---")
st.caption("💡 F&O Dashboard powered by ICICI Breeze API | Enhanced Technical Analysis with 0-10 Scoring")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")

# Connection status
if st.session_state.breeze_connected:
    st.caption(f"🔌 **Status:** ✅ Connected | API Calls: {st.session_state.api_call_count}/5000 daily limit")
else:
    st.caption("🔌 **Status:** ❌ Disconnected - Check credentials in code")
