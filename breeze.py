import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache
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
# IMPORTANT: Replace these with your actual credentials
BREEZE_CONFIG = {
    'app_key': "68`47N89970w1dH7u1s5347j8403f287",
    'secret_key': "5v9k141093cf4361528$z24Q7(Yv2839",
    'session_token': "53705299"
}

# --------------------------
# Config - Top F&O Stocks
# --------------------------
FNO_STOCKS = [
    "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Bharti Airtel", "ITC",
    "State Bank of India", "SBI", "Hindustan Unilever", "HUL", "Bajaj Finance", 
    "Kotak Mahindra Bank", "Axis Bank", "Larsen & Toubro", "L&T", "Asian Paints", 
    "Maruti Suzuki", "Titan", "Sun Pharma", "HCL Tech", "Adani Enterprises",
    "Tata Motors", "Wipro", "NTPC", "Bajaj Finserv", "Tata Steel",
    "Hindalco", "IndusInd Bank", "Mahindra & Mahindra", "M&M", "Coal India",
    "JSW Steel", "Tata Consumer", "Eicher Motors", "BPCL", "Tech Mahindra",
    "Dr Reddy", "Cipla", "UPL", "Britannia", "Divi's Lab", "SBI Life",
    "HDFC Life", "Adani Ports", "ONGC", "IOC", "Vedanta", "Bajaj Auto", 
    "Hero MotoCorp", "GAIL", "UltraTech", "Zomato", "Trent", "DMart",
    "Apollo Hospitals", "Lupin", "DLF", "Bank of Baroda", "Canara Bank",
    "Federal Bank", "InterGlobe Aviation", "Adani Green", "Siemens",
    "Bharat Electronics", "BEL", "HAL", "Shriram Finance", "IRCTC"
]

# Stock code mapping for Breeze API
STOCK_CODE_MAP = {
    "Reliance": "RELIND", "TCS": "TCS", "HDFC Bank": "HDFCBK",
    "Infosys": "INFOSYSTCH", "ICICI Bank": "ICIBK", "Bharti Airtel": "BHARTIARTL",
    "ITC": "ITC", "State Bank of India": "SBIN", "SBI": "SBIN",
    "Hindustan Unilever": "HINDUNILVR", "HUL": "HINDUNILVR",
    "Bajaj Finance": "BAJAJFINSV", "Kotak Mahindra Bank": "KOTAKBANK",
    "Axis Bank": "AXSB", "Larsen & Toubro": "LT", "L&T": "LT",
    "Asian Paints": "ASIANPAINT", "Maruti Suzuki": "MARUTI",
    "Titan": "TITAN", "Sun Pharma": "SUNPHARMA", "HCL Tech": "HCLTECH",
    "Adani Enterprises": "ADANIENT", "Tata Motors": "TATAMOTORS",
    "Wipro": "WIPRO", "NTPC": "NTPC", "Bajaj Finserv": "BAJFINANCE",
    "Tata Steel": "TATASTL", "Hindalco": "HINDALCO",
    "IndusInd Bank": "INDUSINDBK", "Mahindra & Mahindra": "M&M",
    "M&M": "M&M", "Coal India": "COALINDIA", "JSW Steel": "JSWSTEEL",
    "Tata Consumer": "TATACONSUM", "Eicher Motors": "EICHERMOT",
    "BPCL": "BPCL", "Tech Mahindra": "TECHM", "Dr Reddy": "DRREDDY",
    "Cipla": "CIPLA", "UPL": "UPL", "Britannia": "BRITANNIA",
    "Divi's Lab": "DIVISLAB", "ONGC": "ONGC", "IOC": "IOC",
    "Vedanta": "VEDL", "Bajaj Auto": "BAJAJAUTO", "SBI Life": "SBILIFE",
    "HDFC Life": "HDFCLIFE", "Adani Ports": "ADANIPORTS",
    "UltraTech": "ULTRACEMCO", "Hero MotoCorp": "HEROMOTOCO",
    "GAIL": "GAIL", "Zomato": "ZOMATO", "Trent": "TRENT",
    "DMart": "DMART", "Apollo Hospitals": "APOLLOHOSP",
    "Lupin": "LUPIN", "DLF": "DLF", "Bank of Baroda": "BANKBARODA",
    "Canara Bank": "CANBK", "Federal Bank": "FEDERALBNK",
    "InterGlobe Aviation": "INDIGO", "Adani Green": "ADANIGREEN",
    "Siemens": "SIEMENS", "Bharat Electronics": "BEL", "BEL": "BEL",
    "HAL": "HAL", "Shriram Finance": "SHRIRAMFIN", "IRCTC": "IRCTC"
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

# --------------------------
# Breeze Connection
# --------------------------
def connect_breeze():
    """Initialize Breeze connection"""
    try:
        breeze = BreezeConnect(api_key=BREEZE_CONFIG['app_key'])
        breeze.generate_session(
            api_secret=BREEZE_CONFIG['secret_key'],
            session_token=BREEZE_CONFIG['session_token']
        )
        return breeze, True
    except Exception as e:
        st.error(f"❌ Breeze connection failed: {str(e)}")
        return None, False

# Connect to Breeze on first load
if not st.session_state.breeze_connected:
    with st.spinner("🔌 Connecting to Breeze API..."):
        breeze, success = connect_breeze()
        if success:
            st.session_state.breeze_client = breeze
            st.session_state.breeze_connected = True
            st.success("✅ Connected to Breeze API!")
            time.sleep(1)
        else:
            st.error("⚠️ Could not connect to Breeze. Please check your credentials.")

# --------------------------
# Stock Data Functions using Breeze
# --------------------------
@st.cache_data(ttl=300)
def fetch_stock_data_breeze(stock_code, days=90):
    """Fetch historical stock data using Breeze API"""
    try:
        breeze = st.session_state.breeze_client
        if not breeze:
            return pd.DataFrame()
        
        # Calculate dates
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Format dates for Breeze API (ISO8601)
        from_date_str = from_date.strftime("%Y-%m-%dT00:00:00.000Z")
        to_date_str = to_date.strftime("%Y-%m-%dT23:59:59.000Z")
        
        # Fetch historical data
        response = breeze.get_historical_data_v2(
            interval="1day",
            from_date=from_date_str,
            to_date=to_date_str,
            stock_code=stock_code,
            exchange_code="NSE",
            product_type="cash"
        )
        
        if response and 'Success' in response:
            data = response['Success']
            if data:
                df = pd.DataFrame(data)
                # Rename columns to match expected format
                df = df.rename(columns={
                    'datetime': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not fetch data for {stock_code}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_live_quote(stock_code):
    """Get live quote for a stock"""
    try:
        breeze = st.session_state.breeze_client
        if not breeze:
            return None
        
        response = breeze.get_quotes(
            stock_code=stock_code,
            exchange_code="NSE",
            expiry_date="",
            product_type="",
            right="",
            strike_price=""
        )
        
        if response and 'Success' in response:
            return response['Success']
        return None
    except:
        return None

# --------------------------
# Technical Analysis Functions
# --------------------------
def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def generate_signal(stock_code):
    """Generate buy/sell signal"""
    try:
        df = fetch_stock_data_breeze(stock_code, 90)
        
        if df.empty or len(df) < 50:
            return None
        
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = calculate_macd(df['Close'])
        
        current_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal_line = df['Signal'].iloc[-1]
        
        score = 0
        signals = []
        
        if rsi < 30:
            signals.append("RSI Oversold")
            score += 2
        elif rsi > 70:
            signals.append("RSI Overbought")
            score -= 2
        
        if macd > signal_line:
            signals.append("MACD Bullish")
            score += 1
        else:
            signals.append("MACD Bearish")
            score -= 1
        
        if score >= 2:
            recommendation = "🟢 STRONG BUY"
        elif score >= 1:
            recommendation = "🟡 BUY"
        elif score <= -2:
            recommendation = "🔴 STRONG SELL"
        elif score <= -1:
            recommendation = "🟠 SELL"
        else:
            recommendation = "⚪ HOLD"
        
        return {
            'price': current_price,
            'rsi': rsi,
            'macd': macd,
            'signals': ', '.join(signals),
            'recommendation': recommendation,
            'score': score
        }
    except:
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

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📰 News", "📈 Technical", "💹 Charts", "📊 Multi-Chart"])

# --------------------------
# TAB 1: NEWS DASHBOARD
# --------------------------
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

    with col2:
        if st.button("🔄 Refresh News", type="primary", use_container_width=True):
            with st.spinner("Fetching updates..."):
                new_articles = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
                st.session_state.news_articles = new_articles
                st.session_state.last_refresh = datetime.now()
                st.success(f"✅ Loaded {len(new_articles)} articles!")
                time.sleep(0.5)
                st.rerun()

    with col3:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.news_articles = []
            st.rerun()

    # Load initial news if empty
    if not st.session_state.news_articles:
        with st.spinner("Loading news..."):
            st.session_state.news_articles = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
            st.session_state.last_refresh = datetime.now()

    if st.session_state.news_articles:
        df_all = pd.DataFrame(st.session_state.news_articles)
        
        # Metrics
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
        
        # Chart
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
        
        # Display articles
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

# --------------------------
# TAB 2: TECHNICAL ANALYSIS
# --------------------------
with tab2:
    st.title("📈 Technical Analysis (Breeze)")
    st.markdown("Buy/Sell signals based on RSI & MACD")
    
    if not st.session_state.breeze_connected:
        st.error("⚠️ Breeze API not connected. Please check credentials.")
    else:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_stocks = st.selectbox(
                "📊 Stocks to Analyze",
                options=[5, 10, 15, 20],
                index=1
            )
        
        with col2:
            if st.button("🔄 Run Analysis", type="primary", use_container_width=True):
                st.session_state.technical_data = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                stocks_to_analyze = FNO_STOCKS[:num_stocks]
                
                for idx, stock_name in enumerate(stocks_to_analyze):
                    stock_code = STOCK_CODE_MAP.get(stock_name)
                    if not stock_code:
                        continue
                    
                    status_text.text(f"Analyzing {stock_name}... ({idx+1}/{num_stocks})")
                    
                    signal_data = generate_signal(stock_code)
                    if signal_data:
                        signal_data['stock'] = stock_name
                        st.session_state.technical_data.append(signal_data)
                    
                    progress_bar.progress((idx + 1) / num_stocks)
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Analyzed {len(st.session_state.technical_data)} stocks!")
                st.rerun()
        
        if st.session_state.technical_data:
            df_tech = pd.DataFrame(st.session_state.technical_data)
            
            # Summary metrics
            st.subheader("📊 Signal Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🟢 Strong Buy", len(df_tech[df_tech['recommendation'] == '🟢 STRONG BUY']))
            with col2:
                st.metric("🟡 Buy", len(df_tech[df_tech['recommendation'] == '🟡 BUY']))
            with col3:
                st.metric("🟠 Sell", len(df_tech[df_tech['recommendation'] == '🟠 SELL']))
            with col4:
                st.metric("🔴 Strong Sell", len(df_tech[df_tech['recommendation'] == '🔴 STRONG SELL']))
            
            st.markdown("---")
            
            # Display results
            df_tech = df_tech.sort_values('score', ascending=False)
            
            for _, row in df_tech.iterrows():
                with st.expander(f"{row['recommendation']} - {row['stock']} @ ₹{row['price']:.2f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Price:** ₹{row['price']:.2f}")
                        st.markdown(f"**RSI:** {row['rsi']:.2f}")
                    with col2:
                        st.markdown(f"**MACD:** {row['macd']:.4f}")
                        st.markdown(f"**Score:** {row['score']}")
                    st.markdown(f"**Signals:** {row['signals']}")
        else:
            st.info("👆 Click 'Run Analysis' to generate signals")

# --------------------------
# TAB 3: STOCK CHARTS
# --------------------------
with tab3:
    st.title("💹 Stock Charts (Breeze)")
    st.markdown("Candlestick charts with technical indicators")
    
    if not st.session_state.breeze_connected:
        st.error("⚠️ Breeze API not connected. Please check credentials.")
    else:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_chart_stock = st.selectbox(
                "📊 Select Stock",
                options=sorted(FNO_STOCKS[:30]),
                key="chart_stock"
            )
        
        with col2:
            period_options = {"1 Month": 30, "3 Months": 90, "6 Months": 180}
            period_label = st.selectbox("📅 Period", options=list(period_options.keys()), index=1)
            period_days = period_options[period_label]
        
        stock_code = STOCK_CODE_MAP.get(selected_chart_stock)
        
        if stock_code:
            df = fetch_stock_data_breeze(stock_code, period_days)
            
            if not df.empty:
                # Calculate indicators
                df['RSI'] = calculate_rsi(df['Close'])
                df['MACD'], df['Signal'] = calculate_macd(df['Close'])
                
                current_price = df['Close'].iloc[-1]
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
                price_change_pct = (price_change / df['Close'].iloc[0]) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Change", f"₹{price_change:.2f}", f"{price_change_pct:.2f}%")
                with col3:
                    st.metric("High", f"₹{df['High'].max():.2f}")
                with col4:
                    st.metric("Low", f"₹{df['Low'].min():.2f}")
                
                st.markdown("---")
                
                # Candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                )])
                fig.update_layout(
                    title=f"{selected_chart_stock} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI", height=250)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal'))
                fig_macd.update_layout(title="MACD", height=250)
                st.plotly_chart(fig_macd, use_container_width=True)
            else:
                st.warning(f"⚠️ Could not fetch data for {selected_chart_stock}")

# --------------------------
# TAB 4: MULTI-CHART
# --------------------------
with tab4:
    st.title("📊 Multi-Chart Monitor (Breeze)")
    st.markdown("Track multiple stocks simultaneously")
    
    if not st.session_state.breeze_connected:
        st.error("⚠️ Breeze API not connected. Please check credentials.")
    else:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_watchlist = st.multiselect(
                "Select stocks",
                options=sorted(FNO_STOCKS[:30]),
                default=st.session_state.watchlist_stocks[:5],
                max_selections=6
            )
        
        with col2:
            period_multi = st.selectbox("Period", ["1 Day", "5 Days", "30 Days"], index=0, key="multi_period")
            period_map = {"1 Day": 1, "5 Days": 5, "30 Days": 30}
            days = period_map[period_multi]
        
        with col3:
            if st.button("🔄 Refresh", type="primary", use_container_width=True):
                st.rerun()
        
        if selected_watchlist:
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
                            df = fetch_stock_data_breeze(stock_code, days)
                            
                            if not df.empty:
                                current = df['Close'].iloc[-1]
                                prev = df['Close'].iloc[0]
                                change_pct = ((current - prev) / prev) * 100
                                color = "green" if change_pct >= 0 else "red"
                                arrow = "🟢" if change_pct >= 0 else "🔴"
                                
                                st.markdown(f"### {arrow} {stock_name}")
                                st.metric("Price", f"₹{current:.2f}", f"{change_pct:.2f}%")
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['Close'],
                                    mode='lines',
                                    line=dict(color=color, width=2),
                                    fill='tozeroy'
                                ))
                                fig.update_layout(
                                    height=200,
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                            else:
                                st.warning(f"⚠️ No data for {stock_name}")
        else:
            st.info("👆 Select stocks to monitor")

# Footer
st.markdown("---")
st.caption("💡 F&O Dashboard powered by ICICI Breeze API")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")
st.caption("🔌 **Connection Status:** " + ("✅ Connected" if st.session_state.breeze_connected else "❌ Disconnected"))
