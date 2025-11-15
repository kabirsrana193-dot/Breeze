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
    page_title="F&O Dashboard - Breeze (Nifty 200)",
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
# Nifty 200 Stocks
# --------------------------
FNO_STOCKS = [
    "Reliance Industries", "TCS", "HDFC Bank", "Infosys", "ICICI Bank",
    "Bharti Airtel", "ITC", "State Bank of India", "HCL Technologies", "Axis Bank",
    "Kotak Mahindra Bank", "Larsen & Toubro", "Bajaj Finance", "Asian Paints", "Maruti Suzuki",
    "Titan Company", "Sun Pharma", "Wipro", "Ultratech Cement", "Tata Motors",
    "Adani Ports", "Adani Enterprises", "Tech Mahindra", "Power Grid", "NTPC",
    "Coal India", "Tata Steel", "Bajaj Finserv", "Hero MotoCorp", "IndusInd Bank",
    "Mahindra & Mahindra", "Grasim Industries", "Hindalco", "JSW Steel", "SBI Life",
    "ICICI Lombard", "Bajaj Auto", "HDFC Life", "Adani Green", "Shree Cement",
    "Eicher Motors", "UPL", "Tata Consumer", "Britannia", "Nestle India",
    "Hindustan Unilever", "Cipla", "Dr Reddy's", "Divi's Labs", "Apollo Hospitals",
    "IOC", "BPCL", "ONGC", "Gail India", "Pidilite",
    "Berger Paints", "Havells", "Godrej Consumer", "Dabur", "Marico",
    "Colgate", "United Spirits", "Varun Beverages", "Zomato", "Paytm",
    "LIC", "SBI Cards", "Cholamandalam", "Muthoot Finance", "Shriram Finance",
    "Bosch", "ABB India", "Siemens", "Bharat Electronics", "BHEL",
    "HAL", "L&T Technology", "Persistent", "Coforge", "Mphasis",
    "LTIMindtree", "Oracle Financial", "DMart", "Trent", "Jubilant Food",
    "PVR Inox", "IndiGo", "Indian Hotels", "Oberoi Realty", "DLF",
    "Godrej Properties", "Prestige Estates", "Phoenix Mills", "Voltas", "Blue Star",
    "Crompton Greaves", "Polycab", "Dixon Tech", "Amber Enterprises", "Tube Investments",
    "Motherson Sumi", "Bharat Forge", "Balkrishna Ind", "Apollo Tyres", "MRF",
    "Ashok Leyland", "TVS Motor", "Escorts Kubota", "Bajaj Holdings", "ICICI Prudential",
    "Max Financial", "Star Health", "GIC", "New India Assurance", "Bank of Baroda",
    "PNB", "Canara Bank", "Union Bank", "Indian Bank", "Bank of India",
    "Central Bank", "IDBI Bank", "AU Small Finance", "Bandhan Bank", "Federal Bank",
    "IDFC First Bank", "RBL Bank", "Yes Bank", "City Union Bank", "Karur Vysya Bank",
    "Manappuram Finance", "PNB Housing", "Can Fin Homes", "Aarti Industries", "Deepak Nitrite",
    "Navin Fluorine", "PI Industries", "SRF", "Atul Ltd", "Vinati Organics",
    "Gujarat Gas", "Petronet LNG", "IGL", "MGL", "Adani Total Gas",
    "Adani Transmission", "Adani Power", "JSW Energy", "Tata Power", "Torrent Power",
    "CESC", "Alkem Labs", "Lupin", "Aurobindo Pharma", "Biocon",
    "Torrent Pharma", "Zydus Life", "Glenmark", "Ipca Labs", "Natco Pharma",
    "Mankind Pharma", "Laurus Labs", "Max Healthcare", "Fortis Healthcare", "Narayana Health",
    "Metropolis", "Dr Lal PathLabs", "Vedanta", "Jindal Steel", "NMDC",
    "SAIL", "Hindustan Zinc", "NALCO", "Astral", "Supreme Industries",
    "Finolex Cables", "KEI Industries", "Ahluwalia Contracts", "NCC", "PNC Infratech",
    "KEC International", "Kalpataru Projects", "IRB Infrastructure", "NBCC", "Container Corp",
    "Concor", "IRCTC", "Mazagon Dock", "BEML", "Cochin Shipyard",
    "NMDC Steel", "Redington", "V-Guard", "Cera Sanitary", "Kajaria Ceramics",
    "Somany Ceramics", "Orient Electric", "Crompton Consumer", "Symphony", "Whirlpool",
    "IFB Industries", "Butterfly Gandhimathi", "TTK Prestige", "Bajaj Electricals", "GMR Airports",
    "AAI", "Gateway Distriparks", "Allcargo Logistics", "VRL Logistics", "TCI Express",
    "Blue Dart", "Gati", "Mahindra Logistics", "KPIT Technologies", "Tata Elxsi",
    "Cyient", "Zensar Tech", "Sonata Software", "Mastek", "Happiest Minds"
]

# Stock code mapping (6-letter ICICI Breeze codes)
STOCK_CODE_MAP = {
    "Reliance Industries": "RELIND",
    "TCS": "TCS",
    "HDFC Bank": "HDFBAN",
    "Infosys": "INFTEC",
    "ICICI Bank": "ICIBAN",
    "Bharti Airtel": "BHAAIR",
    "ITC": "ITC",
    "State Bank of India": "STABAN",
    "HCL Technologies": "HCLTEC",
    "Axis Bank": "AXIBAN",
    "Kotak Mahindra Bank": "KOTMAH",
    "Larsen & Toubro": "LARTOU",
    "Bajaj Finance": "BAJFI",
    "Asian Paints": "ASIPAI",
    "Maruti Suzuki": "MARUTI",
    "Titan Company": "TITIND",
    "Sun Pharma": "SUNPHA",
    "Wipro": "WIPRO",
    "Ultratech Cement": "ULTCEM",
    "Tata Motors": "TATMOT",
    "Adani Ports": "ADAPOR",
    "Adani Enterprises": "ADAENT",
    "Tech Mahindra": "TECMAH",
    "Power Grid": "POWGRI",
    "NTPC": "NTPC",
    "Coal India": "COALIN",
    "Tata Steel": "TATSTE",
    "Bajaj Finserv": "BAFINS",
    "Hero MotoCorp": "HERHON",
    "IndusInd Bank": "INDBA",
    "Mahindra & Mahindra": "MAHMAH",
    "Grasim Industries": "GRASIM",
    "Hindalco": "HINDAL",
    "JSW Steel": "JSWSTE",
    "SBI Life": "SBILIF",
    "ICICI Lombard": "ICILOM",
    "Bajaj Auto": "BAAUTO",
    "HDFC Life": "HDFSTA",
    "Adani Green": "ADAGRE",
    "Shree Cement": "SHRCEM",
    "Eicher Motors": "EICHER",
    "UPL": "UNIP",
    "Tata Consumer": "TATGLO",
    "Britannia": "BRIIND",
    "Nestle India": "NESIND",
    "Hindustan Unilever": "HINLEV",
    "Cipla": "CIPLA",
    "Dr Reddy's": "DRREDD",
    "Divi's Labs": "DIVLAB",
    "Apollo Hospitals": "APOHOS",
    "IOC": "INDOIL",
    "BPCL": "BHAPET",
    "ONGC": "ONGC",
    "Gail India": "GAIL",
    "Pidilite": "PIDIND",
    "Berger Paints": "BERPAI",
    "Havells": "HAVIND",
    "Godrej Consumer": "GODCON",
    "Dabur": "DABIND",
    "Marico": "MARLIM",
    "Colgate": "COLPAL",
    "United Spirits": "UNISPI",
    "Varun Beverages": "VARBEV",
    "Zomato": "ZOMATO",
    "Paytm": "ONE97",
    "LIC": "LIC",
    "SBI Cards": "SBICAR",
    "Cholamandalam": "CHOINV",
    "Muthoot Finance": "MUTFIN",
    "Shriram Finance": "SHRTRA",
    "Bosch": "BOSLIM",
    "ABB India": "ABB",
    "Siemens": "SIEMEN",
    "Bharat Electronics": "BHAELE",
    "BHEL": "BHEL",
    "HAL": "HINAER",
    "L&T Technology": "LTTEC",
    "Persistent": "PERSYS",
    "Coforge": "NIITEC",
    "Mphasis": "MPHLIM",
    "LTIMindtree": "LTINFO",
    "Oracle Financial": "ORAFIN",
    "DMart": "AVESUP",
    "Trent": "TRENT",
    "Jubilant Food": "JUBFOO",
    "PVR Inox": "PVRLIM",
    "IndiGo": "INTAVI",
    "Indian Hotels": "INDHOT",
    "Oberoi Realty": "OBEREA",
    "DLF": "DLFLIM",
    "Godrej Properties": "GODPRO",
    "Prestige Estates": "PREEST",
    "Phoenix Mills": "PHOMIL",
    "Voltas": "VOLTAS",
    "Blue Star": "BLUSTA",
    "Crompton Greaves": "CROGR",
    "Polycab": "POLI",
    "Dixon Tech": "DIXTEC",
    "Amber Enterprises": "AMBEN",
    "Tube Investments": "TUBEIN",
    "Motherson Sumi": "MOTSU",
    "Bharat Forge": "BHAFOR",
    "Balkrishna Ind": "BALIND",
    "Apollo Tyres": "APOTYR",
    "MRF": "MRFTYR",
    "Ashok Leyland": "ASHLEY",
    "TVS Motor": "TVSMOT",
    "Escorts Kubota": "ESCORT",
    "Bajaj Holdings": "BAJHOL",
    "ICICI Prudential": "ICPRU",
    "Max Financial": "MAXFIN",
    "Star Health": "STAHEA",
    "GIC": "GIC",
    "New India Assurance": "NEWIN",
    "Bank of Baroda": "BANBAR",
    "PNB": "PNBGIL",
    "Canara Bank": "CANBAN",
    "Union Bank": "UNIBAN",
    "Indian Bank": "INDBAN",
    "Bank of India": "BANIND",
    "Central Bank": "CENBAN",
    "IDBI Bank": "IDBI",
    "AU Small Finance": "AUSMA",
    "Bandhan Bank": "BANBAN",
    "Federal Bank": "FEDBAN",
    "IDFC First Bank": "IDFBAN",
    "RBL Bank": "RBLBAN",
    "Yes Bank": "YESBAN",
    "Karur Vysya Bank": "KARVYS",
    "Manappuram Finance": "MANAFI",
    "PNB Housing": "PNBHOU",
    "Can Fin Homes": "CANHOM",
    "Aarti Industries": "AARIND",
    "Deepak Nitrite": "DEENIT",
    "Navin Fluorine": "NAVFLU",
    "PI Industries": "PIIND",
    "SRF": "SRF",
    "Atul Ltd": "ATUL",
    "Vinati Organics": "VINORG",
    "Gujarat Gas": "GUJGA",
    "Petronet LNG": "PETLNG",
    "IGL": "INDGAS",
    "MGL": "MAHGAS",
    "Adani Total Gas": "ADAGAS",
    "Adani Power": "ADAPOW",
    "JSW Energy": "JSWENE",
    "Tata Power": "TATPOW",
    "Torrent Power": "TORPOW",
    "CESC": "CESC",
    "Alkem Labs": "ALKLAB",
    "Lupin": "LUPIN",
    "Aurobindo Pharma": "AURPHA",
    "Biocon": "BIOCON",
    "Torrent Pharma": "TORPHA",
    "Zydus Life": "CADHEA",
    "Glenmark": "GLEPHA",
    "Ipca Labs": "IPCLAB",
    "Natco Pharma": "NATPHA",
    "Mankind Pharma": "MANKIND",
    "Laurus Labs": "LAULAB",
    "Max Healthcare": "MAXHEA",
    "Fortis Healthcare": "FORHEA",
    "Narayana Health": "NARHRU",
    "Vedanta": "VEDLIM",
    "Jindal Steel": "JINSP",
    "NMDC": "NATMIN",
    "SAIL": "SAIL",
    "Hindustan Zinc": "HINZIN",
    "NALCO": "NALCHE"
}



FINANCIAL_RSS_FEEDS = [
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

# --------------------------
# Initialize session state
# --------------------------
if 'breeze_client' not in st.session_state:
    st.session_state.breeze_client = None
if 'breeze_connected' not in st.session_state:
    st.session_state.breeze_connected = False
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []

# --------------------------
# Breeze Connection
# --------------------------
@st.cache_resource
def init_breeze():
    try:
        breeze = BreezeConnect(api_key=app_key)
        breeze.generate_session(api_secret=secret_key, session_token=session_token)
        return breeze, True
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None, False

if not st.session_state.breeze_connected:
    breeze, connected = init_breeze()
    st.session_state.breeze_client = breeze
    st.session_state.breeze_connected = connected

# --------------------------
# Data Functions
# --------------------------
def fetch_historical_data(stock_code, days=30, interval="1day"):
    """Fetch historical data with error handling"""
    try:
        breeze = st.session_state.breeze_client
        if not breeze:
            return None
        
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
        
        time.sleep(0.7)  # Rate limiting
        
        if response and 'Success' in response and response['Success']:
            df = pd.DataFrame(response['Success'])
            
            # Handle different column names
            if 'datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['datetime'])
            elif 'stock_date_time' in df.columns:
                df['Date'] = pd.to_datetime(df['stock_date_time'])
            
            # Standardize column names
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'open':
                    rename_map[col] = 'Open'
                elif col_lower == 'high':
                    rename_map[col] = 'High'
                elif col_lower == 'low':
                    rename_map[col] = 'Low'
                elif col_lower == 'close':
                    rename_map[col] = 'Close'
                elif col_lower == 'volume':
                    rename_map[col] = 'Volume'
            
            df = df.rename(columns=rename_map)
            
            if 'Date' in df.columns:
                df = df.set_index('Date')
                df = df.sort_index()
                
                # Convert to numeric
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove NaN rows
                df = df.dropna(subset=['Close'])
                
                return df
        
        return None
    except Exception as e:
        st.error(f"Error fetching {stock_code}: {str(e)}")
        return None

# --------------------------
# Technical Indicators
# --------------------------
def calculate_sma(data, period):
    """Simple Moving Average"""
    try:
        return data.rolling(window=period).mean()
    except:
        return pd.Series(index=data.index)

def calculate_ema(data, period):
    """Exponential Moving Average"""
    try:
        return data.ewm(span=period, adjust=False).mean()
    except:
        return pd.Series(index=data.index)

def calculate_rsi(data, period=14):
    """RSI Indicator"""
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 0.0001)
        return 100 - (100 / (1 + rs))
    except:
        return pd.Series(index=data.index)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD Indicator"""
    try:
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    except:
        return pd.Series(index=data.index), pd.Series(index=data.index)

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

# --------------------------
# Streamlit App
# --------------------------

st.title("📈 F&O Dashboard - ICICI Breeze")

# Connection Status
if st.session_state.breeze_connected:
    st.success("✅ Connected to Breeze API")
else:
    st.error("❌ Not connected to Breeze API")
    st.stop()

st.markdown("---")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📰 News", "💹 Charts & Indicators", "⚡ Intraday Monitor"])

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
            {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90},
            format_func=lambda x: x,
            key="chart_period"
        )
        days = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90}[period]
    
    with col3:
        interval = st.selectbox(
            "Interval",
            ["1day", "30minute", "5minute", "1minute"],
            format_func=lambda x: {"1day": "Daily", "30minute": "30 Min", "5minute": "5 Min", "1minute": "1 Min"}[x],
            key="chart_interval"
        )
    
    if interval != "1day" and days > 5:
        st.info("ℹ️ Intraday data limited to 5 days")
        days = 5
    
    stock_code = STOCK_CODE_MAP.get(selected_stock)
    
    if stock_code:
        with st.spinner(f"Loading data for {selected_stock}..."):
            df = fetch_historical_data(stock_code, days, interval)
        
        if df is not None and not df.empty and len(df) > 0:
            # Calculate indicators
            df['SMA_20'] = calculate_sma(df['Close'], 20)
            df['SMA_50'] = calculate_sma(df['Close'], 50)
            df['EMA_12'] = calculate_ema(df['Close'], 12)
            df['EMA_26'] = calculate_ema(df['Close'], 26)
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
            
            # Metrics
            current = df['Close'].iloc[-1]
            prev = df['Close'].iloc[0]
            change = current - prev
            change_pct = (change / prev) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{current:.2f}")
            with col2:
                st.metric("Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
            with col3:
                st.metric("High", f"₹{df['High'].max():.2f}")
            with col4:
                st.metric("Low", f"₹{df['Low'].min():.2f}")
            
            st.markdown("---")
            
            # Candlestick chart with moving averages
            fig = go.Figure()
            
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ))
                
                # Add moving averages
                if 'SMA_20' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='blue', width=1.5)
                    ))
                
                if 'SMA_50' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='orange', width=1.5)
                    ))
                
                if 'EMA_12' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['EMA_12'],
                        mode='lines',
                        name='EMA 12',
                        line=dict(color='green', width=1.5, dash='dash')
                    ))
                
                if 'EMA_26' in df.columns:
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
            if 'RSI' in df.columns:
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
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(
                        title="RSI Indicator",
                        height=250,
                        yaxis_range=[0, 100]
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD Chart
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
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
                    fig_macd.update_layout(
                        title="MACD Indicator",
                        height=250
                    )
                    st.plotly_chart(fig_macd, use_container_width=True)
        else:
            st.error(f"Could not fetch data for {selected_stock}")
            st.info("Try: 1) Different stock 2) Check stock code with breeze.get_names()")
    else:
        st.error(f"Stock code not found for {selected_stock}")

# TAB 3: INTRADAY MONITOR
with tab3:
    st.header("⚡ Intraday Multi-Stock Monitor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        watchlist = st.multiselect(
            "Select Stocks",
            FNO_STOCKS,
            default=FNO_STOCKS[:4],
            max_selections=6,
            key="intraday_stocks"
        )
    
    with col2:
        intraday_interval = st.selectbox(
            "Interval",
            ["1minute", "5minute", "30minute"],
            format_func=lambda x: {"1minute": "1 Min", "5minute": "5 Min", "30minute": "30 Min"}[x],
            key="intraday_interval"
        )
    
    with col3:
        if st.button("🔄 Refresh", key="intraday_refresh"):
            st.rerun()
    
    if watchlist:
        num_cols = 2 if len(watchlist) <= 4 else 3
        num_rows = (len(watchlist) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, col in enumerate(cols):
                stock_idx = row * num_cols + col_idx
                
                if stock_idx < len(watchlist):
                    stock_name = watchlist[stock_idx]
                    stock_code = STOCK_CODE_MAP.get(stock_name)
                    
                    with col:
                        if stock_code:
                            df = fetch_historical_data(stock_code, 1, intraday_interval)
                            
                            if df is not None and not df.empty and len(df) >= 2:
                                current = df['Close'].iloc[-1]
                                prev = df['Close'].iloc[0]
                                change = current - prev
                                change_pct = (change / prev) * 100
                                arrow = "🟢" if change_pct >= 0 else "🔴"
                                
                                st.markdown(f"### {arrow} {stock_name}")
                                st.metric("Price", f"₹{current:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                                
                                # Candlestick chart
                                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                                    fig = go.Figure(data=[go.Candlestick(
                                        x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close']
                                    )])
                                    fig.update_layout(
                                        height=250,
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        showlegend=False,
                                        xaxis_rangeslider_visible=False
                                    )
                                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                                else:
                                    # Line chart fallback
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=df.index,
                                        y=df['Close'],
                                        mode='lines',
                                        line=dict(color='green' if change_pct >= 0 else 'red', width=2),
                                        fill='tozeroy'
                                    ))
                                    fig.update_layout(
                                        height=250,
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                            else:
                                st.warning(f"No data for {stock_name}")
                        else:
                            st.error(f"Code not found: {stock_name}")
    else:
        st.info("👆 Select stocks to monitor")

# Footer
st.markdown("---")
st.caption("💡 F&O Dashboard powered by ICICI Breeze API")
st.caption("⚠ **Disclaimer:** For educational purposes only. Not financial advice.")
