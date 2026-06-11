import numpy as np
from scipy import stats
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# --- CONFIG TIMEZONE ---
TARGET_TZ = timezone(timedelta(hours=-5))

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(
    page_title="Alpaca Percentile Analysis", 
    page_icon="📊", 
    layout="wide"
)

class Percentile_Analysis:
    def __init__(
        self,
        df_source: str = None, 
        symbol: str = None,
        start_date: str = None,
        end_date: str = None,
        api_key: str = None,
        api_secret: str = None):
        
        self.df_source = df_source
        self.symbol = symbol
        self.start_date = (
            datetime.strptime(start_date, "%Y-%m-%d")
            if start_date
            else (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d"))
        
        self.end_date = (datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today())
        self.time_frame = TimeFrame.Day
        self.api_key = api_key
        self.api_secret = api_secret
        
        self.data = self.load_dataset() if self.df_source is not None else self.fetch_data()
        self.analysis_result = pd.DataFrame()
        self.snapshot = self.snap()

    def load_dataset(self):
        if self.df_source is not None:
            df = pd.read_csv(self.df_source, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df

    def fetch_data(self) -> pd.DataFrame:
        try:
            client = StockHistoricalDataClient(self.api_key, self.api_secret)
            request = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame.Day,
                start=self.start_date,
                end=self.end_date,
                adjustment="all",
                feed=DataFeed.IEX  
            )
            bars = client.get_stock_bars(request)
            df = (
                bars.df.droplevel(0)
                if isinstance(bars.df.index, pd.MultiIndex)
                else bars.df
            )
            df = df.rename(columns={"close": "price"})
            df["ret"] = df["price"].pct_change().fillna(0)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            st.error(f"Alpaca API error during historical fetch: {e}")
            raise e

    def snap(self):
        client = StockHistoricalDataClient(self.api_key, self.api_secret)
        request = StockSnapshotRequest(symbol_or_symbols=self.symbol)
        snapshots = client.get_stock_snapshot(request)
        return snapshots[self.symbol]

    def PR(self):
        df = self.data.copy()
        df["OC"] = df["open"] / df["price"].shift(1) - 1
        df["HC"] = df["high"] / df["price"].shift(1) - 1
        df["LC"] = df["low"] / df["price"].shift(1) - 1
        df["CC"] = df["price"] / df["price"].shift(1) - 1
        data_sets_per = []
        
        for col in ["CC", "OC", "HC", "LC" ]:
            data = df[col].dropna().to_numpy()
            data_sets_per.append(data)

        daily = self.snapshot.daily_bar
        pre_daily = self.snapshot.previous_daily_bar
        
        input_values = [daily.close, daily.open, daily.high, daily.low]         
        data_sets = [(x + 1) * pre_daily.close for x in data_sets_per] 
        labels = ["Current", "Open", "High", "Low"]
        
        current_metrics = []
        
        for i, (data, val, name) in enumerate(zip(data_sets, input_values, labels)):
            percentile = stats.percentileofscore(data, val)
            pct_change = (input_values[i] / pre_daily.close - 1) * 100
            
            current_metrics.append({
                "name": name,
                "val": val,
                "pct": pct_change,
                "pr": percentile
            })
            
        return  current_metrics, pre_daily.close

    def analyze_drawdowns(self):
        self.data = self.data.sort_index()
        price_series = self.data["price"]
        nav = price_series.dropna()
        hwm = nav.cummax() 
        drawdowns = nav / hwm - 1 
         
        dd_recovery_map = {}
        in_drawdown = False
        Peak_Price_index = None
        trough_index = None
        min_drawdown_val = 0

        for i in range(1, len(nav)):
            if drawdowns.iloc[i] < 0 and not in_drawdown:
                in_drawdown = True
                Peak_Price_index = hwm.index[i-1]
                trough_index = None 
                min_drawdown_val = 0
            
            if in_drawdown:
                if drawdowns.iloc[i] < min_drawdown_val:
                    min_drawdown_val = drawdowns.iloc[i]
                    trough_index = nav.index[i]
                
                if drawdowns.iloc[i] >= 0 or i == (len(nav) - 1):
                    dd_period_max_percent = abs(min_drawdown_val) * 100
                    identifier = dd_period_max_percent
                    recovery_index = nav.index[i]
                    
                    Day_Fall = (trough_index - Peak_Price_index).days
                    Fall_Rate = dd_period_max_percent/ Day_Fall if Day_Fall != 0 else 0
                    Day_Recovery = (recovery_index - trough_index).days
                    recovery_rate = dd_period_max_percent/ Day_Recovery if Day_Recovery != 0 else 0

                    Day_Total = Day_Fall + Day_Recovery
                    cover_status = "Ongoing" if i == (len(nav) - 1) else "Recoverd"
                    
                    historical_data_before_trough = self.data.loc[self.data.index < trough_index]
                    past_matches = historical_data_before_trough[historical_data_before_trough["price"] <= nav.loc[trough_index]]
                    
                    if not past_matches.empty:
                        last_time_at_price_date = past_matches.index[-1]
                        days_apart_from_past = (trough_index - last_time_at_price_date).days
                    else:
                        days_apart_from_past = 0

                    detail = [
                        Peak_Price_index.strftime('%Y-%m-%d'),          
                        trough_index.strftime('%Y-%m-%d'),              
                        recovery_index.strftime('%Y-%m-%d'),            
                        nav.loc[Peak_Price_index],                      
                        nav.loc[trough_index],                          
                        nav.loc[recovery_index],                        
                        Day_Fall,                                       
                        Day_Recovery,                                   
                        days_apart_from_past,                           
                        Day_Total,                                      
                        Fall_Rate,
                        recovery_rate,
                        cover_status,                                   
                    ]
                    dd_recovery_map[identifier] = detail
                    in_drawdown = False
                    trough_index = None 

        analysis_result = pd.DataFrame.from_dict(
            dd_recovery_map, 
            orient='index', 
            columns=["Start", "Trough", "Recovery", "Peak_Price", "Lowest", "End_Price", "Day_Fall", "Day_Recovery", "Day_Retro", "Day_Total", "Fall_Rate", "Recovery_Rate", "Status"])

        analysis_result.index = analysis_result.index.astype(float)
        analysis_result = analysis_result.reset_index(names="DD_%").sort_values(by="Start", ascending=False).round(3).reset_index(drop = True)
        analysis_result['PR'] = ((analysis_result['DD_%'].rank(pct=True) )* 100).map('{:.0f}%'.format)
       
        analysis_result = analysis_result[[
            "DD_%", "PR", "Status", "Start", "Trough", "Recovery", "Peak_Price", "Lowest", "End_Price", "Day_Fall", "Day_Recovery", "Day_Retro", "Day_Total", "Fall_Rate", "Recovery_Rate",
        ]]
        return analysis_result

    def fetch_ohlc_data(self, timeframe: TimeFrame, lookback_days: int) -> pd.DataFrame:
        try:
            client = StockHistoricalDataClient(self.api_key, self.api_secret)
            start_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            
            request = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=timeframe,
                start=start_dt,
                feed=DataFeed.IEX
            )
            bars = client.get_stock_bars(request)
            df = bars.df.droplevel(0) if isinstance(bars.df.index, pd.MultiIndex) else bars.df
            
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize(timezone.utc)
                df.index = df.index.tz_convert(TARGET_TZ)
                
            return df
        except Exception as e:
            st.error(f"Error fetching OHLC data: {e}")
            return pd.DataFrame()

# --- STREAMLIT DASHBOARD APPLICATION EXECUTION ---
st.title("📊 Real-Time Percentile Analysis")
st.caption("Automatically updating live directly from Alpaca data streams.")

st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="SOXL").upper()
refresh_rate = st.sidebar.slider("Refresh Rate (Seconds)", min_value=5, max_value=60, value=10)

st_autorefresh(interval=refresh_rate * 1000, key="data_refresh_heartbeat")

try:
    api_key = st.secrets["ALPACA_KEY"]
    secret_key = st.secrets["ALPACA_SECRET"]    
except KeyError:
    st.error("Missing credentials. Please check your `.streamlit/secrets.toml` file configuration.")
    st.stop()

current_time_tz = datetime.now(TARGET_TZ)
st.write(f"⏱️ **Last update checked at:** `{current_time_tz.strftime('%H:%M:%S')} (UTC-5)`")

try:
    analysis = Percentile_Analysis(
        start_date = "2016-01-01",
        symbol=ticker_input,
        api_key=api_key,
        api_secret=secret_key
    )
    
    metrics, t_minus_close = analysis.PR()
    st.info(f"**Previous Session Close (T-1):** ${t_minus_close:.2f}")
    
    cols = st.columns(4)
    for col, item in zip(cols, metrics):
        with col:
            st.metric(
                label=f"{item['name']} Price", 
                value=f"${item['val']:.2f}", 
                delta=f"{item['pct']:.2f}%"
            )
            st.text(f"Percentile Rank: {item['pr']:.1f}%")
    
    st.markdown("---")
    
    df_drawdown = analysis.analyze_drawdowns()
    selected_row = df_drawdown.iloc[0]   
    items = list(selected_row.items())

    MAX_COLS = 3
    for i in range(0, len(items), MAX_COLS):
        batch = items[i : i + MAX_COLS]
        cols = st.columns(len(batch))
        for col, (col_name, value) in zip(cols, batch):
            with col:
                st.metric(label=col_name, value=str(value))
        
    st.markdown("---")

    st.subheader("Drawdown Historical Analysis Table")
    st.dataframe(df_drawdown, use_container_width=True)
    
    st.markdown("---")

    # --- INTERACTIVE OHLC PLOT SECTION ---
    st.subheader(f"📈 {ticker_input} Interactive OHLC Chart")
    
    # Updated dropdown list to handle new timeframe scopes
    ohlc_option = st.selectbox(
        "Select Chart View (Frequency - Time Frame):",
        options=["1min - 1Day", "1min - 5Day", "15min - 10Day", "1hr - 60Day", "1day - 360Day"],
        index=0
    )
    
    # Enhanced mapping configurations to accommodate hourly and macro day bars
    config_mapping = {
        "1min - 1Day":   {"tf": TimeFrame.Minute, "days": 1},
        "1min - 5Day":   {"tf": TimeFrame.Minute, "days": 5},
        "15min - 10Day": {"tf": TimeFrame(15, TimeFrame.Minute.unit), "days": 10},
        "1hr - 60Day":   {"tf": TimeFrame.Hour, "days": 60},
        "1day - 360Day": {"tf": TimeFrame.Day, "days": 360}
    }
    
    selected_config = config_mapping[ohlc_option]
    
    ohlc_data = analysis.fetch_ohlc_data(
        timeframe=selected_config["tf"], 
        lookback_days=selected_config["days"]
    )
    
    if not ohlc_data.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['open'],
            high=ohlc_data['high'],
            low=ohlc_data['low'],
            close=ohlc_data['close'],
            name=ticker_input,
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        )])
        
        # Format the x-axis dynamic labels cleanly depending on if it's Intraday vs Macro Daily
        x_format = "%Y-%m-%d" if selected_config["tf"] == TimeFrame.Day else "%m-%d %H:%M"
        
        fig.update_layout(
            title=f"{ticker_input} Price Action ({ohlc_option} | Local Time UTC-5)",
            xaxis_title="Time / Date (UTC-5)",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False, 
            template="plotly_dark",           
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        fig.update_xaxes(tickformat=x_format)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No intraday or historical bar data returned for the selected window.")
    
except Exception as e:
    st.error(f"Error updating dashboard metrics: {e}")


