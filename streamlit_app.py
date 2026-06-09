import numpy as np
from scipy import stats
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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
                feed=DataFeed.IEX  # 100% FIXED: Tells Alpaca to use the free real-time tier
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
        # fig, axes = plt.subplots(2,2, figsize=(11, 8.5), constrained_layout=True)
        # axes = axes.flatten()
        
        for i, (data, val, name) in enumerate(zip(data_sets, input_values, labels)):
            percentile = stats.percentileofscore(data, val)
            # prices = data
            # price_mean = prices.mean()
            # price_max = prices.max()
            # price_min = prices.min()
            # price_std = prices.std()
            # pct_change = (input_values[i] / pre_daily.close - 1) * 100
            
            current_metrics.append({
                "name": name,
                "val": val,
                "pct": pct_change,
                "pr": percentile
            })
            
            # axes[i].hist(data, bins=200, color='skyblue', edgecolor='black', alpha=0.7, cumulative=True, density=True, histtype='step')
            # axes[i].axvline(val, color='red', linestyle='--', label=f'{name}: ${val:.2f}')
            # axes[i].axvspan(price_mean - 1 * price_std, price_mean + 1 * price_std, color="lime", alpha=0.25, label=f'±1σ ({price_mean - 1 * price_std:.2f} - {price_mean + 1 * price_std:.2f})')
            # axes[i].axvspan(price_mean - 2 * price_std, price_mean + 2 * price_std, color="orange", alpha=0.2, label=f'±2σ ({price_mean - 2 * price_std:.2f} - {price_mean + 2 * price_std:.2f})')
            # axes[i].axvspan(price_min, price_max, color="lightcoral", alpha=0.15, label=f'±3σ ({price_min:.2f} - {price_max:.2f})')
            # axes[i].axvline(price_mean, color="black", linestyle="-", linewidth=1, label="Mean")

            # FIXED: Removed the broad x-axis MultipleLocator(10) to let matplotlib auto-scale the price units correctly
            # axes[i].yaxis.set_major_locator(mtick.MultipleLocator(0.10))
            # axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # axes[i].axvline(pre_daily.close, color='grey', linestyle=':', label=f'T-1 close: ${pre_daily.close:.2f}')
            # axes[i].set_title(f"{name} {pct_change:.2f}% (Percentile: {percentile:.1f}%)")
            # axes[i].legend(loc="upper left", fontsize='x-small')
            # axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            
        return  current_metrics, pre_daily.close
        # return fig, current_metrics, pre_daily.close


# --- STREAMLIT DASHBOARD APPLICATION EXECUTION ---
st.title("📊 Real-Time Percentile Analysis")
st.caption("Automatically updating live directly from Alpaca data streams.")

st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="SOXL").upper()
refresh_rate = st.sidebar.slider("Refresh Rate (Seconds)", min_value=5, max_value=60, value=10)

# Triggers the page to refresh safely
st_autorefresh(interval=refresh_rate * 1000, key="data_refresh_heartbeat")

# SECURED: Fetch keys using Streamlit environment secrets 
try:
    api_key = st.secrets["ALPACA_KEY"]
    secret_key = st.secrets["ALPACA_SECRET"]
except KeyError:
    st.error("Missing credentials. Please check your `.streamlit/secrets.toml` file configuration.")
    st.stop()

st.write(f"⏱️ **Last update checked at:** `{datetime.now().strftime('%H:%M:%S')}`")

try:
    analysis = Percentile_Analysis(
        symbol=ticker_input,
        api_key=api_key,
        api_secret=secret_key
    )
    
    # fig, metrics, t_minus_close = analysis.PR()
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
    selected_row = analysis.analyze_drawdowns()  
    items = list(selected_row.items())  # List of tuples: [('Col_1', value), ...]

    MAX_COLS = 3

    # 2. Loop through items in batches of 6
    for i in range(0, len(items), MAX_COLS):
        batch = items[i : i + MAX_COLS]
        
        # Create the horizontal row with up to 6 columns
        cols = st.columns(len(batch))
        
        # Render each item inside its matching column
        for col, (col_name, value) in zip(cols, batch):
            with col:
                st.metric(label=col_name, value=str(value))

        
    st.markdown("---")
    # st.pyplot(fig)
    # plt.close(fig) 
    
except Exception as e:
    st.error(f"Error updating dashboard metrics: {e}")

