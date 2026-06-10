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
            pct_change = (input_values[i] / pre_daily.close - 1) * 100
            
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
    def analyze_drawdowns(self):
        """
        Calculates all drawdowns, their durations (downside and recovery), 
        and recovery dates, returning a dictionary mapping drawdown identifiers 
        to a detailed list of metrics.

        Returns:
        dict: A dictionary where keys are strings identifying the drawdown 
            and values are a list of metrics [start_date, Peak_Price, trough_date, 
            Lowest, recovery_date, End_Price, Day_Fall, 
            Day_Recovery_days, status].
        """      
        
        self.data = self.data.sort_index()
        


        price_series = self.data["price"]
        nav = price_series.dropna()
        hwm = nav.cummax() # High Water Mark
        drawdowns = nav / hwm - 1 # Drawdown as a percentage (negative value)
         
        # Initialize a dictionary for results: {identifier: [metrics]}
        dd_recovery_map = {}
        
        in_drawdown = False
        Peak_Price_index = None
        trough_index = None
        min_drawdown_val = 0

        
        for i in range(1, len(nav)):
            

            # --- Detect the start of a drawdown ---
            if drawdowns.iloc[i] < 0 and not in_drawdown:
                in_drawdown = True
                Peak_Price_index = hwm.index[i-1]
                trough_index = None # Reset trough index for the new period
                min_drawdown_val = 0
            
            if in_drawdown:
                # --- Track the lowest point (trough) within the current drawdown ---
                if drawdowns.iloc[i] < min_drawdown_val:
                    min_drawdown_val = drawdowns.iloc[i]
                    trough_index = nav.index[i]
                
                # --- Detect the end/recovery of the drawdown ---
                if drawdowns.iloc[i] >= 0 or i == (len(nav) - 1):

                    dd_period_max_percent = abs(min_drawdown_val) * 100
                    identifier = dd_period_max_percent

                    # Recovery point reached
                    recovery_index = nav.index[i]
                    
                    # Calculate durations
                    Day_Fall = (trough_index - Peak_Price_index).days
                    Fall_Rate = dd_period_max_percent/ Day_Fall 
                    Day_Recovery = (recovery_index - trough_index).days
                    recovery_rate = dd_period_max_percent/ Day_Recovery if Day_Recovery != 0 else 0


                    Day_Total = Day_Fall + Day_Recovery

                    cover_status = "Ongoing" if i == (len(nav) - 1) else "Recoverd"
                    

                    historical_data_before_trough = self.data.loc[self.data.index < trough_index]
                    past_matches = historical_data_before_trough[historical_data_before_trough["price"] <= nav.loc[trough_index]]
                    
                    if not past_matches.empty:
                        last_time_at_price_date = past_matches.index[-1]
                        
                        days_apart_from_past = (trough_index - last_time_at_price_date).days
                        last_time_str = last_time_at_price_date.strftime("%Y-%m-%d")
                    else:
                        # If the asset has never been this cheap before (e.g., an all-time low)
                        last_time_str = "All-Time Low"
                        days_apart_from_past = 0




                    detail = [
                        Peak_Price_index.strftime('%Y-%m-%d'),          # Start Date
                        trough_index.strftime('%Y-%m-%d'),              # Trough Date
                        recovery_index.strftime('%Y-%m-%d'),            # Recovery Date
                        nav.loc[Peak_Price_index],                      # Start Price
                        nav.loc[trough_index],                          # Trough Price
                        nav.loc[recovery_index],                        # Recovery Price
                        Day_Fall,                                       # Downside Duration (days)                        
                        Day_Recovery,                                   # Recovery Duration (days)
                        days_apart_from_past,                           # Days since last time at trough price
                        Day_Total,                                      # Total Duration (days)
                        Fall_Rate,
                        recovery_rate,
                        cover_status,                                   # Status
                    ]

                    dd_recovery_map[identifier] = detail
                    
                    in_drawdown = False
                    trough_index = None # Reset for next loop


        
        analysis_result = pd.DataFrame.from_dict(
            dd_recovery_map, 
            orient='index', 
            columns=["Start",
                "Trough", 
                "Recovery", 
                "Peak_Price",
                "Lowest",
                "End_Price",
                "Day_Fall",                
                "Day_Recovery",
                "Day_Retro",
                "Day_Total",
                "Fall_Rate",
                "Recovery_Rate", 
                "Status"])

        
        
        
        


        analysis_result.index = analysis_result.index.astype(float)
        analysis_result = analysis_result.reset_index(names="DD_%").sort_values(by="Start", ascending=False).round(3).reset_index(drop = True)
        analysis_result['PR'] = ((analysis_result['DD_%'].rank(pct=True) )* 100).map('{:.0f}%'.format)
       
        analysis_result = analysis_result[[
            "DD_%",
            "PR",
            "Status",
            "Start",
            "Trough",
            "Recovery",
            "Peak_Price",
            "Lowest",
            "End_Price",
            "Day_Fall",            
            "Day_Recovery",
            "Day_Retro",
            "Day_Total",
            "Fall_Rate",
            "Recovery_Rate",
            
        ]]
        print(analysis_result)
        return analysis_result.iloc[0]

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

