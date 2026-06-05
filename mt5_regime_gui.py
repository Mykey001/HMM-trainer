import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import threading
import joblib
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME = mt5.TIMEFRAME_H1
DEFAULT_BARS = 400
REFRESH_SECONDS = 10

# ============================================================
# LOAD MODEL
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

GMM_PATH = os.path.join(script_dir, "market_regime_gmm.pkl")
SCALER_PATH = os.path.join(script_dir, "scaler.pkl")

try:
    gmm = joblib.load(GMM_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise Exception(f"Failed loading model files: {e}")

# ============================================================
# REGIME LABELS
# ============================================================

REGIME_NAMES = {
    0: "Low Volatility Bullish",
    1: "Neutral Consolidation",
    2: "High Volatility Bearish",
    3: "Low Volatility Bearish"
}

REGIME_COLORS = {
    0: "green",
    1: "orange",
    2: "red",
    3: "purple"
}

# ============================================================
# FEATURE ENGINEERING
# ============================================================


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi



def engineer_features(df):
    df = df.copy()

    df.columns = [c.lower() for c in df.columns]

    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    df["volatility_24h"] = df["log_returns"].rolling(24).std()
    df["volatility_120h"] = df["log_returns"].rolling(120).std()

    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    df["trend_20_50"] = (
        (df["sma_20"] - df["sma_50"])
        / df["sma_50"]
    )

    df["trend_50_200"] = (
        (df["sma_50"] - df["sma_200"])
        / df["sma_200"]
    )

    df["rsi_14"] = calculate_rsi(df["close"])

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)

    true_range = ranges.max(axis=1)

    df["atr_14"] = true_range.rolling(14).mean()

    df["natr_14"] = df["atr_14"] / df["close"]

    df.dropna(inplace=True)

    return df

# ============================================================
# MT5 FETCHER
# ============================================================


def fetch_market_data(symbol, timeframe, bars=DEFAULT_BARS):

    if not mt5.initialize():
        raise Exception(f"MT5 initialize failed: {mt5.last_error()}")

    symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        mt5.shutdown()
        raise Exception(f"Symbol {symbol} not found")

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    mt5.shutdown()

    if rates is None:
        raise Exception("No rates received")

    df = pd.DataFrame(rates)

    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df

# ============================================================
# PREDICTION ENGINE
# ============================================================


def predict_regime(df):

    featured = engineer_features(df)

    latest = featured.iloc[[-1]]

    features = [
        "volatility_24h",
        "volatility_120h",
        "trend_20_50",
        "trend_50_200",
        "rsi_14",
        "natr_14"
    ]

    X = latest[features]

    X_scaled = scaler.transform(X)

    regime_id = gmm.predict(X_scaled)[0]

    probabilities = gmm.predict_proba(X_scaled)[0]

    confidence = np.max(probabilities) * 100

    return {
        "regime_id": regime_id,
        "regime_name": REGIME_NAMES.get(regime_id, "Unknown"),
        "confidence": confidence,
        "probabilities": probabilities,
        "price": latest["close"].iloc[0],
        "time": latest.index[0],
        "featured_df": featured
    }

# ============================================================
# GUI APPLICATION
# ============================================================


class RegimeDashboard:

    def __init__(self, root):

        self.root = root

        self.root.title("Live Market Regime Dashboard")
        self.root.geometry("1200x800")

        self.running = True

        self.symbol_var = tk.StringVar(value=DEFAULT_SYMBOL)

        self.timeframe_var = tk.StringVar(value="H1")

        self.timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }

        self.build_gui()

        self.start_live_updates()

    # ========================================================
    # GUI
    # ========================================================

    def build_gui(self):

        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(top_frame, text="Symbol:").pack(side="left")

        self.symbol_entry = ttk.Entry(
            top_frame,
            textvariable=self.symbol_var,
            width=15
        )
        self.symbol_entry.pack(side="left", padx=5)

        ttk.Label(top_frame, text="Timeframe:").pack(side="left")

        timeframe_combo = ttk.Combobox(
            top_frame,
            textvariable=self.timeframe_var,
            values=list(self.timeframe_map.keys()),
            width=10,
            state="readonly"
        )
        timeframe_combo.pack(side="left", padx=5)

        refresh_btn = ttk.Button(
            top_frame,
            text="Refresh Now",
            command=self.manual_refresh
        )
        refresh_btn.pack(side="left", padx=10)

        # ====================================================
        # LIVE DATA PANEL
        # ====================================================

        info_frame = ttk.LabelFrame(self.root, text="Live Market Analysis")
        info_frame.pack(fill="x", padx=10, pady=10)

        self.price_var = tk.StringVar(value="--")
        self.regime_var = tk.StringVar(value="--")
        self.confidence_var = tk.StringVar(value="--")
        self.status_var = tk.StringVar(value="Initializing...")

        ttk.Label(info_frame, text="Price:").grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.price_var).grid(row=0, column=1, sticky="w")

        ttk.Label(info_frame, text="Regime:").grid(row=1, column=0, sticky="w")

        self.regime_label = ttk.Label(
            info_frame,
            textvariable=self.regime_var,
            font=("Arial", 14, "bold")
        )
        self.regime_label.grid(row=1, column=1, sticky="w")

        ttk.Label(info_frame, text="Confidence:").grid(row=2, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.confidence_var).grid(row=2, column=1, sticky="w")

        ttk.Label(info_frame, text="Status:").grid(row=3, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.status_var).grid(row=3, column=1, sticky="w")

        # ====================================================
        # PROBABILITY PANEL
        # ====================================================

        self.prob_frame = ttk.LabelFrame(self.root, text="Regime Probabilities")
        self.prob_frame.pack(fill="x", padx=10, pady=10)

        self.prob_labels = []

        for i in range(4):
            lbl = ttk.Label(self.prob_frame, text=f"Regime {i}: --")
            lbl.pack(anchor="w", padx=10, pady=2)
            self.prob_labels.append(lbl)

        # ====================================================
        # CHART AREA
        # ====================================================

        chart_frame = ttk.Frame(self.root)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.figure = Figure(figsize=(10, 6), dpi=100)

        self.ax_price = self.figure.add_subplot(211)
        self.ax_regime = self.figure.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ========================================================
    # UPDATE LOOP
    # ========================================================

    def start_live_updates(self):

        thread = threading.Thread(target=self.update_loop)
        thread.daemon = True
        thread.start()

    def update_loop(self):

        while self.running:

            try:
                self.fetch_and_update()
            except Exception as e:
                self.status_var.set(str(e))

            time.sleep(REFRESH_SECONDS)

    def manual_refresh(self):

        thread = threading.Thread(target=self.fetch_and_update)
        thread.daemon = True
        thread.start()

    # ========================================================
    # FETCH + UPDATE
    # ========================================================

    def fetch_and_update(self):

        symbol = self.symbol_var.get().strip()

        tf_name = self.timeframe_var.get()
        
        timeframe = self.timeframe_map.get(tf_name, mt5.TIMEFRAME_H1)

        self.status_var.set("Fetching data...")

        try:
            df = fetch_market_data(symbol, timeframe, DEFAULT_BARS)
            
            result = predict_regime(df)
            
            self.price_var.set(f"{result['price']:.2f}")
            self.regime_var.set(result['regime_name'])
            self.confidence_var.set(f"{result['confidence']:.1f}%")
            
            regime_color = REGIME_COLORS.get(result['regime_id'], "black")
            self.regime_label.config(foreground=regime_color)
            
            for i, prob in enumerate(result['probabilities']):
                regime_name = REGIME_NAMES.get(i, f"Regime {i}")
                self.prob_labels[i].config(text=f"{regime_name}: {prob*100:.1f}%")
            
            self.update_charts(result['featured_df'], result['regime_id'])
            
            self.status_var.set(f"Updated at {time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    # ========================================================
    # CHART UPDATE
    # ========================================================

    def update_charts(self, df, current_regime):
        
        self.ax_price.clear()
        self.ax_regime.clear()
        
        # Price chart
        self.ax_price.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=1)
        self.ax_price.plot(df.index, df['sma_20'], label='SMA 20', color='orange', linewidth=0.8, alpha=0.7)
        self.ax_price.plot(df.index, df['sma_50'], label='SMA 50', color='green', linewidth=0.8, alpha=0.7)
        
        self.ax_price.set_title('Price Chart')
        self.ax_price.set_ylabel('Price')
        self.ax_price.legend(loc='upper left')
        self.ax_price.grid(True, alpha=0.3)
        
        # RSI chart
        self.ax_regime.plot(df.index, df['rsi_14'], label='RSI 14', color='purple', linewidth=1)
        self.ax_regime.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        self.ax_regime.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        
        self.ax_regime.set_title(f'RSI - Current Regime: {REGIME_NAMES.get(current_regime, "Unknown")}')
        self.ax_regime.set_ylabel('RSI')
        self.ax_regime.set_xlabel('Time')
        self.ax_regime.legend(loc='upper left')
        self.ax_regime.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def on_close(self):
        self.running = False
        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():
    root = tk.Tk()
    app = RegimeDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()