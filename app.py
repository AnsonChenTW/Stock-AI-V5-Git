import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
import os
import shutil

# ==========================================
# 🔧 強制重設 Matplotlib 設定
# ==========================================
# 1. 刪除 Matplotlib 的快取資料夾 (核彈級解法)
try:
    cachedir = matplotlib.get_cachedir()
    if os.path.exists(cachedir):
        shutil.rmtree(cachedir)
except Exception as e:
    print(f"Warning: Could not clear matplotlib cache: {e}")

# 2. 設定後端為 Agg (非互動式，適合伺服器)
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import re
import time
from datetime import datetime
import pytz
import requests
import json
import random

# ==========================================
# ⚙️ 頁面基礎設定
# ==========================================
st.set_page_config(
    page_title="AI 量化操盤助手",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🔑 API Key 讀取與設定
# ==========================================
try:
    # 優先從 Streamlit Secrets 讀取
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_KEY = None
    print("⚠️ 系統提示：未檢測到 GEMINI_API_KEY，將自動切換至「演算法備援模式」。")

# --- 全域樣式 ---
FONT_STYLE = "font-family: -apple-system, system-ui, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;"

# ==========================================
# 🧠 核心：混合模式 (Hybrid Core)
# ==========================================

def call_gemini_api(prompt):
    """
    嘗試連線 AI，失敗直接回傳 None (不報錯)，讓後續程式切換備援。
    """
    if not GEMINI_KEY: return None

    models_to_try = ["gemini-1.5-flash", "gemini-pro"]
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_KEY}"
        try:
            # Timeout 設極短 (3秒)，連不上就馬上切換演算法，使用者體驗最好
            response = requests.post(url, headers=headers, json=data, timeout=3)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    raw_text = result['candidates'][0]['content']['parts'][0]['text']
                    return raw_text.replace("```html", "").replace("```", "").strip()
        except:
            continue
            
    return None # 回傳 None 代表 AI 掛了/忙碌中，請用演算法接手

# ==========================================
# 🤖 演算法備援 (Rule-Based Fallback)
# ==========================================

def generate_fallback_strategy(ticker, d):
    # 趨勢文案
    if d['price'] > d['ma20']:
        trend = "股價位於月線之上，短線格局偏多"
        trend_icon = "📈"
    else:
        trend = "股價跌破月線，短線格局轉弱"
        trend_icon = "📉"

    # 動能文案
    if d['rsi'] > 70:
        mom = "RSI 過熱，短線有回檔風險"
    elif d['rsi'] < 30:
        mom = "RSI 超賣，短線有機會反彈"
    else:
        mom = "RSI 位於中性區，動能平穩"

    # 籌碼文案
    vpoc_dist = ((d['price'] - d['vpoc']) / d['vpoc']) * 100
    if d['price'] > d['vpoc']:
        chip = f"股價位於籌碼密集區 (POC) 上方 {vpoc_dist:.1f}%，支撐力道強"
    else:
        chip = f"股價位於籌碼密集區 (POC) 下方 {abs(vpoc_dist):.1f}%，上檔有套牢賣壓"

    # 建議
    if d['score'] >= 6:
        action = "多方操作 (Long)"
        bg = "#e8f5e9" # Green bg
    elif d['score'] <= 2:
        action = "保守觀望 (Defensive)"
        bg = "#ffebee" # Red bg
    else:
        action = "區間操作 (Range)"
        bg = "#fff3e0" # Orange bg

    # 注意：HTML 字串無縮排 (靠左對齊)，這是為了避免 Markdown 解析錯誤
    html = f"""
<div style='background-color:{bg}; padding:12px; border-radius:8px; margin-top:10px; font-size:14px; line-height:1.6;'>
<div style='font-weight:bold; color:#555; margin-bottom:5px;'>🤖 系統自動診斷 (AI 連線備援)</div>
<ul style='margin:0; padding-left:20px;'>
<li><b>{trend_icon} 趨勢：</b>{trend}。</li>
<li><b>⚡ 動能：</b>{mom} (RSI: {d['rsi']:.0f})。</li>
<li><b>🧱 籌碼：</b>{chip}。</li>
</ul>
<hr style='border-top:1px dashed #ccc; margin:8px 0;'>
<div><b>🎯 操作建議：{action}</b></div>
<div style='font-size:12px; color:#777;'>建議停損：{d['atr']*2:.2f} (2xATR)</div>
</div>
"""
    return html

def generate_fallback_brief(tickers):
    t_str = ", ".join(tickers)
    # 注意：HTML 字串無縮排
    return f"""
<h4>🚨 市場連線壅塞 (System Notice)</h4>
<p>由於 Google AI 伺服器暫時無法回應 (IP Rate Limit)，本份早報由系統演算法自動生成。</p>
<ul>
<li><b>今日觀察清單：</b>{t_str}。</li>
<li><b>操作提醒：</b>請直接參考下方個股卡片中的<b>「量化評分 (Score)」</b>與<b>「R/R 風報比」</b>。</li>
<li><b>資金流向：</b>評分 > 6 且 RVOL > 1.2 之個股，代表資金動能強勁。</li>
</ul>
"""

# ==========================================
# 📊 技術指標運算
# ==========================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = calculate_atr(df, period=1) 
    atr_smooth = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr_smooth)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_volume_profile(df, bins=50):
    price_min = df['Low'].min()
    price_max = df['High'].max()
    hist, bin_edges = np.histogram(df['Close'], bins=bins, range=(price_min, price_max), weights=df['Volume'])
    max_idx = np.argmax(hist)
    poc_price = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
    return poc_price

# ==========================================
# 📈 數據與繪圖
# ==========================================

@st.cache_data(ttl=300) # 快取 5 分鐘
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df_daily = stock.history(period="2y")
        df_intraday = stock.history(period="5d", interval="1m")
        return df_daily, df_intraday
    except:
        return pd.DataFrame(), pd.DataFrame()

def create_chart_image(df, ticker, poc_price):
    if len(df) < 50: return None
    plot_df = df.tail(150).copy() 
    
    # 建立圖表：上圖(價格)佔 3 份，下圖(RSI)佔 1 份
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=90, gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('white') 
    
    # 上圖：K線與均線 (強制使用英文標籤，避開中文亂碼)
    ax1.plot(plot_df.index, plot_df['Close'], color='#333', linewidth=1.5, label='Price')
    ax1.plot(plot_df.index, plot_df['MA20'], color='#f39c12', linewidth=1, alpha=0.8, label='MA20')
    ax1.plot(plot_df.index, plot_df['MA50'], color='#27ae60', linewidth=1.5, alpha=0.8, label='MA50')
    ax1.plot(plot_df.index, plot_df['MA200'], color='#2980b9', linewidth=1.5, alpha=0.8, label='MA200')
    ax1.axhline(y=poc_price, color='purple', linestyle='--', linewidth=1, alpha=0.6, label='POC')
    
    ax1.set_title(f"{ticker} Daily Chart", fontsize=10, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.legend(loc='upper left', fontsize='x-small', frameon=False, ncol=2)
    ax1.grid(True, linestyle=':', alpha=0.3)
    
    # 下圖：RSI
    ax2.plot(plot_df.index, plot_df['RSI'], color='#8e44ad', linewidth=1)
    ax2.axhline(70, color='red', linestyle=':', linewidth=0.5)
    ax2.axhline(30, color='green', linestyle=':', linewidth=0.5)
    ax2.set_ylabel('RSI', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    # 轉為 Base64 圖片字串
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=False, facecolor='white')
    plt.close()
    buf.seek(0)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}" style="width:100%; border-radius:8px;">'

# ==========================================
# ⚙️ 單一股票處理
# ==========================================

def process_single_stock(ticker):
    ticker = ticker.strip().upper()
    df, df_intraday = get_stock_data(ticker)
    
    if df.empty or len(df) < 200: return None, ticker, None

    is_tw = ".TW" in ticker or ".TWO" in ticker
    current_price = df['Close'].iloc[-1]
    last_dt = df.index[-1].strftime('%Y-%m-%d')
    if not df_intraday.empty:
        current_price = df_intraday['Close'].iloc[-1]
        last_dt = df_intraday.index[-1].strftime('%Y-%m-%d %H:%M')

    # 指標運算
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['ATR'] = calculate_atr(df)
    df['ADX'] = calculate_adx(df)
    
    macd, signal = calculate_macd(df['Close'])
    poc_price = calculate_volume_profile(df.tail(252))
    
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    curr_vol = df['Volume'].iloc[-1]
    rvol = curr_vol / avg_vol if avg_vol > 0 else 0
    
    #
