import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
# 設定後端為 Agg (非互動式，適合伺服器環境，防止崩潰)
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
import os
# 新增: 用於解析新聞
from bs4 import BeautifulSoup 

# ==========================================
# ⚙️ 頁面基礎設定 (中文標題)
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
# 📰 新聞爬蟲模組 (新增功能)
# ==========================================
@st.cache_data(ttl=600) # 新聞快取 10 分鐘
def get_market_news(ticker):
    """
    簡單爬取 Google Finance 或 Yahoo Finance 的新聞標題
    作為 AI 無法連線時的替代資訊來源。
    """
    news_list = []
    
    # 針對台股與美股做不同處理
    is_tw = ".TW" in ticker or ".TWO" in ticker
    search_query = ticker.replace(".TW", "").replace(".TWO", "")
    
    try:
        # 使用 Yahoo Finance 的 RSS Feed (最穩定且免費)
        url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=4)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.findAll('item')
            for item in items[:3]: # 只抓最新的 3 則
                title = item.title.text
                link = item.link.text
                # 簡單過濾掉太短的標題
                if len(title) > 10:
                    news_list.append(f"<li><a href='{link}' target='_blank' style='color:#333; text-decoration:none;'>📰 {title}</a></li>")
    except:
        pass

    if not news_list:
        return "<li>暫無即時新聞訊號 (No immediate news found)</li>"
    
    return "".join(news_list)

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
            # Timeout 設極短 (3秒)，連不上就馬上切換演算法
            response = requests.post(url, headers=headers, json=data, timeout=3)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    raw_text = result['candidates'][0]['content']['parts'][0]['text']
                    return raw_text.replace("```html", "").replace("```", "").strip()
        except:
            continue
            
    return None 

# ==========================================
# 🤖 演算法備援 (整合新聞)
# ==========================================

def generate_fallback_strategy(ticker, d, news_html):
    # 趨勢文案 (中文)
    if d['price'] > d['ma20']:
        trend = "股價位於月線 (MA20) 之上，短線格局偏多"
        trend_icon = "📈"
    else:
        trend = "股價跌破月線 (MA20)，短線格局轉弱"
        trend_icon = "📉"

    # 動能文案 (中文)
    if d['rsi'] > 70:
        mom = "RSI 過熱 (Overbought)，短線有回檔風險"
    elif d['rsi'] < 30:
        mom = "RSI 超賣 (Oversold)，短線有機會反彈"
    else:
        mom = "RSI 位於中性區，動能平穩"

    # 籌碼文案 (中文)
    vpoc_dist = ((d['price'] - d['vpoc']) / d['vpoc']) * 100
    if d['price'] > d['vpoc']:
        chip = f"股價位於籌碼密集區 (POC) 上方 {vpoc_dist:.1f}%，支撐力道強"
    else:
        chip = f"股價位於籌碼密集區 (POC) 下方 {abs(vpoc_dist):.1f}%，上檔有套牢賣壓"

    # 建議 (中文)
    if d['score'] >= 6:
        action = "多方操作 (Long)"
        bg = "#e8f5e9" # Green bg
    elif d['score'] <= 2:
        action = "保守觀望 (Defensive)"
        bg = "#ffebee" # Red bg
    else:
        action = "區間操作 (Range)"
        bg = "#fff3e0" # Orange bg

    # HTML (包含新聞區塊)
    html = f"""
<div style='background-color:{bg}; padding:12px; border-radius:8px; margin-top:10px; font-size:14px; line-height:1.6;'>
<div style='font-weight:bold; color:#555; margin-bottom:5px;'>🤖 系統自動診斷 (演算法 + 新聞搜索)</div>
<ul style='margin:0; padding-left:20px;'>
<li><b>{trend_icon} 趨勢：</b>{trend}。</li>
<li><b>⚡ 動能：</b>{mom} (RSI: {d['rsi']:.0f})。</li>
<li><b>🧱 籌碼：</b>{chip}。</li>
</ul>
<div style='margin-top:8px; padding-top:8px; border-top:1px dashed #ccc;'>
    <div style='font-size:12px; color:#555; margin-bottom:3px;'><b>📰 市場消息 (News Feed):</b></div>
    <ul style='margin:0; padding-left:20px; font-size:12px; color:#444;'>
        {news_html}
    </ul>
</div>
<hr style='border-top:1px dashed #ccc; margin:8px 0;'>
<div><b>🎯 操作建議：{action}</b></div>
<div style='font-size:12px; color:#777;'>建議停損：{d['atr']*2:.2f} (2xATR)</div>
</div>
"""
    return html

def generate_fallback_brief(tickers):
    t_str = ", ".join(tickers)
    # 修改後的早報提示
    return f"""
<h4>🌐 複合式市場掃描 (Hybrid Market Scan)</h4>
<p style='color:#666; font-size:14px;'>由於 AI 伺服器流量管制，本早報已自動切換為<b>「演算法量化 + 實時新聞抓取」</b>模式，確保資料即時性。</p>
<ul>
<li><b>今日觀察清單：</b>{t_str}。</li>
<li><b>量化訊號：</b>請參考下方個股卡片中的<b>「量化評分 (Score)」</b>。</li>
<li><b>市場消息：</b>系統已自動為每檔個股抓取最新 Yahoo Finance 新聞標題，請見各別分析卡片。</li>
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

@st.cache_data(ttl=300) 
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
    
    # 建立圖表：這部分維持「全英文」，確保圖表不會出現方塊字
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=90, gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('white') 
    
    # 上圖：K線與均線 (英文標籤)
    ax1.plot(plot_df.index, plot_df['Close'], color='#333', linewidth=1.5, label='Price')
    ax1.plot(plot_df.index, plot_df['MA20'], color='#f39c12', linewidth=1, alpha=0.8, label='MA20')
    ax1.plot(plot_df.index, plot_df['MA50'], color='#27ae60', linewidth=1.5, alpha=0.8, label='MA50')
    ax1.plot(plot_df.index, plot_df['MA200'], color='#2980b9', linewidth=1.5, alpha=0.8, label='MA200')
    ax1.axhline(y=poc_price, color='purple', linestyle='--', linewidth=1, alpha=0.6, label='POC')
    
    ax1.set_title(f"{ticker} Daily Chart", fontsize=10, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.legend(loc='upper left', fontsize='x-small', frameon=False, ncol=2)
    ax1.grid(True, linestyle=':', alpha=0.3)
    
    # 下圖：RSI (英文標籤)
    ax2.plot(plot_df.index, plot_df['RSI'], color='#8e44ad', linewidth=1)
    ax2.axhline(70, color='red', linestyle=':', linewidth=0.5)
    ax2.axhline(30, color='green', linestyle=':', linewidth=0.5)
    ax2.set_ylabel('RSI', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
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
    
    # R/R
    support = df['MA50'].iloc[-1]
    resistance = df['High'].tail(252).max()
    if current_price >= resistance * 0.99: resistance = current_price * 1.2
    
    risk = current_price - support
    reward = resistance - current_price
    if risk > 0:
        rr_val = reward / risk
        rr_display = f"1 : {rr_val:.1f}"
        if rr_val >= 3: rr_color = "#27ae60"
        elif rr_val >= 2: rr_color = "#2980b9"
        else: rr_color = "#c0392b"
    else:
        rr_val = 0
        rr_display = "⚠️ 風險高"
        rr_color = "#c0392b"

    score = 0
    if current_price > df['MA20'].iloc[-1]: score += 1
    if current_price > df['MA50'].iloc[-1]: score += 1
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1]: score += 1
    if df['RSI'].iloc[-1] > 50: score += 1
    if macd.iloc[-1] > signal.iloc[-1]: score += 1
    if df['ADX'].iloc[-1] > 25: score += 1
    if current_price > poc_price: score += 1
    if rvol > 1.2: score += 1

    data_dict = {
        'price': current_price,
        'ma20': df['MA20'].iloc[-1],
        'ma50': df['MA50'].iloc[-1],
        'rvol': rvol,
        'atr': df['ATR'].iloc[-1],
        'adx': df['ADX'].iloc[-1],
        'rsi': df['RSI'].iloc[-1],
        'vpoc': poc_price,
        'score': score
    }

    rank_data = {
        'ticker': ticker,
        'score': score,
        'rvol': rvol,
        'price': current_price,
        'market': 'TW' if is_tw else 'US'
    }
    
    # 抓取新聞 (無論 AI 是否連線都執行)
    news_html = get_market_news(ticker)

    # 策略生成
    prompt = f"""
    量化交易員分析 {ticker}。
    Price: {current_price:.2f}, RVOL: {rvol:.2f}, ATR: {df['ATR'].iloc[-1]:.2f}, 
    ADX: {df['ADX'].iloc[-1]:.0f}, RSI: {df['RSI'].iloc[-1]:.0f}, VPOC: {poc_price:.2f}
    請用 HTML (<h4>, <ul>) 給出：1. 量化診斷 2. 交易計劃 (含停損)。
    請使用繁體中文 (Traditional Chinese) 回答，但保留專有名詞 (如 RVOL, RSI, POC)。
    """
    ai_result = call_gemini_api(prompt)
    
    if ai_result:
        strategy_html = f"<div>{ai_result}</div><div style='font-size:10px; color:#aaa; text-align:right;'>Analysis by Gemini</div>"
        # 如果是 AI 回答，我們也可以選擇把新聞附在下面
        strategy_html += f"<div style='margin-top:10px; font-size:12px; color:#555;'><b>📰 最新新聞快遞:</b><ul style='margin:0; padding-left:20px;'>{news_html}</ul></div>"
    else:
        # 如果 AI 掛了，使用內建演算法 + 新聞爬蟲
        strategy_html = generate_fallback_strategy(ticker, data_dict, news_html)
    
    chart_html = create_chart_image(df, ticker, poc_price)
    
    rvol_color = "#d35400" if rvol > 1.2 else "#555"
    currency = "NT$" if is_tw else "$"

    # 組合卡片 HTML (無縮排)
    card_html = f"""
<div style="border:1px solid #e0e0e0; border-radius:12px; padding:16px; margin-bottom:20px; background-color: white; color: #333; {FONT_STYLE}">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<h2 style="margin:0; color:#2c3e50;">{ticker} <span style="font-size:14px; color:#aaa; font-weight:normal;">(Score: {score}/8)</span></h2>
<div style="font-size:12px; color:#999;">{last_dt}</div>
</div>
<div style="text-align:right;">
<div style="font-size:24px; font-weight:800; color:#2c3e50;">{currency}{current_price:.2f}</div>
</div>
</div>

<div style="display:flex; justify-content:space-between; margin-top:10px; background:#f8f9fa; padding:8px; border-radius:8px;">
<div style="text-align:center;"><div style="font-size:10px; color:#777;">RVOL</div><div style="font-weight:bold; color:{rvol_color}">{rvol:.1f}x</div></div>
<div style="text-align:center;"><div style="font-size:10px; color:#777;">R/R</div><div style="font-weight:bold; color:{rr_color}">{rr_display}</div></div>
<div style="text-align:center;"><div style="font-size:10px; color:#777;">ATR</div><div style="font-weight:bold;">{df['ATR'].iloc[-1]:.1f}</div></div>
<div style="text-align:center;"><div style="font-size:10px; color:#777;">RSI</div><div style="font-weight:bold;">{df['RSI'].iloc[-1]:.0f}</div></div>
</div>

<div style="margin-top:8px; font-size:12px; color:#555; display:flex; justify-content:space-between; padding:0 5px;">
<span>🛡️ Support: <b>{currency}{support:.2f}</b></span>
<span>🎯 Target: <b>{currency}{resistance:.2f}</b></span>
</div>

<div style="margin-top:10px;">{chart_html}</div>

<div style="margin-top:15px; padding-top:10px; border-top:1px dashed #eee; font-size:14px; line-height:1.5;">
{strategy_html}
</div>
</div>
"""
    return card_html, ticker, rank_data

# ==========================================
# 📊 排行榜
# ==========================================

def generate_ranking_html(rank_list):
    if not rank_list: return ""
    sorted_list = sorted(rank_list, key=lambda x: (x['score'], x['rvol']), reverse=True)
    
    # 排行榜標題改回中文，表頭英文
    html = f"""
<div style='background-color:#f0f4c3; color:#33691e; padding:15px; border-radius:12px; margin-bottom:25px; border:2px solid #dce775; {FONT_STYLE}'>
<h3 style='margin-top:0; border-bottom:1px solid #c0ca33; padding-bottom:10px;'>🏆 AI 資金效率排行榜</h3>
<table style='width:100%; font-size:14px; border-collapse: collapse;'>
<tr style='text-align:left; color:#558b2f;'>
<th style='padding:5px;'>Rank</th><th style='padding:5px;'>Symbol</th><th style='padding:5px;'>Score</th><th style='padding:5px;'>RVOL</th><th style='padding:5px;'>Price</th>
</tr>
"""
    for i, item in enumerate(sorted_list):
        rank_num = i + 1
        score_color = "#2e7d32" if item['score'] >= 6 else "#f57f17" if item['score'] >= 4 else "#c62828"
        row_bg = "#f9fbe7" if i % 2 == 0 else "transparent"
        currency = "NT$" if item['market'] == "TW" else "$"
        
        # HTML 字串無縮排
        html += f"""
<tr style='background-color:{row_bg}; border-bottom:1px dashed #e6ee9c;'>
<td style='padding:8px; font-weight:bold;'>#{rank_num}</td>
<td style='padding:8px;'><b>{item['ticker']}</b></td>
<td style='padding:8px; color:{score_color}; font-weight:bold;'>{item['score']}/8</td>
<td style='padding:8px;'>{item['rvol']:.1f}x</td>
<td style='padding:8px;'>{currency}{item['price']:.2f}</td>
</tr>
"""
    html += "</table></div>"
    return html

# ==========================================
# 🚀 Streamlit 主程式介面
# ==========================================

st.title("🚀 AI 量化操盤助手 (Pro)")
st.markdown(f"""
<div style='background-color:#e3f2fd; color:#0d47a1; padding:15px; border-radius:10px; margin-bottom:20px;'>
    <b>混合分析模式：</b>AI 分析 + 量化演算法 + 實時新聞掃描。<br>
    <span style='font-size:12px; color:#555;'>圖表採用英文顯示，文字分析支援繁體中文。</span>
</div>
""", unsafe_allow_html=True)

# 提醒使用者輸入 Key (如果沒設定)
if not GEMINI_KEY:
    st.warning("⚠️ 檢測到您尚未設定 API Key，系統將使用「演算法備援模式」。請至 Secrets 設定 GEMINI_API_KEY 以啟用 AI 分析。")

# 側邊欄輸入 (恢復中文)
with st.sidebar:
    st.header("🔍 股票輸入")
    us_input = st.text_area("🇺🇸 美股 (例如: TSM NVDA)", height=100)
    tw_input = st.text_area("🇹🇼 台股 (例如: 2330 2603)", height=100)
    run_btn = st.button("執行全方位分析", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("Created with ❤️ by Streamlit")

# 主執行邏輯
if run_btn:
    if not us_input and not tw_input:
        st.warning("⚠️ 請至少輸入一支股票代號")
    else:
        # 1. 解析輸入
        all_inputs = []
        if us_input: all_inputs.extend(re.split(r'[ ,\n]+', us_input))
        if tw_input: 
            for t in re.split(r'[ ,\n]+', tw_input):
                if t.strip() and t.isdigit(): all_inputs.append(f"{t}.TW")
                elif t.strip(): all_inputs.append(t)
        
        valid_tickers = []
        ranking_data = []
        cards_html_list = []
        
        # 2. 進度條設定
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_stocks = len([x for x in all_inputs if x.strip()])
        processed_count = 0

        for t in all_inputs:
            if not t.strip(): continue
            
            status_text.text(f"正在分析: {t} ...")
            
            card, valid_ticker, rank_item = process_single_stock(t)
            
            if card:
                cards_html_list.append(card)
                valid_tickers.append(valid_ticker)
                if rank_item: ranking_data.append(rank_item)
            
            processed_count += 1
            progress_bar.progress(processed_count / total_stocks)

        status_text.empty()
        progress_bar.empty()

        if not valid_tickers:
            st.error("❌ 未找到有效股票數據")
        else:
            # 4. 生成總結 (Header)
            with st.spinner("🤖 AI 正在撰寫華爾街早報..."):
                prompt = f"華爾街早報。股票：{', '.join(valid_tickers)}。宏觀與資金流向。精簡HTML。請使用繁體中文 (Traditional Chinese) 回答。"
                ai_brief = call_gemini_api(prompt)
                
                if not ai_brief:
                    brief_html = generate_fallback_brief(valid_tickers)
                else:
                    brief_html = ai_brief

            # A. 早報區塊
            final_header = f"""
<div style='background-color:#fffbeb; color:#2c3e50; padding:20px; border-radius:12px; margin-bottom:25px; border:2px solid #f1c40f; box-shadow: 0 4px 10px rgba(0,0,0,0.05); {FONT_STYLE}'>
    <h3 style='margin-top:0; color:#d35400; border-bottom:1px solid #f39c12; padding-bottom:10px;'>☕ 華爾街交易員早報 (Morning Brief)</h3>
    <div style='font-size:15px; line-height:1.6;'>{brief_html}</div>
</div>
"""
            st.markdown(final_header, unsafe_allow_html=True)

            # B. 排行榜區塊
            ranking_html = generate_ranking_html(ranking_data)
            st.markdown(ranking_html, unsafe_allow_html=True)

            # C. 個股卡片區塊
            st.markdown("### 📊 個股深度分析")
            for card_html in cards_html_list:
                st.markdown(card_html, unsafe_allow_html=True)
