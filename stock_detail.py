import warnings
warnings.filterwarnings("ignore")

import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


def _is_rate_limit(e):
    msg = str(e).lower()
    return "rate" in msg or "429" in msg or "too many" in msg

# ── colour palette ─────────────────────────────────────────────────────────────
PLOT_BG   = "#f8f9ff"
PAPER_BG  = "#ffffff"
FONT_COL  = "#111111"
GRID_COL  = "#dde3f5"
UP_COL    = "#1aaa55"
DOWN_COL  = "#cc3300"
LINE_COL  = "#1a56db"
MA50_COL  = "#e6a817"
MA200_COL = "#9b59b6"
SPY_COL   = "#888888"
EST_COL   = "#7aaae8"

BASE_LAYOUT = dict(
    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_COL, size=15),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10),
)
AXIS = dict(gridcolor=GRID_COL, showgrid=True, tickfont=dict(size=14))


def apply_styles():
    st.markdown("""
<style>
  html, body, [class*="css"] { font-size: 18px !important; }
  h1  { font-size: 2.4rem !important; margin-bottom: 0.2rem !important; }
  h2  { font-size: 1.5rem !important; margin-top: 1.8rem !important;
        padding-bottom: 6px !important; border-bottom: 2px solid #1a56db !important;
        color: #1a56db !important; }
  [data-testid="metric-container"] > div:nth-child(2)
      { font-size: 2rem !important; font-weight: 700; }
  .stDataFrame td, .stDataFrame th { font-size: 1rem !important; }
  hr  { margin: 1.5rem 0 !important; }
  .insight { background:#f0f4ff; border-left:5px solid #1a56db;
             padding:12px 16px; border-radius:6px; margin:10px 0;
             font-size:1.05rem; line-height:1.6; }
  .insight-warn { background:#fff4f0; border-left:5px solid #cc3300; }
  .insight-ok   { background:#f0fff4; border-left:5px solid #1aaa55; }
  .insight-neu  { background:#f8f8f8; border-left:5px solid #999; }
  [data-testid="stTextInput"] input { font-size: 1.4rem !important; font-weight: 700 !important; padding: 10px 14px !important; }
</style>
""", unsafe_allow_html=True)


def insight(text, kind="neu"):
    st.markdown(f"<div class='insight insight-{kind}'>{text}</div>",
                unsafe_allow_html=True)


def _rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@st.cache_data(ttl=1800)
def load(sym):
    end   = datetime.today() + timedelta(days=1)   # +1 so end is exclusive-safe
    start = end - timedelta(days=549)
    s_str = start.strftime("%Y-%m-%d")
    e_str = end.strftime("%Y-%m-%d")

    # ── 1 request: batch-download price data for ticker + SPY ─────────────────
    dl_list = [sym, "SPY"] if sym != "SPY" else ["SPY"]
    raw = yf.download(dl_list, start=s_str, end=e_str,
                      progress=False, group_by="ticker", auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No price data for {sym}")

    def _extract(df, tkr):
        if isinstance(df.columns, pd.MultiIndex):
            lvl = df.columns.get_level_values(0)
            return df[tkr].copy() if tkr in lvl else pd.DataFrame()
        return df.copy()

    hist = _extract(raw, sym)
    spy  = _extract(raw, "SPY")

    if hist.empty:
        raise ValueError(f"No price data for {sym}")

    hist["Return"] = hist["Close"].pct_change() * 100
    hist["MA50"]   = hist["Close"].rolling(50).mean()
    hist["MA200"]  = hist["Close"].rolling(200).mean()
    hist["RSI"]    = _rsi(hist["Close"])
    time.sleep(0.5)

    # ── 1 request: ticker metadata — degrade gracefully if rate-limited ────────
    t    = yf.Ticker(sym)
    info = {}
    try:
        info = t.info or {}
    except Exception as e:
        if not _is_rate_limit(e):
            raise
        # rate limited on info — continue with empty dict; chart still renders
    time.sleep(0.5)

    # ── 1 optional request: sector ETF ────────────────────────────────────────
    sector_etf_map = {
        "Technology": "XLK", "Healthcare": "XLV", "Energy": "XLE",
        "Financial Services": "XLF", "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP", "Industrials": "XLI",
        "Basic Materials": "XLB", "Real Estate": "XLRE",
        "Utilities": "XLU", "Communication Services": "XLC",
    }
    sector_etf_sym = sector_etf_map.get(info.get("sector", ""))
    sector_rets    = pd.Series(dtype=float)
    try:
        if sector_etf_sym:
            s_raw = yf.download(sector_etf_sym, start=s_str, end=e_str,
                                progress=False, auto_adjust=True)
            if not s_raw.empty:
                s_close = (s_raw["Close"] if "Close" in s_raw.columns
                           else s_raw.xs("Close", level=0, axis=1).iloc[:, 0])
                sector_rets = s_close.pct_change().dropna() * 100
    except Exception:
        pass
    time.sleep(0.5)

    # ── remaining calls: all optional, already wrapped ────────────────────────
    try:
        fin = t.financials
        if fin is None or fin.empty:
            fin = t.income_stmt
    except Exception:
        fin = pd.DataFrame()
    time.sleep(0.4)

    try:    rev_est  = t.revenue_estimate
    except: rev_est  = pd.DataFrame()
    try:    earn_est = t.earnings_estimate
    except: earn_est = pd.DataFrame()
    time.sleep(0.4)

    try:    news     = t.news or []
    except: news     = []
    try:    cal      = t.calendar
    except: cal      = None
    try:    rec_sum  = t.recommendations_summary
    except: rec_sum  = pd.DataFrame()
    try:    insiders = t.insider_transactions
    except: insiders = pd.DataFrame()

    return (hist.dropna(subset=["Return"]), spy, info, fin, rev_est, earn_est,
            news, cal, rec_sum, insiders, sector_etf_sym, sector_rets)


@st.cache_data(ttl=1800)
def load_summary(sym):
    """Lightweight fetch for watchlist rows — price metrics only."""
    try:
        t   = yf.Ticker(sym)
        end = datetime.today() + timedelta(days=1)   # +1 so end is exclusive-safe
        start = datetime(end.year, 1, 1)
        hist  = t.history(start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"))
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist.empty:
            return None

        current_price = float(hist["Close"].iloc[-1])
        prev_close    = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        today_pct     = (current_price / prev_close - 1) * 100

        price_5d_ago  = float(hist["Close"].iloc[-6]) if len(hist) >= 6 else float(hist["Close"].iloc[0])
        pct_5d        = (current_price / price_5d_ago - 1) * 100

        ytd_pct       = (current_price / float(hist["Close"].iloc[0]) - 1) * 100

        week52_high   = float(hist["High"].max())
        week52_low    = float(hist["Low"].min())

        # company name — fast_info is quicker than full info
        name = sym
        beta = None
        try:
            fi   = t.fast_info
            name = getattr(fi, "name", None) or sym
            beta = t.info.get("beta")
        except Exception:
            pass

        return {
            "name":         name,
            "price":        current_price,
            "today_pct":    today_pct,
            "pct_5d":       pct_5d,
            "ytd_pct":      ytd_pct,
            "week52_high":  week52_high,
            "week52_low":   week52_low,
            "beta":         beta,
        }
    except Exception:
        return None


# ── fundamental helpers ────────────────────────────────────────────────────────
def get_row(fin_df, *keys):
    for k in keys:
        if k in fin_df.index:
            return fin_df.loc[k].sort_index()
    return pd.Series(dtype=float)


def hist_series(fin_df, *keys):
    s = get_row(fin_df, *keys)
    if s.empty:
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.index = s.index.year
    return s.astype(float)


def est_series(est_df, row_key="avg"):
    if est_df is None or est_df.empty:
        return pd.Series(dtype=float)
    try:
        df_t = est_df.T if est_df.shape[0] < est_df.shape[1] else est_df
        s = (df_t.loc[row_key] if row_key in df_t.index
             else df_t.loc["Avg Estimate"] if "Avg Estimate" in df_t.index
             else df_t.iloc[0])
        yr  = datetime.today().year
        out = {}
        for k, v in s.items():
            k = str(k)
            if   k == "0y":  out[yr]     = v
            elif k == "+1y": out[yr + 1] = v
            elif k == "+2y": out[yr + 2] = v
        return pd.Series(out, dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def bar_chart(hist_s, est_s, title, y_label, fmt_billions=False):
    fig = go.Figure()

    def fmt_val(v):
        if fmt_billions:
            if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
            if abs(v) >= 1e6: return f"${v/1e6:.0f}M"
            return f"${v:.0f}"
        return f"{v:.2f}"

    def val_colors(values, default):
        return [DOWN_COL if v < 0 else default for v in values]

    if not hist_s.empty:
        fig.add_trace(go.Bar(x=[str(y) for y in hist_s.index], y=hist_s.values,
            name="Actual", marker_color=val_colors(hist_s.values, LINE_COL),
            text=[fmt_val(v) for v in hist_s.values],
            textposition="outside", textfont=dict(size=15, color=FONT_COL),
            cliponaxis=False))
    if not est_s.empty:
        fwd = est_s[~est_s.index.isin(hist_s.index)] if not hist_s.empty else est_s
        if not fwd.empty:
            fig.add_trace(go.Bar(x=[str(y) for y in fwd.index], y=fwd.values,
                name="Estimate", marker_color=val_colors(fwd.values, EST_COL),
                text=[fmt_val(v) for v in fwd.values],
                textposition="outside", textfont=dict(size=15, color=FONT_COL),
                cliponaxis=False))
    if hist_s.empty and est_s.empty:
        fig.add_annotation(text="No data available", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False, font=dict(size=18, color="#888"))

    all_vals = [v for v in list(hist_s.values) + (list(est_s.values) if not est_s.empty else [])
                if pd.notna(v)]
    ymax = max(all_vals) * 1.35 if all_vals else None
    ymin = min(all_vals) * 1.35 if all_vals and min(all_vals) < 0 else None

    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color=FONT_COL), x=0, y=0.97),
        height=480, margin=dict(l=20, r=20, t=80, b=40),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COL, size=15), barmode="group",
        legend=dict(font=dict(size=15), orientation="h", y=-0.1),
        xaxis=dict(tickfont=dict(size=16), type="category"),
        yaxis=dict(gridcolor=GRID_COL, tickfont=dict(size=14),
                   title=y_label, title_font=dict(size=14),
                   range=[ymin, ymax] if ymax else None, automargin=True),
        uniformtext=dict(mode="hide", minsize=11))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def render(ticker, thr=5):
    """Render the full stock detail dashboard for a given ticker."""

    with st.spinner(f"Loading {ticker}…"):
        data, last_err = None, None
        for attempt in range(3):
            try:
                data = load(ticker)
                break
            except Exception as e:
                last_err = e
                if _is_rate_limit(e) and attempt < 2:
                    time.sleep(15 * (attempt + 1))  # 15s, then 30s
                else:
                    break
        if data is None:
            if _is_rate_limit(last_err):
                st.warning(
                    "⚠️ Yahoo Finance is temporarily rate-limiting this server. "
                    "Wait 30–60 seconds and click Retry.")
                if st.button("🔄 Retry", key=f"retry_{ticker}"):
                    load.clear()
                    st.rerun()
            else:
                st.error(f"Could not load '{ticker}': {last_err}")
            return
        (df_full, spy, info, fin, rev_est, earn_est,
         news, cal, rec_sum, insiders,
         sector_etf_sym, sector_rets) = data

    if df_full.empty:
        st.error(f"No price data for '{ticker}'.")
        return

    cutoff = datetime.today() - timedelta(days=365)
    df     = df_full[df_full.index >= pd.Timestamp(cutoff).tz_localize(df_full.index.tz)]

    returns       = df["Return"]
    latest_ret    = returns.iloc[-1]
    latest_date   = returns.index[-1].strftime("%b %d")
    current_price = df["Close"].iloc[-1]
    up_days   = int((returns >  0.5).sum())
    down_days = int((returns < -0.5).sum())
    flat_days = int(len(returns) - up_days - down_days)
    total     = len(returns)
    big_drops = int((returns <= -thr).sum())
    big_rips  = int((returns >=  thr).sum())

    def perf(days):
        if len(df) > days:
            return (df["Close"].iloc[-1] / df["Close"].iloc[-(days + 1)] - 1) * 100
        return None

    perf_5d  = perf(5)
    perf_30d = perf(30)
    jan1     = df[df.index.year == datetime.today().year]
    perf_ytd = (df["Close"].iloc[-1] / jan1["Close"].iloc[0] - 1) * 100 if not jan1.empty else None

    # ── title ──────────────────────────────────────────────────────────────────
    name = info.get("longName") or info.get("shortName") or ticker
    st.title(f"📈  {name}  ({ticker})")
    _price_col = UP_COL if latest_ret >= 0 else DOWN_COL
    st.markdown(
        f"<div style='margin-top:-10px; margin-bottom:14px;'>"
        f"<span style='font-size:2.2rem; font-weight:800; color:{FONT_COL};'>${current_price:.2f}</span>"
        f"&nbsp;&nbsp;"
        f"<span style='font-size:1.5rem; font-weight:700; color:{_price_col};'>"
        f"{'▲' if latest_ret>=0 else '▼'} {latest_ret:+.1f}% today</span>"
        f"&nbsp;&nbsp;<span style='font-size:1rem; color:#888;'>as of {latest_date}</span>"
        f"</div>",
        unsafe_allow_html=True)
    st.markdown("---")

    # ── KPI strip ──────────────────────────────────────────────────────────────
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    r1c1.metric("Current price",  f"${current_price:.2f}")
    r1c2.metric("Today",          f"{latest_ret:+.1f}%",  help=latest_date)
    r1c3.metric("Last 5 days",    f"{perf_5d:+.1f}%"  if perf_5d  is not None else "—")
    r1c4.metric("Last 30 days",   f"{perf_30d:+.1f}%" if perf_30d is not None else "—")
    r1c5.metric("YTD",            f"{perf_ytd:+.1f}%" if perf_ytd is not None else "—")

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    r2c1.metric("Best day",           f"{returns.max():+.1f}%")
    r2c2.metric("Worst day",          f"{returns.min():+.1f}%")
    r2c3.metric("Median daily move",  f"{returns.median():+.1f}%")
    r2c4.metric(f"Big drops ≥{thr}%", f"{big_drops}  ({big_drops/total*100:.0f}%)")
    r2c5.metric(f"Big rips  ≥{thr}%", f"{big_rips}  ({big_rips/total*100:.0f}%)")
    st.markdown("---")

    # ── overall assessment ─────────────────────────────────────────────────────
    def build_assessment():
        score  = 0.0
        points = []

        summary  = info.get("longBusinessSummary", "")
        if summary:
            first_sent = summary.split(".")[0].strip() + "."
            points.append(("🏢", first_sent, None))

        ma50_v  = df["MA50"].dropna().iloc[-1]  if df["MA50"].dropna().size  else None
        ma200_v = df["MA200"].dropna().iloc[-1] if df["MA200"].dropna().size else None
        rsi_v   = df["RSI"].dropna().iloc[-1]   if df["RSI"].dropna().size   else None

        if ma200_v:
            if current_price > ma200_v:
                score += 1.0
                points.append(("📈", f"Price (${current_price:.2f}) is <b>above the 200-day moving average</b> (${ma200_v:.2f}) — long-term trend is up.", True))
            else:
                score -= 1.0
                points.append(("📉", f"Price (${current_price:.2f}) is <b>below the 200-day moving average</b> (${ma200_v:.2f}) — long-term trend has turned negative.", False))
        if ma50_v and ma200_v:
            if ma50_v > ma200_v:
                score += 0.5
            else:
                score -= 0.5
                points.append(("⚠️", "The 50-day MA has crossed <b>below</b> the 200-day MA (Death Cross) — a bearish technical signal.", False))

        if rsi_v is not None:
            if rsi_v < 30:
                score += 1.0
                points.append(("🔻", f"RSI is <b>{rsi_v:.0f} — oversold</b>. The stock has likely fallen further than fundamentals justify; historically a buying opportunity.", True))
            elif rsi_v > 70:
                score -= 0.5
                points.append(("🔺", f"RSI is <b>{rsi_v:.0f} — overbought</b>. Momentum is stretched; near-term pullback risk is elevated.", False))
            elif rsi_v >= 50:
                score += 0.3
            else:
                score -= 0.3

        if not spy.empty:
            spy_12 = spy[spy.index >= pd.Timestamp(cutoff).tz_localize(spy.index.tz)]
            if not spy_12.empty and len(df) > 0:
                stk_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
                spy_ret = (spy_12["Close"].iloc[-1] / spy_12["Close"].iloc[0] - 1) * 100
                diff    = stk_ret - spy_ret
                if diff > 15:
                    score += 1.0
                    points.append(("🏆", f"<b>Outperforming the S&P 500 by {diff:+.0f}%</b> over 12 months — strong relative strength.", True))
                elif diff < -15:
                    score -= 1.0
                    points.append(("🐢", f"<b>Underperforming the S&P 500 by {abs(diff):.0f}%</b> over 12 months — significant relative weakness.", False))

        rev_hist_f  = pd.Series(dtype=float)
        earn_hist_f = pd.Series(dtype=float)
        if not fin.empty:
            rev_hist_f  = hist_series(fin, "Total Revenue", "Revenue")
            earn_hist_f = hist_series(fin, "Net Income", "Net Income Common Stockholders",
                                      "Net Income From Continuing Operation Net Minority Interest")

        if len(rev_hist_f) >= 2:
            rev_vals   = rev_hist_f.dropna().values
            rev_growth = (rev_vals[-1] - rev_vals[-2]) / abs(rev_vals[-2]) * 100
            if rev_growth > 15:
                score += 1.0
                points.append(("💰", f"Revenue is <b>growing strongly (+{rev_growth:.0f}% last year)</b> — business is expanding.", True))
            elif rev_growth > 0:
                score += 0.5
                points.append(("💰", f"Revenue is growing (+{rev_growth:.0f}% last year).", True))
            elif rev_growth < -10:
                score -= 1.0
                points.append(("💸", f"Revenue is <b>shrinking ({rev_growth:.0f}% last year)</b> — business is contracting.", False))
            else:
                score -= 0.3
                points.append(("💸", f"Revenue is flat/slightly declining ({rev_growth:.0f}% last year).", False))

        if len(earn_hist_f) >= 1:
            last_earn = earn_hist_f.dropna().values[-1]
            if last_earn > 0:
                score += 0.5
                if len(earn_hist_f) >= 2:
                    prev_earn = earn_hist_f.dropna().values[-2]
                    if last_earn > prev_earn:
                        score += 0.5
                        points.append(("✅", "Company is <b>profitable and earnings are growing</b> year over year.", True))
                    else:
                        points.append(("✅", "Company is profitable (earnings positive).", True))
            else:
                score -= 1.0
                points.append(("❌", f"Company is <b>not yet profitable</b> (net loss). This increases risk but is common for high-growth companies.", False))

        rec_mean   = info.get("recommendationMean")
        n_analysts = info.get("numberOfAnalystOpinions", 0)
        if rec_mean and n_analysts:
            if rec_mean <= 2.0:   score += 1.0
            elif rec_mean <= 2.5: score += 0.5
            elif rec_mean >= 4.0: score -= 1.0
            elif rec_mean >= 3.5: score -= 0.5

        short_pct_v = info.get("shortPercentOfFloat")
        if short_pct_v:
            sp = short_pct_v * 100
            if sp > 20:
                points.append(("⚡", f"<b>Very high short interest ({sp:.0f}% of float)</b> — significant squeeze potential if positive catalysts arrive, but also signals market skepticism.", None))
            elif sp > 10:
                points.append(("⚡", f"Elevated short interest ({sp:.0f}% of float) — worth watching.", None))

        if cal is not None:
            try:
                earn_dates = (cal.get("Earnings Date", []) if isinstance(cal, dict)
                             else cal.loc["Earnings Date"].dropna().tolist()
                             if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index else [])
                future_dates = [d for d in earn_dates if (pd.Timestamp(d) - pd.Timestamp(datetime.today())).days >= 0]
                if future_dates:
                    days_away = (pd.Timestamp(future_dates[0]) - pd.Timestamp(datetime.today())).days
                    if 0 <= days_away <= 21:
                        points.append(("📅", f"<b>Earnings in {days_away} days</b> — expect heightened volatility around the announcement.", None))
            except Exception:
                pass

        if   score >= 3.5: verdict, badge_color = "STRONG BUY",  "#0a7a3a"
        elif score >= 2.0: verdict, badge_color = "BUY",         "#1aaa55"
        elif score >= 0.5: verdict, badge_color = "HOLD",        "#888800"
        elif score >= -1:  verdict, badge_color = "SELL",        "#cc6600"
        else:              verdict, badge_color = "STRONG SELL", "#cc3300"

        return verdict, badge_color, score, points

    verdict, badge_color, score, points = build_assessment()

    st.subheader("Overall assessment")
    st.markdown(
        f"<div style='background:{badge_color}; color:white; display:inline-block; "
        f"padding:10px 28px; border-radius:8px; font-size:1.6rem; font-weight:700; "
        f"letter-spacing:2px; margin-bottom:16px;'>{verdict}</div>"
        f"<span style='color:#666; font-size:0.95rem; margin-left:14px;'>"
        f"Algorithmic score: {score:+.1f} &nbsp;·&nbsp; Not financial advice</span>",
        unsafe_allow_html=True)

    for emoji, text, positive in points:
        if positive is True:    bullet_color, bg = UP_COL,   "#f0fff4"
        elif positive is False: bullet_color, bg = DOWN_COL, "#fff4f0"
        else:                   bullet_color, bg = "#555",   "#f8f8f8"
        st.markdown(
            f"<div style='background:{bg}; border-left:4px solid {bullet_color}; "
            f"padding:9px 14px; margin:5px 0; border-radius:4px; font-size:1.05rem;'>"
            f"{emoji}&nbsp; {text}</div>",
            unsafe_allow_html=True)
    st.markdown("---")

    # ── up / down bar ──────────────────────────────────────────────────────────
    st.subheader("Up days vs Down days")
    _up_pct   = up_days   / total * 100
    _flat_pct = flat_days / total * 100
    _dn_pct   = down_days / total * 100
    st.markdown(
        f"<div style='display:flex; height:38px; border-radius:8px; overflow:hidden; margin:12px 0 6px 0;'>"
        f"  <div style='width:{_up_pct:.1f}%; background:{UP_COL}; display:flex; align-items:center;"
        f"              justify-content:center; color:white; font-weight:700; font-size:1rem;'>"
        f"    {'▲ ' + str(up_days) if _up_pct > 8 else ''}</div>"
        f"  <div style='width:{_flat_pct:.1f}%; background:#aaaaaa; display:flex; align-items:center;"
        f"              justify-content:center; color:white; font-size:0.9rem;'>"
        f"    {'— ' + str(flat_days) if _flat_pct > 5 else ''}</div>"
        f"  <div style='width:{_dn_pct:.1f}%; background:{DOWN_COL}; display:flex; align-items:center;"
        f"              justify-content:center; color:white; font-weight:700; font-size:1rem;'>"
        f"    {'▼ ' + str(down_days) if _dn_pct > 8 else ''}</div>"
        f"</div>"
        f"<div style='display:flex; justify-content:space-between; font-size:1rem; color:#444; margin-top:6px;'>"
        f"  <span><b style='color:{UP_COL}'>▲ Up</b>&nbsp; {up_days} days &nbsp;({_up_pct:.0f}%)</span>"
        f"  <span><b style='color:#888'>— Flat</b>&nbsp; {flat_days} days &nbsp;({_flat_pct:.0f}%)</span>"
        f"  <span><b style='color:{DOWN_COL}'>▼ Down</b>&nbsp; {down_days} days &nbsp;({_dn_pct:.0f}%)</span>"
        f"</div>",
        unsafe_allow_html=True)
    st.markdown("---")

    # ── price performance tabs ─────────────────────────────────────────────────
    st.subheader("Price performance")
    perf_tabs  = st.tabs(["5 days", "30 days", "Year to date", "12 months"])
    period_map = {"5 days": 5, "30 days": 30, "Year to date": None, "12 months": 365}

    for tab, label in zip(perf_tabs, period_map):
        with tab:
            days     = period_map[label]
            slice_df = (df[df.index.year == datetime.today().year]
                        if days is None else df.iloc[-days:] if len(df) >= days else df)
            if slice_df.empty:
                st.info("Not enough data."); continue
            base  = slice_df["Close"].iloc[0]
            chg   = (slice_df["Close"].iloc[-1] / base - 1) * 100
            color = UP_COL if chg >= 0 else DOWN_COL
            st.markdown(
                f"<span style='font-size:1.4rem;font-weight:700;color:{color}'>"
                f"{'▲' if chg>=0 else '▼'} {chg:+.1f}%</span>"
                f"<span style='font-size:1rem;color:#555;'>  "
                f"${slice_df['Close'].iloc[0]:.2f} → ${slice_df['Close'].iloc[-1]:.2f}</span>",
                unsafe_allow_html=True)
            fig_p = go.Figure(go.Scatter(
                x=slice_df.index, y=slice_df["Close"], mode="lines",
                fill="tozeroy", fillcolor=f"rgba({'26,170,85' if chg>=0 else '204,51,0'},0.08)",
                line=dict(color=color, width=2.5)))
            fig_p.update_layout(**BASE_LAYOUT, height=280,
                xaxis=dict(**AXIS), yaxis=dict(**AXIS, title="Price ($)"),
                showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)
    st.markdown("---")

    # ── vs S&P 500 ─────────────────────────────────────────────────────────────
    st.subheader(f"{ticker} vs S&P 500 — last 12 months")
    st.caption("Both indexed to 100 at the start of the period — so you're comparing % gain/loss, not price.")

    spy_12 = spy[spy.index >= pd.Timestamp(cutoff).tz_localize(spy.index.tz)] if not spy.empty else pd.DataFrame()

    if not spy_12.empty and len(df) > 0:
        stk_idx = df["Close"] / df["Close"].iloc[0] * 100
        spy_idx = spy_12["Close"] / spy_12["Close"].iloc[0] * 100
        stk_end = stk_idx.iloc[-1]
        spy_end = spy_idx.iloc[-1]
        diff    = stk_end - spy_end

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(x=stk_idx.index, y=stk_idx.values,
            mode="lines", name=ticker, line=dict(color=LINE_COL, width=2.5)))
        fig_vs.add_trace(go.Scatter(x=spy_idx.index, y=spy_idx.values,
            mode="lines", name="S&P 500", line=dict(color=SPY_COL, width=2, dash="dash")))
        fig_vs.add_hline(y=100, line_color="#333", line_dash="dot", line_width=1)
        fig_vs.update_layout(**BASE_LAYOUT, height=380,
            legend=dict(font=dict(size=15), orientation="h", y=-0.12),
            xaxis=dict(**AXIS), yaxis=dict(**AXIS, title="Indexed (start = 100)"))
        st.plotly_chart(fig_vs, use_container_width=True)

        if diff > 10:
            insight(f"<b>{ticker} is outperforming the S&P 500 by {diff:+.1f} points</b> over the past year. "
                    f"This means your stock is beating the broad market — the move is stock-specific, not just "
                    f"a rising tide lifting all boats.", "ok")
        elif diff < -10:
            insight(f"<b>{ticker} is underperforming the S&P 500 by {abs(diff):.1f} points</b> over the past year. "
                    f"If the market is also down, part of your losses are just the market. "
                    f"But {ticker} has an additional stock-specific drag worth understanding.", "warn")
        else:
            insight(f"<b>{ticker} is roughly in line with the S&P 500</b> (difference: {diff:+.1f} points). "
                    f"The stock is moving broadly with the market.", "neu")
    else:
        st.info("S&P 500 comparison data not available.")
    st.markdown("---")

    # ── price + MAs + volume + RSI ─────────────────────────────────────────────
    st.subheader("Price, Moving Averages & RSI")

    ma50  = df["MA50"].dropna()
    ma200 = df["MA200"].dropna()
    rsi   = df["RSI"].dropna()
    latest_rsi   = rsi.iloc[-1]   if not rsi.empty   else None
    latest_ma50  = ma50.iloc[-1]  if not ma50.empty  else None
    latest_ma200 = ma200.iloc[-1] if not ma200.empty else None

    vol_colors = [UP_COL if r >= 0 else DOWN_COL for r in returns]

    fig_ma = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.20, 0.25],
                            vertical_spacing=0.03)
    fig_ma.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
        name="Price", line=dict(color=LINE_COL, width=2.5)), row=1, col=1)
    if not ma50.empty:
        fig_ma.add_trace(go.Scatter(x=ma50.index, y=ma50.values, mode="lines",
            name="50-day MA", line=dict(color=MA50_COL, width=1.8, dash="dot")), row=1, col=1)
    if not ma200.empty:
        fig_ma.add_trace(go.Scatter(x=ma200.index, y=ma200.values, mode="lines",
            name="200-day MA", line=dict(color=MA200_COL, width=1.8, dash="dash")), row=1, col=1)
    fig_ma.add_trace(go.Bar(x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, marker_line_width=0), row=2, col=1)
    if not rsi.empty:
        fig_ma.add_trace(go.Scatter(x=rsi.index, y=rsi.values, mode="lines",
            name="RSI (14)", line=dict(color="#333", width=1.8),
            fill="tozeroy", fillcolor="rgba(26,86,219,0.06)"), row=3, col=1)
        fig_ma.add_hline(y=70, line_color=DOWN_COL, line_dash="dash", line_width=1,
                         annotation_text="Overbought 70", annotation_font_size=12,
                         annotation_font_color=DOWN_COL, row=3, col=1)
        fig_ma.add_hline(y=30, line_color=UP_COL, line_dash="dash", line_width=1,
                         annotation_text="Oversold 30", annotation_font_size=12,
                         annotation_font_color=UP_COL, row=3, col=1)

    fig_ma.update_layout(**BASE_LAYOUT, height=580,
        legend=dict(font=dict(size=14), orientation="h", y=-0.06))
    fig_ma.update_xaxes(**AXIS)
    fig_ma.update_yaxes(**AXIS)
    fig_ma.update_yaxes(title_text="Price ($)", title_font=dict(size=13), row=1, col=1)
    fig_ma.update_yaxes(title_text="Volume",    title_font=dict(size=13), row=2, col=1)
    fig_ma.update_yaxes(title_text="RSI",       title_font=dict(size=13), row=3, col=1, range=[0, 100])
    st.plotly_chart(fig_ma, use_container_width=True)

    if latest_ma50 and latest_ma200:
        above50  = current_price > latest_ma50
        above200 = current_price > latest_ma200
        golden   = latest_ma50 > latest_ma200

        ma_msg = (f"<b>50-day MA: ${latest_ma50:.2f}</b> &nbsp;|&nbsp; "
                  f"<b>200-day MA: ${latest_ma200:.2f}</b><br><br>")
        if above200 and above50:
            ma_msg += (f"Price is <b>above both moving averages</b> — a classically bullish setup. "
                       f"The 50-day MA acts as near-term support; the 200-day MA is the big-picture "
                       f"trend line. As long as the price stays above both, the long-term trend is intact.")
            if golden:
                ma_msg += (" The 50-day MA is also above the 200-day MA — this is called a "
                           "<b>Golden Cross</b>, which many investors treat as a strong buy signal.")
            kind = "ok"
        elif above200 and not above50:
            ma_msg += (f"Price is <b>below the 50-day MA but above the 200-day MA</b>. "
                       f"Short-term momentum has weakened, but the long-term uptrend is still intact. "
                       f"This is often a normal pullback within an uptrend — not necessarily alarming.")
            kind = "neu"
        elif not above200 and above50:
            ma_msg += (f"Price is <b>below the 200-day MA but above the 50-day MA</b>. "
                       f"The long-term trend has turned negative, but the stock is showing short-term "
                       f"strength. Watch whether price can reclaim the 200-day MA — that would be a "
                       f"meaningful recovery signal.")
            kind = "warn"
        else:
            ma_msg += (f"Price is <b>below both moving averages</b> — a bearish setup in the eyes of "
                       f"most technical analysts. ")
            if not golden:
                ma_msg += ("The 50-day MA has also crossed <b>below</b> the 200-day MA — this is "
                           "called a <b>Death Cross</b>, which is often seen as a significant "
                           "warning signal. It does not mean the stock can't recover, but it "
                           "suggests the market trend is clearly negative.")
            kind = "warn"
        insight(ma_msg, kind)

    if latest_rsi is not None:
        if latest_rsi >= 70:
            rsi_msg  = (f"<b>RSI: {latest_rsi:.0f} — Overbought.</b> "
                        f"The RSI measures how fast and how much the stock has moved recently. "
                        f"Above 70 means the stock has run up quickly and may be due for a pullback. "
                        f"It does not mean you must sell — but it does say momentum is stretched.")
            rsi_kind = "warn"
        elif latest_rsi <= 30:
            rsi_msg  = (f"<b>RSI: {latest_rsi:.0f} — Oversold.</b> "
                        f"The stock has fallen sharply in a short time. An RSI below 30 often "
                        f"means selling pressure is exhausted and a bounce is likely. "
                        f"This is the classic signal that a drop may have gone too far, too fast. "
                        f"Combined with a strong fundamental thesis, this can be a buying opportunity.")
            rsi_kind = "ok"
        elif latest_rsi >= 50:
            rsi_msg  = (f"<b>RSI: {latest_rsi:.0f} — Bullish momentum.</b> "
                        f"Above 50 means buyers are in control. No extreme reading in either direction.")
            rsi_kind = "ok"
        else:
            rsi_msg  = (f"<b>RSI: {latest_rsi:.0f} — Bearish momentum.</b> "
                        f"Below 50 means sellers have had the upper hand recently. "
                        f"Not yet oversold (that would be below 30), but momentum is negative.")
            rsi_kind = "warn"
        insight(rsi_msg, rsi_kind)
    st.markdown("---")

    # ── daily moves ────────────────────────────────────────────────────────────
    st.subheader("Daily moves — every trading day")
    st.caption("Each bar is one trading day. Green = up, red = down. Height shows how big the move was.")

    daily_colors = [UP_COL if r >= 0 else DOWN_COL for r in returns]
    fig_daily = go.Figure(go.Bar(x=returns.index, y=returns.values,
        marker_color=daily_colors, marker_line_width=0,
        hovertemplate="%{x|%b %d, %Y}:  %{y:+.2f}%<extra></extra>"))
    fig_daily.add_hline(y=0, line_color="#333333", line_width=1.2)
    fig_daily.add_trace(go.Scatter(x=[returns.index[-1]], y=[latest_ret],
        mode="markers+text", marker=dict(color="#e6a817", size=12, symbol="diamond"),
        text=[f"  Today: {latest_ret:+.1f}%"], textposition="top right",
        textfont=dict(size=13, color="#b07a00"), showlegend=False))
    fig_daily.update_layout(**BASE_LAYOUT, height=420, showlegend=False, bargap=0.15,
        xaxis=dict(**AXIS), yaxis=dict(**AXIS, title="Daily move (%)", ticksuffix="%"))
    st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown("---")

    # ── after a big drop ───────────────────────────────────────────────────────
    st.subheader(f"After a drop ≥ {thr}% — what happened after 30 days?")
    drop_idx = returns[returns <= -thr].index

    if drop_idx.empty:
        st.info(f"No drops ≥ {thr}% in the last 12 months. Try a lower threshold.")
    else:
        n_drops     = len(drop_idx)
        LOOKFORWARD = 30
        st.caption(
            f"Each bar shows how many of the {n_drops} past drops ended up at that % level "
            f"30 trading days later. Green = recovered, red = still down.")

        outcomes_30 = []
        for d in drop_idx:
            loc = df.index.get_indexer([d])[0]
            if loc >= 0 and loc + LOOKFORWARD < len(df):
                outcomes_30.append(
                    (df["Close"].iloc[loc + LOOKFORWARD] / df["Close"].iloc[loc] - 1) * 100)

        if outcomes_30:
            spread = max(abs(min(outcomes_30)), abs(max(outcomes_30)))
            bin_w  = max(5, round(spread / 6 / 5) * 5)
            lo     = (min(outcomes_30) // bin_w) * bin_w - bin_w
            hi     = (max(outcomes_30) // bin_w) * bin_w + bin_w * 2
            edges  = np.arange(lo, hi, bin_w)
            counts, _ = np.histogram(outcomes_30, bins=edges)
            mids   = (edges[:-1] + edges[1:]) / 2
            colors = [UP_COL if m >= 0 else DOWN_COL for m in mids]

            n_tot_hist  = len(outcomes_30)
            _annotations = []
            _ymax = max(counts) * 1.65
            _step = _ymax * 0.10

            for _mid, _cnt, _col in zip(mids, counts, colors):
                if _cnt > 0:
                    _ypos = _cnt + _ymax * 0.03
                    _annotations.append(dict(x=_mid, y=_ypos,
                        text=f"<b>{_mid:+.0f}%</b>",
                        font=dict(size=22, color=_col), showarrow=False,
                        xanchor="center", yanchor="bottom"))
                    _annotations.append(dict(x=_mid, y=_ypos + _step,
                        text=f"<b>{_cnt/n_tot_hist*100:.0f}%</b> of drops",
                        font=dict(size=14, color=FONT_COL), showarrow=False,
                        xanchor="center", yanchor="bottom"))
                    _annotations.append(dict(x=_mid, y=_ypos + _step * 2,
                        text=f"{_cnt} drop{'s' if _cnt!=1 else ''}",
                        font=dict(size=11, color="#888"), showarrow=False,
                        xanchor="center", yanchor="bottom"))

            fig_hist30 = go.Figure(go.Bar(
                x=mids, y=counts, marker_color=colors,
                marker_line_width=0.5, marker_line_color="white",
                width=bin_w * 0.85, cliponaxis=False,
                hovertemplate="%{y} drop(s) ended around %{x:.0f}%<extra></extra>"))
            avg_30 = np.mean(outcomes_30)
            fig_hist30.add_vline(x=0, line_color="#333", line_dash="dash", line_width=2,
                                 annotation_text=" Break-even", annotation_font_size=13)
            fig_hist30.add_vline(x=avg_30, line_color="#e6a817", line_dash="dot", line_width=2,
                                 annotation_text=f" Avg: {avg_30:+.1f}%",
                                 annotation_font_color="#b07a00", annotation_font_size=13)
            fig_hist30.update_layout(
                **{**BASE_LAYOUT, "margin": dict(l=10, r=10, t=50, b=10)},
                height=460,
                xaxis=dict(**AXIS, title="Price change 30 days after the drop (%)", ticksuffix="%",
                           title_font=dict(size=14)),
                yaxis=dict(**AXIS, title="Number of drops", title_font=dict(size=14),
                           dtick=1, range=[0, max(counts) * 1.65]),
                showlegend=False, annotations=_annotations)
            st.plotly_chart(fig_hist30, use_container_width=True)

            n_pos = sum(1 for x in outcomes_30 if x >= 0)
            n_tot = len(outcomes_30)
            if n_pos >= n_tot * 0.65 and avg_30 >= 3:
                insight(f"<b>After big drops, this stock has tended to recover within 30 days.</b> "
                        f"{n_pos} out of {n_tot} past drops ({n_pos/n_tot*100:.0f}%) ended higher after a month, "
                        f"with an average gain of {avg_30:+.1f}%. "
                        f"History supports holding through the dip — assuming your thesis is intact.", "ok")
            elif n_pos <= n_tot * 0.4 or avg_30 <= -3:
                insight(f"<b>Big drops here have not reliably recovered within 30 days.</b> "
                        f"Only {n_pos} out of {n_tot} ({n_pos/n_tot*100:.0f}%) ended higher after a month "
                        f"(average: {avg_30:+.1f}%). Worth re-examining whether the thesis has changed "
                        f"when a big drop occurs.", "warn")
            else:
                insight(f"<b>Mixed outcomes after big drops.</b> "
                        f"{n_pos}/{n_tot} ended higher after 30 days (average: {avg_30:+.1f}%). "
                        f"No strong pattern — each drop needs to be judged on its own merits.", "neu")

        st.markdown("##### Breakdown by time window")
        windows = [1, 2, 3, 5, 10, 20]
        for w in windows:
            outcomes = []
            for d in drop_idx:
                loc = df.index.get_indexer([d])[0]
                if loc >= 0 and loc + w < len(df):
                    outcomes.append((df["Close"].iloc[loc+w] / df["Close"].iloc[loc] - 1) * 100)
            if not outcomes: continue
            n_pos  = sum(1 for x in outcomes if x > 0)
            n_tot  = len(outcomes)
            avg    = np.mean(outcomes)
            pct    = n_pos / n_tot * 100
            color  = UP_COL if pct >= 55 else DOWN_COL if pct < 45 else "#888800"
            arrow  = "↑" if avg >= 0 else "↓"
            filled = int(round(pct / 10))
            bar    = "█" * filled + "░" * (10 - filled)
            label  = f"{w} day{'s' if w > 1 else ''} later"
            st.markdown(
                f"<div style='font-size:1.15rem;padding:8px 0;border-bottom:1px solid #e8e8e8;'>"
                f"<span style='display:inline-block;width:140px;color:#555;'>{label}</span>"
                f"<span style='color:{color};font-weight:700;font-size:1.25rem;'>"
                f"&nbsp;{n_pos} out of {n_tot} ({pct:.0f}%)&nbsp;</span>"
                f"<span style='color:#555;'>times higher &nbsp;</span>"
                f"<span style='color:{color};letter-spacing:2px;'>{bar}&nbsp;</span>"
                f"<span style='color:#555;font-size:0.95rem;'>avg {arrow}{abs(avg):.1f}%</span>"
                f"</div>", unsafe_allow_html=True)
        st.caption("Measured from the closing price on the day of the drop.")
    st.markdown("---")

    # ── all big moves table ────────────────────────────────────────────────────
    st.subheader(f"All big moves  (≥ ±{thr}%)")
    big_moves = df[abs(returns) >= thr][["Close", "Return"]].copy()
    big_moves = big_moves.sort_index(ascending=False)
    for w, col in [(1, "+1 day (%)"), (5, "+5 days (%)"), (20, "+20 days (%)")]:
        big_moves[col] = np.nan
        for idx in big_moves.index:
            loc = df.index.get_indexer([idx])[0]
            if loc >= 0 and loc + w < len(df):
                big_moves.loc[idx, col] = (df["Close"].iloc[loc+w] / df["Close"].iloc[loc] - 1) * 100
    big_moves.index = big_moves.index.strftime("%b %d, %Y")
    big_moves.columns = ["Close ($)", "Move (%)", "+ 1 day (%)", "+ 5 days (%)", "+ 20 days (%)"]

    def color_val(val):
        if pd.isna(val): return ""
        return f"color: {UP_COL if val > 0 else DOWN_COL}; font-weight: 600"

    styled = (big_moves.style
        .format({"Close ($)": "{:.2f}", "Move (%)": "{:+.1f}",
                 "+ 1 day (%)": "{:+.1f}", "+ 5 days (%)": "{:+.1f}", "+ 20 days (%)": "{:+.1f}"},
                na_rep="—")
        .map(color_val, subset=["Move (%)", "+ 1 day (%)", "+ 5 days (%)", "+ 20 days (%)"]))
    st.dataframe(styled, use_container_width=True, height=400)
    st.markdown("---")

    # ── market positioning ─────────────────────────────────────────────────────
    st.subheader("Market positioning")

    short_pct   = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")

    if short_pct is not None:
        sp = short_pct * 100
        if sp > 20:
            si_msg  = (f"<b>Short interest: {sp:.1f}% of float</b> "
                       f"({'days to cover: ' + str(round(short_ratio,1)) if short_ratio else ''})<br><br>"
                       f"This is <b>very high</b>. A large share of investors are betting the stock will fall. "
                       f"This can work two ways: (1) it may reflect genuine concern about the company's prospects, "
                       f"or (2) if good news arrives, all those short-sellers will rush to close their positions "
                       f"by buying the stock, causing a violent upward spike — a <b>short squeeze</b>. "
                       f"High short interest in a volatile stock like this is a major source of extreme up-days.")
            si_kind = "warn"
        elif sp > 10:
            si_msg  = (f"<b>Short interest: {sp:.1f}% of float</b><br><br>"
                       f"<b>Elevated</b>. A meaningful minority of investors are short this stock. "
                       f"Worth monitoring — rising short interest signals growing skepticism.")
            si_kind = "warn"
        elif sp > 5:
            si_msg  = (f"<b>Short interest: {sp:.1f}% of float</b><br><br>"
                       f"<b>Moderate</b>. Some short interest but not unusual for a volatile small/mid cap.")
            si_kind = "neu"
        else:
            si_msg  = (f"<b>Short interest: {sp:.1f}% of float</b><br><br>"
                       f"<b>Low</b>. The market is not heavily betting against this stock.")
            si_kind = "ok"
        insight(si_msg, si_kind)
    else:
        st.caption("Short interest data not available.")

    rec_mean   = info.get("recommendationMean")
    n_analysts = info.get("numberOfAnalystOpinions", 0)

    if rec_mean is not None and n_analysts:
        if   rec_mean <= 1.5: cons_label, cons_color = "Strong Buy",  UP_COL
        elif rec_mean <= 2.5: cons_label, cons_color = "Buy",         "#2d9e5f"
        elif rec_mean <= 3.5: cons_label, cons_color = "Hold",        "#888800"
        elif rec_mean <= 4.5: cons_label, cons_color = "Sell",        "#cc6600"
        else:                 cons_label, cons_color = "Strong Sell", DOWN_COL

        rec_msg = (
            f"<b>Analyst consensus: "
            f"<span style='color:{cons_color}'>{cons_label}</span></b> "
            f"(score {rec_mean:.1f}/5.0, based on {n_analysts} analysts)<br><br>"
            f"Analysts rate stocks on a 1–5 scale: 1 = Strong Buy, 3 = Hold, 5 = Strong Sell. "
            f"A score of {rec_mean:.1f} puts this stock in the <b>{cons_label}</b> camp. "
            f"Note: analyst consensus tends to lag reality — it is a useful sanity check, "
            f"not a precise timing signal.")
        insight(rec_msg, "ok" if rec_mean <= 2.5 else "warn" if rec_mean >= 3.5 else "neu")

    beta = info.get("beta")
    if beta:
        if beta > 2:
            b_msg  = (f"<b>Beta: {beta:.2f}</b><br><br>"
                      f"Beta measures how much this stock moves relative to the overall market. "
                      f"A beta of {beta:.2f} means that when the S&P 500 moves 1%, this stock "
                      f"tends to move ~{beta:.1f}%. This is a <b>very high-volatility stock</b> — "
                      f"it amplifies both gains and losses significantly. On bad market days, "
                      f"expect this to fall harder than the index; on good days, it should rise more.")
            b_kind = "warn"
        elif beta > 1.5:
            b_msg  = f"<b>Beta: {beta:.2f}</b> — significantly more volatile than the market."
            b_kind = "warn"
        elif beta > 1:
            b_msg  = f"<b>Beta: {beta:.2f}</b> — slightly more volatile than the market."
            b_kind = "neu"
        else:
            b_msg  = f"<b>Beta: {beta:.2f}</b> — less volatile than the market."
            b_kind = "ok"
        insight(b_msg, b_kind)

    if cal is not None:
        try:
            if isinstance(cal, dict):
                earn_dates = cal.get("Earnings Date", [])
            elif isinstance(cal, pd.DataFrame):
                earn_dates = cal.loc["Earnings Date"].dropna().tolist() if "Earnings Date" in cal.index else []
            else:
                earn_dates = []

            future_dates = [d for d in earn_dates
                            if (pd.Timestamp(d) - pd.Timestamp(datetime.today())).days >= 0]
            if future_dates:
                next_earn = pd.Timestamp(future_dates[0])
                days_away = (next_earn - pd.Timestamp(datetime.today())).days
                if 0 <= days_away <= 14:
                    earn_msg  = (f"<b>⚠️ Earnings in {days_away} days ({next_earn.strftime('%b %d, %Y')})</b><br><br>"
                                 f"Earnings announcements are the single biggest source of large single-day moves "
                                 f"for stocks like this. Expect elevated volatility around this date — "
                                 f"both before (speculation) and after (reaction).")
                    earn_kind = "warn"
                elif 0 <= days_away <= 45:
                    earn_msg  = (f"<b>Next earnings: {next_earn.strftime('%b %d, %Y')} ({days_away} days away)</b><br><br>"
                                 f"Earnings are coming up. Volatility often increases in the weeks before "
                                 f"as investors position themselves.")
                    earn_kind = "neu"
                else:
                    earn_msg  = f"<b>Next earnings: {next_earn.strftime('%b %d, %Y')} ({days_away} days away)</b>"
                    earn_kind = "neu"
                insight(earn_msg, earn_kind)
        except Exception:
            pass

    avg_vol_20 = df["Volume"].rolling(20).mean().iloc[-1]
    today_vol  = df["Volume"].iloc[-1]
    vol_ratio  = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1

    if vol_ratio >= 2.0:
        if latest_ret < 0:
            insight(f"<b>⚠️ Very high volume: {vol_ratio:.1f}× the 20-day average on a down day.</b><br><br>"
                    f"This selloff has real conviction — many investors are actively choosing to exit. "
                    f"A high-volume drop is more meaningful than a low-volume drift down.", "warn")
        else:
            insight(f"<b>📣 Very high volume: {vol_ratio:.1f}× the 20-day average on an up day.</b><br><br>"
                    f"This rally has broad participation — strong buying conviction behind the move.", "ok")
    elif vol_ratio >= 1.5:
        if latest_ret < 0:
            insight(f"<b>Volume is elevated today: {vol_ratio:.1f}× the 20-day average.</b> "
                    f"Moderately above normal on a down day — worth watching but not alarming.", "warn")
        else:
            insight(f"<b>Volume is elevated today: {vol_ratio:.1f}× the 20-day average.</b> "
                    f"Moderately above normal on an up day — some additional buying interest.", "ok")
    elif vol_ratio <= 0.4:
        insight(f"<b>Volume today is very low</b> ({vol_ratio:.1f}× the 20-day average).<br><br>"
                f"Low-volume moves are less reliable — price can be pushed around with fewer trades. "
                f"Don't read too much into today's price action.", "neu")
    else:
        insight(f"<b>Volume today is normal</b> ({vol_ratio:.1f}× the 20-day average) — "
                f"no unusual trading activity.", "neu")
    st.markdown("---")

    # ── cash & debt health ─────────────────────────────────────────────────────
    st.subheader("Cash & debt health")
    st.caption("A company's financial strength — can it survive a rough patch?")

    total_cash    = info.get("totalCash")
    total_debt    = info.get("totalDebt")
    free_cf       = info.get("freeCashflow")
    current_ratio = info.get("currentRatio")

    def fmt_big(v):
        if v is None: return "n/a"
        if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6: return f"${v/1e6:.0f}M"
        return f"${v:,.0f}"

    cd1, cd2, cd3, cd4 = st.columns(4)
    cd1.metric("Cash on hand",            fmt_big(total_cash))
    cd2.metric("Total debt",              fmt_big(total_debt))
    net_cash = (total_cash or 0) - (total_debt or 0)
    cd3.metric("Net cash / debt",         fmt_big(net_cash))
    cd4.metric("Free cash flow (annual)", fmt_big(free_cf))
    st.markdown("")

    if total_cash is not None and total_debt is not None:
        if net_cash > 0:
            insight(f"<b>Net cash position: {fmt_big(net_cash)} (cash exceeds debt).</b><br><br>"
                    f"The company has more cash than debt — a healthy sign. "
                    f"It has a financial cushion to weather downturns, invest in growth, "
                    f"or buy back shares without needing to borrow.", "ok")
        else:
            insight(f"<b>Net debt position: {fmt_big(abs(net_cash))} (debt exceeds cash).</b><br><br>"
                    f"The company owes more than it holds in cash. This is not automatically bad "
                    f"— many healthy companies carry debt — but it means financial stress during "
                    f"a downturn would hit harder.", "warn")

    if free_cf is not None:
        if free_cf > 0:
            insight(f"<b>Free cash flow: {fmt_big(free_cf)} positive.</b> "
                    f"The business is generating real cash after all expenses and investments. "
                    f"This is the lifeblood of a company — positive FCF means it doesn't need "
                    f"to keep raising money to survive.", "ok")
        else:
            if total_cash and total_cash > 0 and free_cf < 0:
                quarters = (total_cash / abs(free_cf)) * 4
                insight(f"<b>Free cash flow is negative ({fmt_big(free_cf)}/year).</b><br><br>"
                        f"The company is burning cash. At this rate, the current cash pile "
                        f"({fmt_big(total_cash)}) would last roughly "
                        f"<b>{quarters:.0f} quarters ({quarters/4:.1f} years)</b> "
                        f"before needing to raise more money. "
                        f"If growth justifies it, this can be fine — but it is a risk to monitor.", "warn")
            else:
                insight(f"<b>Free cash flow is negative.</b> The company is spending more than it earns. "
                        f"Common for high-growth companies, but increases dependency on external financing.", "warn")

    if current_ratio is not None:
        if current_ratio >= 2:
            insight(f"<b>Current ratio: {current_ratio:.1f}</b> — very healthy. "
                    f"The company has {current_ratio:.1f}× more short-term assets than short-term liabilities. "
                    f"No near-term liquidity risk.", "ok")
        elif current_ratio >= 1:
            insight(f"<b>Current ratio: {current_ratio:.1f}</b> — adequate. "
                    f"Short-term obligations are covered, with some buffer.", "neu")
        else:
            insight(f"<b>Current ratio: {current_ratio:.1f}</b> — below 1. "
                    f"Short-term liabilities exceed short-term assets. "
                    f"The company may face cash pressure if conditions tighten.", "warn")
    st.markdown("---")

    # ── fundamentals ──────────────────────────────────────────────────────────
    st.subheader("Fundamentals")
    if fin.empty:
        st.info("No financial data available for this ticker.")
    else:
        rev_hist = hist_series(fin, "Total Revenue", "Revenue")
        rev_fwd  = est_series(rev_est)
        st.plotly_chart(bar_chart(rev_hist, rev_fwd, "Revenue", "Revenue ($)", fmt_billions=True),
                        use_container_width=True)

        earn_hist = hist_series(fin, "Net Income", "Net Income Common Stockholders",
                                "Net Income From Continuing Operation Net Minority Interest")
        shares_out = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        eps_fwd_ni = est_series(earn_est)
        earn_fwd   = (eps_fwd_ni * shares_out) if shares_out and not eps_fwd_ni.empty else pd.Series(dtype=float)
        st.plotly_chart(bar_chart(earn_hist, earn_fwd, "Net Earnings", "Net Income ($)", fmt_billions=True),
                        use_container_width=True)

        eps_hist = hist_series(fin, "Basic EPS", "Diluted EPS",
                               "Normalized Basic EPS", "Normalized Diluted EPS")
        eps_fwd  = est_series(earn_est)
        st.plotly_chart(bar_chart(eps_hist, eps_fwd, "Earnings Per Share (EPS)", "EPS ($)"),
                        use_container_width=True)

        pe_hist = pd.Series(dtype=float)
        if not eps_hist.empty:
            pe_rows = {}
            for yr, eps_val in eps_hist.items():
                if eps_val and eps_val > 0:
                    try:
                        yr_end = df_full[df_full.index.year == yr]
                        if not yr_end.empty:
                            pe_rows[yr] = yr_end["Close"].iloc[-1] / eps_val
                    except Exception:
                        pass
            pe_hist = pd.Series(pe_rows, dtype=float)

        pe_fwd = pd.Series(dtype=float)
        if not eps_fwd.empty:
            pe_fwd = (current_price / eps_fwd).pipe(lambda s: s[s > 0])

        st.plotly_chart(bar_chart(pe_hist, pe_fwd, "Price / Earnings (P/E)", "P/E (x)"),
                        use_container_width=True)
