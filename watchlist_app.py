import warnings
warnings.filterwarnings("ignore")

import json
import io
import os
import pandas as pd
import streamlit as st
from stock_detail import apply_styles, render, load_summary, UP_COL, DOWN_COL, FONT_COL

st.set_page_config(page_title="My Watchlist", page_icon="📊", layout="wide")
apply_styles()

# ── extra styles specific to the watchlist ─────────────────────────────────────
st.markdown("""
<style>
  .wl-header {
    font-size: 1.05rem; font-weight: 700; color: #555;
    padding: 6px 0; border-bottom: 2px solid #dde3f5;
    margin-bottom: 4px;
  }
  .wl-row {
    padding: 7px 0;
    border-bottom: 1px solid #f0f0f0;
    line-height: 1.3;
  }
  .wl-ticker {
    font-size: 1.1rem; font-weight: 800; color: #1a56db;
    cursor: pointer;
  }
  .wl-name {
    font-size: 0.85rem; color: #777; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .wl-price    { font-size: 1.05rem; font-weight: 700; }
  .wl-pct-up   { color: #1aaa55; font-weight: 700; }
  .wl-pct-down { color: #cc3300; font-weight: 700; }
  .wl-neutral  { color: #888; }
  .group-title {
    font-size: 1.25rem; font-weight: 700; color: #1a56db;
    padding: 10px 0 4px 0;
    border-bottom: 2px solid #1a56db;
    margin-top: 18px; margin-bottom: 6px;
  }
  .preview-group {
    font-size: 1rem; font-weight: 700; color: #444;
    margin: 10px 0 4px 0;
  }
</style>
""", unsafe_allow_html=True)

WATCHLIST_PATH    = os.path.join(os.path.dirname(__file__), "watchlist.json")
TICKER_GROUPS_PATH = os.path.join(os.path.dirname(__file__), "ticker_groups.json")


# ── persistence helpers ────────────────────────────────────────────────────────
def load_watchlist():
    with open(WATCHLIST_PATH, "r") as f:
        return json.load(f)

def save_watchlist(data):
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(data, f, indent=2)

def load_ticker_groups():
    with open(TICKER_GROUPS_PATH, "r") as f:
        return json.load(f)

def save_ticker_groups(data):
    with open(TICKER_GROUPS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def all_tickers(data):
    return [s["ticker"] for g in data["groups"] for s in g["stocks"]]

def add_stock(data, ticker, group_name, weight=0.0):
    ticker = ticker.upper().strip()
    if ticker in all_tickers(data):
        return False, "Already in watchlist."
    for g in data["groups"]:
        if g["name"] == group_name:
            g["stocks"].append({"ticker": ticker, "weight": weight})
            save_watchlist(data)
            return True, "Added."
    return False, "Group not found."

def remove_stock(data, ticker):
    for g in data["groups"]:
        g["stocks"] = [s for s in g["stocks"] if s["ticker"] != ticker]
    save_watchlist(data)

def move_stock(data, ticker, new_group_name):
    stock_entry = None
    for g in data["groups"]:
        match = [s for s in g["stocks"] if s["ticker"] == ticker]
        if match:
            stock_entry = match[0]
            g["stocks"] = [s for s in g["stocks"] if s["ticker"] != ticker]
            break
    if stock_entry is None:
        return
    for g in data["groups"]:
        if g["name"] == new_group_name:
            g["stocks"].append(stock_entry)
            break
    else:
        data["groups"].append({"name": new_group_name, "stocks": [stock_entry]})
    save_watchlist(data)


# ── paste parser ───────────────────────────────────────────────────────────────
def parse_weight(val):
    """Convert weight string to float percentage.
    If the value had a % sign, trust it as-is (0.69% stays 0.69).
    Only multiply by 100 if no % sign and value looks like a decimal fraction (e.g. 0.0069)."""
    s = str(val).strip()
    has_pct = "%" in s
    s = s.replace("%", "").replace(",", ".")
    try:
        v = float(s)
        if not has_pct and 0 < v < 0.1:   # e.g. 0.0069 → 0.69%
            v = v * 100
        return round(v, 4)
    except ValueError:
        return 0.0

def parse_paste(text):
    """Parse pasted text (tab- or comma-separated) into rows of {ticker, weight}.
    Skips header rows and blank lines automatically."""
    rows = []
    for line in text.strip().splitlines():
        # split on tab first (Excel copy), then comma, then whitespace
        if "\t" in line:
            parts = line.split("\t")
        elif "," in line:
            parts = line.split(",")
        else:
            parts = line.split()

        if len(parts) < 2:
            continue

        sym    = parts[0].strip().upper()
        weight = parts[1].strip()

        # skip header rows
        if sym.lower() in ("symbol", "ticker", "tick", "sym", ""):
            continue
        # skip rows where ticker looks like a number
        if sym.replace(".", "").replace("-", "").isdigit():
            continue

        rows.append({"ticker": sym, "weight": parse_weight(weight)})
    return rows

def build_watchlist_from_upload(rows, tg_data):
    """
    rows: list of {"ticker": str, "weight": float}
    Returns a new watchlist dict using the ticker_groups mapping.
    Also returns a list of unknown tickers (mapped to Other).
    Updates ticker_groups.json with any new tickers assigned to Other.
    """
    mapping    = tg_data["tickers"]
    group_order = tg_data["group_order"]

    # ensure Other is in group_order
    if "Other" not in group_order:
        group_order.append("Other")

    groups = {name: [] for name in group_order}
    unknown = []

    for row in rows:
        sym    = row["ticker"]
        weight = row["weight"]
        group  = mapping.get(sym, "Other")
        if group not in groups:
            groups[group] = []
        groups[group].append({"ticker": sym, "weight": weight})
        if sym not in mapping:
            unknown.append(sym)
            # persist new ticker as Other in mapping
            mapping[sym] = "Other"

    # save updated mapping if new tickers were added
    if unknown:
        save_ticker_groups(tg_data)

    watchlist = {
        "groups": [
            {"name": name, "stocks": groups[name]}
            for name in group_order
            if groups.get(name)   # skip empty groups
        ]
    }
    return watchlist, unknown


# ── session state ──────────────────────────────────────────────────────────────
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "thr" not in st.session_state:
    st.session_state.thr = 5
if "upload_preview" not in st.session_state:
    st.session_state.upload_preview = None   # holds parsed rows before confirm


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL VIEW
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.selected_ticker:
    ticker = st.session_state.selected_ticker

    col_back, col_set = st.columns([8, 2])
    with col_back:
        if st.button("← Back to Watchlist", key="back_btn"):
            st.session_state.selected_ticker = None
            st.rerun()
    with col_set:
        with st.expander("⚙️ Settings"):
            st.session_state.thr = st.slider(
                "'Big move' threshold (%)", 3, 15, st.session_state.thr, key="detail_thr")
            st.caption("Data via Yahoo Finance · last 12 months")

    render(ticker, st.session_state.thr)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST VIEW
# ══════════════════════════════════════════════════════════════════════════════
wl_data  = load_watchlist()
tg_data  = load_ticker_groups()
group_names = [g["name"] for g in wl_data["groups"]]

# ── page header ────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([7, 3])
with hdr_left:
    st.markdown("<h1 style='margin-bottom:0'>📊 My Watchlist</h1>", unsafe_allow_html=True)
    st.caption("Click any ticker to open the full analysis. Data refreshes every 5 minutes.")
with hdr_right:
    edit_label = "✅ Done editing" if st.session_state.edit_mode else "✏️ Edit watchlist"
    if st.button(edit_label, key="edit_toggle"):
        st.session_state.edit_mode = not st.session_state.edit_mode
        st.session_state.upload_preview = None
        st.rerun()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Portfolio summary")

# collect data for all non-CASH stocks that have weights
_all_stocks = [(s["ticker"], s["weight"])
               for g in wl_data["groups"]
               for s in g["stocks"]
               if s["ticker"] != "CASH" and s.get("weight", 0) > 0]
_total_weight = sum(w for _, w in _all_stocks)

# gather summaries (already cached from the watchlist rows below)
_port_rows = []
for sym, weight in _all_stocks:
    s = load_summary(sym)
    if s:
        _port_rows.append({
            "ticker":    sym,
            "weight":    weight,
            "today_pct": s["today_pct"],
            "pct_5d":    s["pct_5d"],
            "ytd_pct":   s["ytd_pct"],
            "price":     s["price"],
            "beta":      s.get("beta"),
        })

if _port_rows:
    import plotly.graph_objects as go

    # weighted returns (normalise weights to sum to 100 among stocks with data)
    _w_total = sum(r["weight"] for r in _port_rows)

    def _wret(key):
        return sum(r[key] * r["weight"] / _w_total for r in _port_rows)

    port_today = _wret("today_pct")
    port_5d    = _wret("pct_5d")
    port_ytd   = _wret("ytd_pct")

    # ── KPI strip ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    def _kpi_color(v): return UP_COL if v >= 0 else DOWN_COL

    # weighted beta (only stocks with beta data)
    _beta_rows = [r for r in _port_rows if r.get("beta")]
    _beta_w    = sum(r["weight"] for r in _beta_rows)
    port_beta  = sum(r["beta"] * r["weight"] / _beta_w for r in _beta_rows) if _beta_w > 0 else None

    k1.metric("Weighted today",  f"{port_today:+.1f}%")
    k2.metric("Weighted 5-day",  f"{port_5d:+.1f}%")
    k3.metric("Weighted YTD",    f"{port_ytd:+.1f}%")
    k4.metric("Portfolio beta",  f"{port_beta:.2f}" if port_beta else "—",
              help="Weighted avg beta — how much your portfolio moves vs the S&P 500")
    cash_w = next((s["weight"] for g in wl_data["groups"]
                   for s in g["stocks"] if s["ticker"] == "CASH"), 0)
    k5.metric("Cash weight",     f"{cash_w:.2f}%")

    st.markdown("")

    # ── allocation by group (donut) + top holdings (bar) ──────────────────────
    col_donut, col_bar = st.columns(2)

    with col_donut:
        st.markdown("**Allocation by group**")
        group_weights = {}
        for g in wl_data["groups"]:
            gw = sum(s["weight"] for s in g["stocks"])
            if gw > 0:
                group_weights[g["name"]] = gw
        labels = list(group_weights.keys())
        values = list(group_weights.values())
        colors = ["#1a56db","#e6a817","#1aaa55","#9b59b6",
                  "#cc3300","#2d9e5f","#888888","#f0a500"]
        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.5, textinfo="percent",
            marker=dict(colors=colors[:len(labels)]),
            textfont=dict(size=12),
            hovertemplate="%{label}: %{value:.2f}%<extra></extra>"))
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="v", x=1.02, y=0.5,
                        font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
            height=320,
            margin=dict(l=10, r=120, t=10, b=10),
            paper_bgcolor="#ffffff")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        st.markdown("**Top holdings by weight**")
        top = sorted(_port_rows, key=lambda r: r["weight"], reverse=True)[:10]
        bar_colors = [UP_COL if r["ytd_pct"] >= 0 else DOWN_COL for r in top]
        fig_top = go.Figure(go.Bar(
            x=[r["weight"] for r in top],
            y=[r["ticker"] for r in top],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{r['weight']:.2f}%  (YTD {r['ytd_pct']:+.1f}%)" for r in top],
            textposition="outside",
            textfont=dict(size=12),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
        fig_top.update_layout(
            height=320,
            margin=dict(l=10, r=120, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8f9ff",
            xaxis=dict(title="Weight (%)", gridcolor="#dde3f5"),
            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)

    # ── today's winners & losers ───────────────────────────────────────────────
    st.markdown("**Today's movers** (weighted by position size)")
    sorted_today = sorted(_port_rows, key=lambda r: r["today_pct"], reverse=True)
    winners = sorted_today[:3]
    losers  = sorted_today[-3:][::-1]

    wl1, wl2 = st.columns(2)
    with wl1:
        st.markdown("<div style='color:#1aaa55;font-weight:700;margin-bottom:4px'>▲ Top gainers</div>",
                    unsafe_allow_html=True)
        for r in winners:
            st.markdown(
                f"<div style='padding:6px 0;border-bottom:1px solid #f0f0f0;'>"
                f"<b>{r['ticker']}</b> "
                f"<span style='color:{UP_COL};font-weight:700'>{r['today_pct']:+.1f}%</span>"
                f"<span style='color:#888;font-size:0.9rem'> · {r['weight']:.2f}% of portfolio</span>"
                f"</div>", unsafe_allow_html=True)
    with wl2:
        st.markdown("<div style='color:#cc3300;font-weight:700;margin-bottom:4px'>▼ Top losers</div>",
                    unsafe_allow_html=True)
        for r in losers:
            st.markdown(
                f"<div style='padding:6px 0;border-bottom:1px solid #f0f0f0;'>"
                f"<b>{r['ticker']}</b> "
                f"<span style='color:{DOWN_COL};font-weight:700'>{r['today_pct']:+.1f}%</span>"
                f"<span style='color:#888;font-size:0.9rem'> · {r['weight']:.2f}% of portfolio</span>"
                f"</div>", unsafe_allow_html=True)

    # ── concentration warning ──────────────────────────────────────────────────
    _top5_w = sum(r["weight"] for r in sorted(_port_rows, key=lambda r: r["weight"], reverse=True)[:5])
    if _top5_w > 50:
        st.markdown(
            f"<div class='insight insight-warn' style='margin-top:12px'>"
            f"⚠️ <b>Concentration:</b> your top 5 positions account for "
            f"<b>{_top5_w:.1f}%</b> of the portfolio. A single bad earnings "
            f"report in any of those could have an outsized impact.</div>",
            unsafe_allow_html=True)

st.markdown("---")

# ── upload section ─────────────────────────────────────────────────────────────
with st.expander("📋 Paste portfolio (replaces current watchlist)", expanded=False):
    st.markdown(
        "Copy the **Symbol** and **Weight** columns from your Excel/broker and paste below. "
        "Header row is optional — the app handles it automatically.",
        unsafe_allow_html=True)

    pasted = st.text_area(
        "Paste here", height=200, placeholder="AAOI\t8.84%\nAMD\t4.24%\nMU\t10.34%\n…",
        label_visibility="collapsed")

    if st.button("Preview", key="preview_btn", disabled=not pasted.strip()):
        rows = parse_paste(pasted)
        if rows:
            st.session_state.upload_preview = rows
            st.rerun()
        else:
            st.error("No valid tickers found — make sure each line has a ticker and a weight.")

    # ── preview ────────────────────────────────────────────────────────────────
    if st.session_state.upload_preview:
        rows   = st.session_state.upload_preview
        new_wl, unknown = build_watchlist_from_upload(rows, tg_data)

        st.markdown(f"**Preview — {len(rows)} stocks detected, grouped as follows:**")

        for g in new_wl["groups"]:
            if not g["stocks"]:
                continue
            st.markdown(f"<div class='preview-group'>📁 {g['name']}</div>",
                        unsafe_allow_html=True)
            preview_df = pd.DataFrame(g["stocks"]).rename(
                columns={"ticker": "Ticker", "weight": "Weight (%)"})
            preview_df["Weight (%)"] = preview_df["Weight (%)"].map("{:.2f}%".format)
            st.dataframe(preview_df, use_container_width=False, hide_index=True)

        if unknown:
            st.info(f"**{len(unknown)} new ticker(s) auto-assigned to 'Other':** "
                    f"{', '.join(unknown)}. "
                    f"You can reassign them later via the Edit watchlist button, "
                    f"or ask me in chat to move them to the right group.")

        col_confirm, col_cancel, _ = st.columns([1.5, 1.5, 5])
        with col_confirm:
            if st.button("✅ Confirm & replace", type="primary", key="confirm_upload"):
                save_watchlist(new_wl)
                st.cache_data.clear()
                st.session_state.upload_preview = None
                st.success("Watchlist replaced successfully!")
                st.rerun()
        with col_cancel:
            if st.button("❌ Cancel", key="cancel_upload"):
                st.session_state.upload_preview = None
                st.rerun()

# ── edit mode forms ────────────────────────────────────────────────────────────
if st.session_state.edit_mode:
    with st.expander("🗂️ Create new group", expanded=False):
        ng1, ng2 = st.columns([4, 1])
        with ng1:
            new_group_name = st.text_input("Group name", placeholder="e.g. Biotech",
                                           key="new_group_input").strip()
        with ng2:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            if st.button("Create", key="create_group_btn", type="primary"):
                if not new_group_name:
                    st.warning("Enter a group name first.")
                elif new_group_name in [g["name"] for g in wl_data["groups"]]:
                    st.warning(f"'{new_group_name}' already exists.")
                else:
                    wl_data["groups"].append({"name": new_group_name, "stocks": []})
                    save_watchlist(wl_data)
                    tg_data["group_order"].append(new_group_name)
                    save_ticker_groups(tg_data)
                    st.success(f"Group '{new_group_name}' created.")
                    st.rerun()

    with st.expander("➕ Add a stock", expanded=True):
        ac1, ac2, ac3, ac4 = st.columns([2, 2, 2, 1])
        with ac1:
            new_ticker = st.text_input("Ticker", placeholder="e.g. TSLA").upper().strip()
        with ac2:
            new_group  = st.selectbox("Group", group_names)
        with ac3:
            new_weight = st.number_input("Weight (%)", min_value=0.0, max_value=100.0,
                                         value=0.0, step=0.1, format="%.2f")
        with ac4:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            if st.button("Add", key="add_btn", type="primary"):
                if new_ticker:
                    ok, msg = add_stock(wl_data, new_ticker, new_group, new_weight)
                    if ok:
                        # also persist in ticker_groups mapping
                        tg_data["tickers"][new_ticker] = new_group
                        save_ticker_groups(tg_data)
                        st.success(f"{new_ticker} added to {new_group}.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.warning(msg)
                else:
                    st.warning("Enter a ticker first.")

# ── column layout ─────────────────────────────────────────────────────────────
def col_widths():
    if st.session_state.edit_mode:
        return [2, 3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 2.0, 0.8]
    return [2, 3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8]

def col_header():
    labels = ["Ticker", "Company", "Weight", "Price", "Today", "5 Days", "YTD", "52w Range"]
    if st.session_state.edit_mode:
        labels += ["Move to group", ""]
    else:
        labels += [""]
    for col, lbl in zip(st.columns(col_widths()), labels):
        col.markdown(f"<div class='wl-header'>{lbl}</div>", unsafe_allow_html=True)


def pct_html(val):
    if val is None:
        return "<span class='wl-neutral'>—</span>"
    cls   = "wl-pct-up" if val >= 0 else "wl-pct-down"
    arrow = "▲" if val >= 0 else "▼"
    return f"<span class='{cls}'>{arrow} {val:+.1f}%</span>"


def range_bar_html(price, low, high):
    if low is None or high is None or high <= low:
        return "<span class='wl-neutral'>—</span>"
    pct = max(0, min(100, (price - low) / (high - low) * 100))
    return (
        f"<div style='font-size:0.75rem;color:#888;'>${low:.0f} – ${high:.0f}</div>"
        f"<div style='background:#e8e8e8;border-radius:4px;height:6px;margin-top:2px;'>"
        f"<div style='background:#1a56db;width:{pct:.0f}%;height:6px;border-radius:4px;'></div>"
        f"</div>"
    )


# ── render each group ──────────────────────────────────────────────────────────
for group in wl_data["groups"]:
    stocks = group["stocks"]
    if not stocks:
        continue

    st.markdown(f"<div class='group-title'>{group['name']}</div>", unsafe_allow_html=True)
    col_header()

    for stock in stocks:
        sym        = stock["ticker"]
        weight     = stock.get("weight", 0.0)
        cur_group  = group["name"]
        summary    = load_summary(sym)
        cols       = st.columns(col_widths())

        with cols[0]:
            if st.button(sym, key=f"btn_{sym}", help=f"Open {sym} detail"):
                st.session_state.selected_ticker = sym
                st.rerun()

        with cols[1]:
            name = summary["name"] if summary else sym
            st.markdown(f"<div class='wl-name' title='{name}'>{name}</div>",
                        unsafe_allow_html=True)

        with cols[2]:
            st.markdown(f"<div class='wl-row wl-neutral'>{weight:.2f}%</div>",
                        unsafe_allow_html=True)

        if summary:
            price     = summary["price"]
            today_pct = summary["today_pct"]
            pct_5d    = summary["pct_5d"]
            ytd_pct   = summary["ytd_pct"]
            w52_high  = summary["week52_high"]
            w52_low   = summary["week52_low"]
            price_col = UP_COL if today_pct >= 0 else DOWN_COL

            with cols[3]:
                st.markdown(f"<div class='wl-row wl-price' style='color:{price_col}'>"
                            f"${price:.2f}</div>", unsafe_allow_html=True)
            with cols[4]:
                st.markdown(f"<div class='wl-row'>{pct_html(today_pct)}</div>",
                            unsafe_allow_html=True)
            with cols[5]:
                st.markdown(f"<div class='wl-row'>{pct_html(pct_5d)}</div>",
                            unsafe_allow_html=True)
            with cols[6]:
                st.markdown(f"<div class='wl-row'>{pct_html(ytd_pct)}</div>",
                            unsafe_allow_html=True)
            with cols[7]:
                st.markdown(f"<div class='wl-row'>{range_bar_html(price, w52_low, w52_high)}</div>",
                            unsafe_allow_html=True)
        else:
            for c in cols[3:8]:
                c.markdown("<div class='wl-row wl-neutral'>—</div>", unsafe_allow_html=True)

        if st.session_state.edit_mode:
            with cols[8]:
                new_group = st.selectbox(
                    "", group_names,
                    index=group_names.index(cur_group) if cur_group in group_names else 0,
                    key=f"grp_{sym}",
                    label_visibility="collapsed")
                if new_group != cur_group:
                    move_stock(wl_data, sym, new_group)
                    tg_data["tickers"][sym] = new_group
                    save_ticker_groups(tg_data)
                    st.rerun()
            with cols[9]:
                if st.button("🗑️", key=f"del_{sym}", help=f"Remove {sym}"):
                    remove_stock(wl_data, sym)
                    st.cache_data.clear()
                    st.rerun()
        else:
            cols[8].markdown("", unsafe_allow_html=True)

st.markdown("---")
st.caption("Data via Yahoo Finance · Prices delayed ~15 min · Not financial advice")
