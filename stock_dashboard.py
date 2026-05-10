import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from stock_detail import apply_styles, render

st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")
apply_styles()

# ── top controls ───────────────────────────────────────────────────────────────
_tc1, _tc2, _tc3 = st.columns([5, 1, 1])
with _tc1:
    ticker = st.text_input("", value="AAOI",
                           placeholder="🔍  Enter ticker symbol  (e.g. AAOI · BE · TSLA · NVDA)",
                           label_visibility="collapsed").upper().strip()
with _tc3:
    with st.expander("⚙️ Settings"):
        thr = st.slider("'Big move' threshold (%)", 3, 15, 5)
        st.caption("Data via Yahoo Finance · last 12 months")

render(ticker, thr)
