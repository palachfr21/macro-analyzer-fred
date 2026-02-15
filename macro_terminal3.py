"""
╔══════════════════════════════════════════════════════════════════╗
║  MACRO CORRELATION TERMINAL                                      ║
║  Production-Grade Macroeconomic Analysis Dashboard               ║
║  Powered by FRED API · Streamlit · Plotly · Statsmodels          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import BytesIO

# ── Optional imports with graceful fallback ──────────────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG & DARK THEME
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Macro Correlation Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2035;
    --bg-card-hover: #1f2847;
    --border: #2a3553;
    --text-primary: #e2e8f0;
    --text-secondary: #8899b4;
    --accent-cyan: #06d6a0;
    --accent-blue: #4361ee;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #8b5cf6;
    --glow-cyan: rgba(6, 214, 160, 0.15);
    --glow-blue: rgba(67, 97, 238, 0.15);
}

/* ── Global styling ── */
.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: var(--accent-cyan) !important;
    font-size: 1.6rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stDateInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stSlider > div > div > div {
    color: var(--accent-cyan) !important;
}

/* ── Buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px var(--glow-blue) !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border) !important;
    opacity: 0.4 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-secondary) !important;
}

/* ── Custom metric strip ── */
.metric-strip {
    display: flex;
    gap: 16px;
    margin: 20px 0;
}

.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    transition: all 0.25s ease;
}

.metric-card:hover {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 24px var(--glow-cyan);
    transform: translateY(-2px);
}

.metric-card .label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent-cyan);
    line-height: 1;
}

.metric-card .sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-top: 6px;
}

/* ── Header bar ── */
.terminal-header {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.terminal-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
}

.terminal-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}

.terminal-header .subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

/* ── Signal badge ── */
.signal-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.signal-positive { background: rgba(6, 214, 160, 0.15); color: #06d6a0; border: 1px solid rgba(6, 214, 160, 0.3); }
.signal-negative { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
.signal-neutral  { background: rgba(136, 153, 180, 0.15); color: #8899b4; border: 1px solid rgba(136, 153, 180, 0.3); }

/* ── Info boxes ── */
.info-box {
    background: var(--bg-card);
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--border) !important;
}

/* ── Tooltip override ── */
div[data-testid="stTooltipIcon"] svg { color: var(--text-secondary); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* ── Wider selectbox dropdowns ── */

/* ── CLOSED STATE: the selected value shown in the collapsed selectbox ── */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    min-width: 100% !important;
}

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    min-height: 48px !important;
    height: auto !important;
    line-height: 1.45 !important;
    padding: 8px 32px 8px 12px !important;
    font-size: 0.84rem !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0 !important;
    display: flex !important;
    align-items: center !important;
    word-break: break-word !important;
}

/* Make the inner span also wrap */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    word-break: break-word !important;
    line-height: 1.45 !important;
}

/* ── OPEN STATE: dropdown list items ── */
div[data-baseweb="popover"] ul li {
    white-space: normal !important;
    word-break: break-word !important;
    padding: 10px 14px !important;
    font-size: 0.83rem !important;
    line-height: 1.45 !important;
    border-bottom: 1px solid rgba(42, 53, 83, 0.4) !important;
}

div[data-baseweb="popover"] ul {
    max-height: 400px !important;
}

/* ── Smart transform recommendation block (vertical layout) ── */
.rec-block {
    background: rgba(67, 97, 238, 0.08);
    border: 1px solid rgba(67, 97, 238, 0.2);
    border-left: 3px solid rgba(67, 97, 238, 0.6);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0 10px 0;
}

.rec-block .rec-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.78rem;
    color: var(--accent-blue);
    letter-spacing: 0.02em;
    margin-bottom: 5px;
}

.rec-block .rec-explain {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    line-height: 1.5;
    color: var(--text-secondary);
    opacity: 0.85;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# 1. CONSTANTS & REFERENCE LIBRARY
# ════════════════════════════════════════════════════════════════════

# ── Series Library ──────────────────────────────────────────────
# Key   = friendly display name
# Value = dict with fred_id, category, and recommended transform
#
# TRANSFORM RECOMMENDATION CODES:
#   "level" → Raw Level   (rates, spreads, indices already in %)
#   "pct"   → Log Returns (price levels, asset prices)
#   "diff"  → Δ Absolute  (interest rates → basis-point changes)
#   "yoy"   → YoY %       (macro levels with seasonal patterns)

SERIES_LIBRARY = {
    # ── 📈 Rates & Yields ──────────────────────────────────────
    "Yield Curve 10Y-2Y Spread": {
        "fred_id": "T10Y2Y", "category": "📈 Rates & Yields",
        "rec_transform": "level",
        "rec_reason": "Already a spread in percentage points — use Level to preserve sign and zero-crossing signals.",
    },
    "Federal Funds Rate": {
        "fred_id": "FEDFUNDS", "category": "📈 Rates & Yields",
        "rec_transform": "diff",
        "rec_reason": "Interest rate in %. Use Δ (Diff) to capture basis-point moves between meetings.",
    },
    "10-Year Treasury Yield": {
        "fred_id": "DGS10", "category": "📈 Rates & Yields",
        "rec_transform": "diff",
        "rec_reason": "Interest rate level — Δ Diff converts to basis-point changes for stationarity.",
    },
    "2-Year Treasury Yield": {
        "fred_id": "DGS2", "category": "📈 Rates & Yields",
        "rec_transform": "diff",
        "rec_reason": "Short-term rate — Δ Diff gives rate-of-change in bps, better for correlation analysis.",
    },
    "10-Year Real Interest Rate": {
        "fred_id": "REAINTRATREARAT10Y", "category": "📈 Rates & Yields",
        "rec_transform": "level",
        "rec_reason": "Real rate can be negative; Level preserves sign. Use Diff for change analysis.",
    },
    "High Yield OAS Spread": {
        "fred_id": "BAMLH0A0HYM2", "category": "📈 Rates & Yields",
        "rec_transform": "level",
        "rec_reason": "Credit spread in bps — Level shows risk appetite; spikes = stress events.",
    },

    # ── 💹 Inflation & Prices ──────────────────────────────────
    "CPI – All Urban Consumers": {
        "fred_id": "CPIAUCSL", "category": "💹 Inflation & Prices",
        "rec_transform": "yoy",
        "rec_reason": "Price index level — YoY % gives the headline inflation rate and removes seasonality.",
    },
    "Core CPI (ex Food & Energy)": {
        "fred_id": "CPILFESL", "category": "💹 Inflation & Prices",
        "rec_transform": "yoy",
        "rec_reason": "Core price index — YoY % is the standard way to read underlying inflation trends.",
    },
    "Personal Consumption Expenditures": {
        "fred_id": "PCE", "category": "💹 Inflation & Prices",
        "rec_transform": "yoy",
        "rec_reason": "Nominal spending level — YoY % shows consumer spending growth and seasonality.",
    },
    "PCE Price Index": {
        "fred_id": "PCEPI", "category": "💹 Inflation & Prices",
        "rec_transform": "yoy",
        "rec_reason": "The Fed's preferred inflation gauge — YoY % is the standard policy metric.",
    },
    "10-Year Breakeven Inflation": {
        "fred_id": "T10YIE", "category": "💹 Inflation & Prices",
        "rec_transform": "level",
        "rec_reason": "Already in % (market-implied inflation). Level shows expectations; Diff for changes.",
    },
    "Gold Price (USD/Troy Oz)": {
        "fred_id": "GOLDPMGBD228NLBM", "category": "💹 Inflation & Prices",
        "rec_transform": "pct",
        "rec_reason": "Asset price level — Log Returns (% Change) removes trend for stationary analysis.",
    },

    # ── 🏭 Real Economy ────────────────────────────────────────
    "Unemployment Rate": {
        "fred_id": "UNRATE", "category": "🏭 Real Economy",
        "rec_transform": "level",
        "rec_reason": "Already in % — Level is standard for labor market analysis. Diff for momentum.",
    },
    "Nonfarm Payrolls": {
        "fred_id": "PAYEMS", "category": "🏭 Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Employment count in thousands — YoY % shows job growth pace and smooths seasonality.",
    },
    "Real GDP": {
        "fred_id": "GDPC1", "category": "🏭 Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Quarterly GDP level — YoY % gives the standard economic growth rate.",
    },
    "Industrial Production Index": {
        "fred_id": "INDPRO", "category": "🏭 Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Index level with trends — YoY % reveals the cyclical manufacturing signal.",
    },
    "Retail Sales": {
        "fred_id": "RSAFS", "category": "🏭 Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Nominal sales in $M — YoY % strips out seasonality and inflation drift.",
    },
    "Housing Starts": {
        "fred_id": "HOUST", "category": "🏭 Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Thousands of units — YoY % reveals the housing cycle more clearly than levels.",
    },

    # ── 📊 Sentiment & Markets ─────────────────────────────────
    "CBOE VIX Index": {
        "fred_id": "VIXCLS", "category": "📊 Sentiment & Markets",
        "rec_transform": "level",
        "rec_reason": "Implied volatility in % — Level is standard; mean-reverting by nature.",
    },
    "S&P 500 Index": {
        "fred_id": "SP500", "category": "📊 Sentiment & Markets",
        "rec_transform": "pct",
        "rec_reason": "Price index — Log Returns (% Change) is essential for removing the upward trend.",
    },
    "M2 Money Supply": {
        "fred_id": "WM2NS", "category": "📊 Sentiment & Markets",
        "rec_transform": "yoy",
        "rec_reason": "Monetary aggregate in $B — YoY % shows liquidity growth. Previously M2SL, now WM2NS on FRED.",
    },
    "Consumer Sentiment (UMich)": {
        "fred_id": "UMCSENT", "category": "📊 Sentiment & Markets",
        "rec_transform": "level",
        "rec_reason": "Survey index (0–100+) — Level is standard for sentiment; already bounded and comparable.",
    },
    "Trade-Weighted USD Index": {
        "fred_id": "DTWEXBGS", "category": "📊 Sentiment & Markets",
        "rec_transform": "pct",
        "rec_reason": "Currency index level — Log Returns shows dollar strength/weakness momentum.",
    },
    "Initial Jobless Claims": {
        "fred_id": "ICSA", "category": "📊 Sentiment & Markets",
        "rec_transform": "level",
        "rec_reason": "Weekly claims count — Level is the standard read. Already high-frequency and mean-reverting.",
    },
}

# ── Transform options ──────────────────────────────────────────
TRANSFORM_OPTIONS = {
    "Raw Level": "level",
    "% Change (Log Returns)": "pct",
    "Absolute Difference (Δ)": "diff",
    "Year-over-Year % Change": "yoy",
}

# Reverse lookup: transform code → display key
TRANSFORM_CODE_TO_KEY = {v: k for k, v in TRANSFORM_OPTIONS.items()}

FREQ_OPTIONS = {
    "Daily":      "D",
    "Weekly":     "W",
    "Monthly":    "MS",
    "Quarterly":  "QS",
}

# Color palette
C_CYAN   = "#06d6a0"
C_BLUE   = "#4361ee"
C_AMBER  = "#f59e0b"
C_RED    = "#ef4444"
C_PURPLE = "#8b5cf6"
C_GRID   = "#1e293b"
C_BG     = "#0a0e17"
C_CARD   = "#1a2035"

# ════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def build_series_list():
    """Return the ordered list of friendly names for the selectbox."""
    return list(SERIES_LIBRARY.keys())


def get_fred_id(friendly_name: str) -> str:
    """Look up FRED ID from a friendly name."""
    return SERIES_LIBRARY[friendly_name]["fred_id"]


def get_recommended_transform(friendly_name: str) -> dict:
    """Return the recommended transform code and reason for a series."""
    meta = SERIES_LIBRARY.get(friendly_name, {})
    return {
        "code": meta.get("rec_transform", "level"),
        "reason": meta.get("rec_reason", ""),
    }


def format_series_option(name: str) -> str:
    """Format a series for display: 'Friendly Name  · FRED_ID'."""
    meta = SERIES_LIBRARY.get(name, {})
    fid = meta.get("fred_id", "")
    return f"{name}  · {fid}"


def apply_transform(series: pd.Series, mode: str) -> pd.Series:
    """Apply the chosen stationarity transformation."""
    if mode == "pct":
        return np.log(series / series.shift(1)).dropna() * 100  # log returns in %
    elif mode == "diff":
        return series.diff().dropna()
    elif mode == "yoy":
        # YoY requires guessing freq; try 12 periods for monthly, 4 for quarterly
        freq_guess = pd.infer_freq(series.index)
        if freq_guess and "Q" in freq_guess:
            return series.pct_change(periods=4).dropna() * 100
        else:
            return series.pct_change(periods=12).dropna() * 100
    return series  # "level"


def resample_to_lower_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Resample both columns to the lowest common frequency."""
    freqs = {"D": 1, "W": 7, "MS": 30, "QS": 90, "AS": 365}

    def estimate_freq(s):
        diffs = s.dropna().index.to_series().diff().dt.days.median()
        if diffs is None or np.isnan(diffs):
            return "MS"
        if diffs < 3:
            return "D"
        elif diffs < 14:
            return "W"
        elif diffs < 75:
            return "MS"
        else:
            return "QS"

    f1 = estimate_freq(df.iloc[:, 0])
    f2 = estimate_freq(df.iloc[:, 1])
    target = f1 if freqs.get(f1, 30) >= freqs.get(f2, 30) else f2

    return df.resample(target).last().dropna()


def compute_correlation_stats(x: pd.Series, y: pd.Series):
    """Compute Pearson r, R², p-value, OLS slope & intercept."""
    mask = x.notna() & y.notna()
    x_clean, y_clean = x[mask].values, y[mask].values
    n = len(x_clean)

    if n < 3:
        return {"r": np.nan, "r2": np.nan, "pvalue": np.nan,
                "slope": np.nan, "intercept": np.nan, "n": n}

    if STATSMODELS_AVAILABLE:
        X = sm.add_constant(x_clean)
        model = sm.OLS(y_clean, X).fit()
        return {
            "r": np.corrcoef(x_clean, y_clean)[0, 1],
            "r2": model.rsquared,
            "pvalue": model.pvalues[1] if len(model.pvalues) > 1 else np.nan,
            "slope": model.params[1] if len(model.params) > 1 else np.nan,
            "intercept": model.params[0],
            "n": n,
        }
    else:
        r = np.corrcoef(x_clean, y_clean)[0, 1]
        slope = np.polyfit(x_clean, y_clean, 1)
        from scipy import stats as sp_stats
        _, pval = sp_stats.pearsonr(x_clean, y_clean)
        return {
            "r": r, "r2": r**2, "pvalue": pval,
            "slope": slope[0], "intercept": slope[1], "n": n,
        }


def significance_label(p: float) -> str:
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "★★★  p < 0.001"
    elif p < 0.01:
        return "★★  p < 0.01"
    elif p < 0.05:
        return "★  p < 0.05"
    else:
        return "Not significant"


def correlation_signal(r: float) -> tuple:
    """Returns (label, css_class)."""
    if np.isnan(r):
        return ("—", "signal-neutral")
    ar = abs(r)
    if ar >= 0.7:
        sign = "STRONG +" if r > 0 else "STRONG −"
        cls = "signal-positive" if r > 0 else "signal-negative"
    elif ar >= 0.4:
        sign = "MODERATE +" if r > 0 else "MODERATE −"
        cls = "signal-positive" if r > 0 else "signal-negative"
    else:
        sign = "WEAK"
        cls = "signal-neutral"
    return (sign, cls)


# ════════════════════════════════════════════════════════════════════
# 3. DATA FETCHING (CACHED)
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(api_key: str, series_id: str,
                      start: str, end: str) -> pd.Series:
    """Fetch a single FRED series with caching."""
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id,
                           observation_start=start,
                           observation_end=end)
    data.name = series_id
    data.index = pd.DatetimeIndex(data.index)
    return data.dropna()


# ════════════════════════════════════════════════════════════════════
# 4. PLOTLY CHART BUILDERS
# ════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=C_BG,
    plot_bgcolor=C_BG,
    font=dict(family="DM Sans, sans-serif", color="#c9d1d9", size=12),
    margin=dict(l=60, r=60, t=60, b=50),
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
)


def build_dual_axis_chart(df, col_a, col_b, label_a, label_b, lag):
    """Time-series chart with dual Y axes."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    lag_suffix = f" (lag {lag:+d}m)" if lag != 0 else ""

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[col_a],
            name=f"{label_a}{lag_suffix}",
            line=dict(color=C_CYAN, width=2.2),
            hovertemplate="%{y:.4f}",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[col_b],
            name=label_b,
            line=dict(color=C_AMBER, width=2.2),
            hovertemplate="%{y:.4f}",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text="<b>Time Series Comparison</b>",
            font=dict(family="JetBrains Mono, monospace", size=15, color="#e2e8f0"),
            x=0.01,
        ),
        height=440,
    )

    fig.update_xaxes(
        gridcolor=C_GRID, zeroline=False,
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        title_text=label_a, secondary_y=False,
        gridcolor=C_GRID, zeroline=False,
        title_font=dict(color=C_CYAN, size=12),
        tickfont=dict(color=C_CYAN, size=10),
    )
    fig.update_yaxes(
        title_text=label_b, secondary_y=True,
        gridcolor="rgba(30,41,59,0.4)", zeroline=False,
        title_font=dict(color=C_AMBER, size=12),
        tickfont=dict(color=C_AMBER, size=10),
    )
    return fig


def build_scatter_chart(x, y, label_x, label_y, stats):
    """Scatter plot with OLS regression line."""
    mask = x.notna() & y.notna()
    xc, yc = x[mask], y[mask]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xc, y=yc, mode="markers",
        marker=dict(
            color=C_CYAN, size=5, opacity=0.55,
            line=dict(width=0.5, color="rgba(6,214,160,0.3)"),
        ),
        name="Observations",
        hovertemplate=f"{label_x}: %{{x:.3f}}<br>{label_y}: %{{y:.3f}}",
    ))

    # Regression line
    if not np.isnan(stats["slope"]):
        x_range = np.linspace(xc.min(), xc.max(), 100)
        y_hat = stats["intercept"] + stats["slope"] * x_range

        eq_text = (
            f"y = {stats['slope']:.4f}·x "
            f"{'+ ' if stats['intercept'] >= 0 else '− '}"
            f"{abs(stats['intercept']):.4f}"
        )

        fig.add_trace(go.Scatter(
            x=x_range, y=y_hat, mode="lines",
            line=dict(color=C_AMBER, width=2, dash="dot"),
            name=f"OLS: {eq_text}",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text="<b>Correlation Scatter + OLS Regression</b>",
            font=dict(family="JetBrains Mono, monospace", size=15, color="#e2e8f0"),
            x=0.01,
        ),
        xaxis=dict(title=label_x, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(title=label_y, gridcolor=C_GRID, zeroline=False),
        height=440,
    )
    return fig


def build_rolling_corr_chart(df, col_a, col_b, windows=(30, 90, 252)):
    """Rolling correlation chart for multiple windows."""
    fig = go.Figure()
    colors = [C_CYAN, C_BLUE, C_PURPLE]

    for w, c in zip(windows, colors):
        if len(df) > w:
            rolling = df[col_a].rolling(w).corr(df[col_b])
            fig.add_trace(go.Scatter(
                x=df.index, y=rolling,
                name=f"{w}-period",
                line=dict(color=c, width=1.8),
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="#4a5568", line_width=0.8)
    fig.add_hline(y=0.7, line_dash="dot", line_color="rgba(6,214,160,0.3)", line_width=0.6)
    fig.add_hline(y=-0.7, line_dash="dot", line_color="rgba(239,68,68,0.3)", line_width=0.6)

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text="<b>Rolling Correlation</b>",
            font=dict(family="JetBrains Mono, monospace", size=15, color="#e2e8f0"),
            x=0.01,
        ),
        yaxis=dict(title="Pearson r", gridcolor=C_GRID, range=[-1.05, 1.05]),
        xaxis=dict(gridcolor=C_GRID),
        height=360,
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# 5. SIDEBAR CONFIGURATION
# ════════════════════════════════════════════════════════════════════

series_names = build_series_list()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:20px;">
        <span style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700;
                     color:#06d6a0;">⟨ MACRO TERMINAL ⟩</span><br>
        <span style="font-size:0.7rem; color:#8899b4;">Correlation Engine v2.1</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── API Key ──
    api_key = st.text_input(
        "🔑 FRED API Key",
        type="password",
        help="Get your free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        placeholder="Enter your FRED API key...",
    )

    if api_key:
        st.session_state["fred_api_key"] = api_key

    st.markdown("---")

    # ────────────────────────────────────────────────────────────
    # VARIABLE A
    # ────────────────────────────────────────────────────────────
    st.markdown("##### Variable A")

    sel_a = st.selectbox(
        "Select Series A",
        series_names,
        index=0,
        key="sel_a",
        format_func=format_series_option,
        help="Choose from the curated library or type a custom FRED ID below.",
    )
    custom_a = st.text_input("…or enter custom FRED ID", key="custom_a",
                             placeholder="e.g. DCOILWTICO")

    is_custom_a = bool(custom_a.strip())
    series_id_a = custom_a.strip().upper() if is_custom_a else get_fred_id(sel_a)

    # Smart transform recommendation for A
    if not is_custom_a:
        rec_a = get_recommended_transform(sel_a)
        rec_key_a = TRANSFORM_CODE_TO_KEY.get(rec_a["code"], "Raw Level")
        rec_idx_a = list(TRANSFORM_OPTIONS.keys()).index(rec_key_a)
        st.markdown(
            f'<div class="rec-block">'
            f'<div class="rec-title">⚡ REC: {rec_key_a}</div>'
            f'<div class="rec-explain">{rec_a["reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        rec_idx_a = 0  # default to Raw Level for custom IDs

    transform_a = st.selectbox(
        "Transform A",
        list(TRANSFORM_OPTIONS.keys()),
        index=rec_idx_a,
        key="tf_a",
        help="**Raw Level**: Use for rates (FEDFUNDS, UNRATE) that are already stationary-like.  \n"
             "**% Change (Log Returns)**: Use for prices & indices (SP500, Gold) "
             "to remove trends and compare returns.  \n"
             "**Absolute Δ**: Use for interest rates to capture basis-point moves.  \n"
             "**YoY %**: Use for economic series with seasonal patterns (CPI, PAYEMS).  \n\n"
             "💡 The ⚡ REC badge above shows the recommended transform for this series.",
    )

    st.markdown("---")

    # ────────────────────────────────────────────────────────────
    # VARIABLE B
    # ────────────────────────────────────────────────────────────
    st.markdown("##### Variable B")

    sel_b = st.selectbox(
        "Select Series B",
        series_names,
        index=5,
        key="sel_b",
        format_func=format_series_option,
        help="Choose from the curated library or type a custom FRED ID below.",
    )
    custom_b = st.text_input("…or enter custom FRED ID", key="custom_b",
                             placeholder="e.g. DEXUSEU")

    is_custom_b = bool(custom_b.strip())
    series_id_b = custom_b.strip().upper() if is_custom_b else get_fred_id(sel_b)

    # Smart transform recommendation for B
    if not is_custom_b:
        rec_b = get_recommended_transform(sel_b)
        rec_key_b = TRANSFORM_CODE_TO_KEY.get(rec_b["code"], "Raw Level")
        rec_idx_b = list(TRANSFORM_OPTIONS.keys()).index(rec_key_b)
        st.markdown(
            f'<div class="rec-block">'
            f'<div class="rec-title">⚡ REC: {rec_key_b}</div>'
            f'<div class="rec-explain">{rec_b["reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        rec_idx_b = 0

    transform_b = st.selectbox(
        "Transform B",
        list(TRANSFORM_OPTIONS.keys()),
        index=rec_idx_b,
        key="tf_b",
        help="Choose the transformation for Series B. "
             "The ⚡ REC badge above shows the recommended transform.  \n\n"
             "**Why does it matter?** Correlating a trending price index (SP500) "
             "with a bounded rate (FEDFUNDS) in raw levels produces spurious correlation. "
             "Proper transforms make the comparison statistically meaningful.",
    )

    st.markdown("---")
    st.markdown("##### Parameters")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date",
                                   value=datetime(2000, 1, 1),
                                   min_value=datetime(1950, 1, 1))
    with col_d2:
        end_date = st.date_input("End Date",
                                 value=datetime.today())

    lag = st.slider(
        "⏱ Lag (months)",
        min_value=-24, max_value=24, value=0, step=1,
        help="Shift Variable A forward (+) or backward (−) in time relative to Variable B. "
             "A lag of +6 means A leads B by 6 months, useful for testing predictive power.",
    )

    auto_resample = st.checkbox("Auto-resample to lowest frequency", value=True,
                                help="If the two series have different frequencies, "
                                     "auto-align to the lower one.")

    st.markdown("---")


# ════════════════════════════════════════════════════════════════════
# 6. MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div class="terminal-header">
    <h1>📊 Macro Correlation Terminal</h1>
    <div class="subtitle">
        FRED-powered macroeconomic variable analysis · Correlation · Lead/Lag · OLS Regression
    </div>
</div>
""", unsafe_allow_html=True)


# ── Guard: need API key ──
if not FRED_AVAILABLE:
    st.error("⚠️ `fredapi` is not installed. Run `pip install fredapi` to enable FRED data access.")
    st.stop()

if not api_key:
    st.markdown("""
    <div class="info-box">
        <strong>Welcome.</strong> Enter your FRED API key in the sidebar to begin.
        Free keys are available at
        <a href="https://fred.stlouisfed.org/docs/api/api_key.html"
           style="color:#4361ee;" target="_blank">fred.stlouisfed.org</a>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="border-left-color: #8b5cf6;">
        <strong>Quick start ideas:</strong><br>
        • <code>SP500</code> (% Change) vs <code>VIXCLS</code> (Level) — classic fear/greed inverse.<br>
        • <code>T10Y2Y</code> (Level) vs <code>UNRATE</code> (Level) at lag +12 — yield curve predicting recession.<br>
        • <code>WM2NS</code> (YoY %) vs <code>CPIAUCSL</code> (YoY %) at lag +18 — money supply leading inflation.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Fetch data ──
with st.spinner(f"Fetching `{series_id_a}` from FRED…"):
    try:
        raw_a = fetch_fred_series(api_key, series_id_a,
                                  start_date.strftime("%Y-%m-%d"),
                                  end_date.strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"❌ Could not fetch **{series_id_a}**: {e}")
        st.stop()

with st.spinner(f"Fetching `{series_id_b}` from FRED…"):
    try:
        raw_b = fetch_fred_series(api_key, series_id_b,
                                  start_date.strftime("%Y-%m-%d"),
                                  end_date.strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"❌ Could not fetch **{series_id_b}**: {e}")
        st.stop()


# ── Transform ──
mode_a = TRANSFORM_OPTIONS[transform_a]
mode_b = TRANSFORM_OPTIONS[transform_b]

trans_a = apply_transform(raw_a, mode_a)
trans_b = apply_transform(raw_b, mode_b)

label_a = f"{series_id_a} ({mode_a})"
label_b = f"{series_id_b} ({mode_b})"

# ── Apply lag ──
if lag != 0:
    trans_a = trans_a.shift(lag)

# ── Merge & resample ──
df = pd.DataFrame({label_a: trans_a, label_b: trans_b})

if auto_resample:
    df = resample_to_lower_freq(df)

df = df.dropna()

if len(df) < 3:
    st.warning("⚠️ Not enough overlapping data points after transformations. "
               "Try adjusting the date range, transforms, or lag.")
    st.stop()


# ── Statistics ──
stats = compute_correlation_stats(df[label_a], df[label_b])
sig_label = significance_label(stats["pvalue"])
signal_text, signal_class = correlation_signal(stats["r"])

# ── Metric Strip (custom HTML) ──
r_display  = f"{stats['r']:.4f}" if not np.isnan(stats['r']) else "—"
r2_display = f"{stats['r2']:.4f}" if not np.isnan(stats['r2']) else "—"
p_display  = f"{stats['pvalue']:.2e}" if not np.isnan(stats['pvalue']) else "—"
n_display  = f"{stats['n']:,}"

st.markdown(f"""
<div class="metric-strip">
    <div class="metric-card">
        <div class="label">Pearson r</div>
        <div class="value">{r_display}</div>
        <div class="sub"><span class="{signal_class} signal-badge">{signal_text}</span></div>
    </div>
    <div class="metric-card">
        <div class="label">R² (Determination)</div>
        <div class="value">{r2_display}</div>
        <div class="sub">{float(stats['r2'])*100:.1f}% variance explained</div>
    </div>
    <div class="metric-card">
        <div class="label">P-Value</div>
        <div class="value">{p_display}</div>
        <div class="sub">{sig_label}</div>
    </div>
    <div class="metric-card">
        <div class="label">Observations</div>
        <div class="value">{n_display}</div>
        <div class="sub">Lag: {lag:+d} months</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Charts ──
tab_ts, tab_scatter, tab_rolling, tab_data = st.tabs([
    "📈 Time Series", "🔘 Scatter + OLS", "🔄 Rolling Correlation", "📋 Data Table"
])

with tab_ts:
    fig_ts = build_dual_axis_chart(df, label_a, label_b,
                                   label_a, label_b, lag)
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

with tab_scatter:
    fig_sc = build_scatter_chart(df[label_a], df[label_b],
                                 label_a, label_b, stats)
    st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    # ── Regression equation box ──
    if not np.isnan(stats["slope"]):
        st.markdown(f"""
        <div class="info-box">
            <strong>OLS Regression:</strong>
            <code style="color:#f59e0b;">
            {label_b} = {stats['slope']:.6f} × {label_a}
            {'+ ' if stats['intercept'] >= 0 else '− '}{abs(stats['intercept']):.6f}
            </code>
            <br><span style="font-size:0.75rem;">
            R² = {stats['r2']:.4f} · p = {stats['pvalue']:.2e} · n = {stats['n']}
            </span>
        </div>
        """, unsafe_allow_html=True)

with tab_rolling:
    fig_rc = build_rolling_corr_chart(df, label_a, label_b)
    st.plotly_chart(fig_rc, use_container_width=True, config={"displayModeBar": False})

with tab_data:
    st.dataframe(
        df.style.format("{:.4f}").set_properties(
            **{"color": "#c9d1d9", "background-color": "#111827", "font-size": "0.85rem"}
        ),
        use_container_width=True,
        height=400,
    )

    # ── CSV Download ──
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=True)
    csv_buffer.seek(0)

    st.download_button(
        label="⬇ Download Merged CSV",
        data=csv_buffer,
        file_name=f"macro_corr_{series_id_a}_{series_id_b}_lag{lag}.csv",
        mime="text/csv",
    )


# ── Lag Sensitivity Analysis ──
with st.expander("🔬 Lag Sensitivity Scan (−24 to +24 months)", expanded=False):
    st.markdown("""
    <div class="info-box" style="border-left-color:#8b5cf6;">
        Scans all lag values to find the optimal lead/lag that maximises |r|.
        This is useful for identifying if one variable leads or lags the other.
    </div>
    """, unsafe_allow_html=True)

    lag_results = []
    trans_a_base = apply_transform(raw_a, mode_a)

    for test_lag in range(-24, 25):
        shifted = trans_a_base.shift(test_lag)
        temp_df = pd.DataFrame({label_a: shifted, label_b: trans_b}).dropna()
        if len(temp_df) >= 10:
            r_val = np.corrcoef(temp_df[label_a].values,
                                temp_df[label_b].values)[0, 1]
            lag_results.append({"lag": test_lag, "r": r_val, "abs_r": abs(r_val),
                                "n": len(temp_df)})

    if lag_results:
        lag_df = pd.DataFrame(lag_results)
        best = lag_df.loc[lag_df["abs_r"].idxmax()]

        colors_lag = [C_CYAN if abs(r) >= 0.4 else C_GRID for r in lag_df["r"]]
        colors_lag = [C_AMBER if lag_df.iloc[i]["lag"] == best["lag"] else c
                      for i, c in enumerate(colors_lag)]

        fig_lag = go.Figure(go.Bar(
            x=lag_df["lag"], y=lag_df["r"],
            marker_color=colors_lag,
            hovertemplate="Lag %{x}m: r = %{y:.4f}<extra></extra>",
        ))
        fig_lag.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            title=dict(
                text=f"<b>Optimal Lag: {int(best['lag']):+d} months (r = {best['r']:.4f})</b>",
                font=dict(family="JetBrains Mono, monospace", size=14, color="#e2e8f0"),
                x=0.01,
            ),
            xaxis=dict(title="Lag (months)", gridcolor=C_GRID, dtick=3),
            yaxis=dict(title="Pearson r", gridcolor=C_GRID, range=[-1.05, 1.05]),
            height=340,
        )
        fig_lag.add_hline(y=0, line_dash="dash", line_color="#4a5568", line_width=0.8)
        st.plotly_chart(fig_lag, use_container_width=True, config={"displayModeBar": False})


# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:16px 0;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#4a5568;">
        MACRO CORRELATION TERMINAL · Built with Streamlit + FRED API + Plotly + Statsmodels<br>
        Data: Federal Reserve Bank of St. Louis · Not investment advice
    </span>
</div>
""", unsafe_allow_html=True)
