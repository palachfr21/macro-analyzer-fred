"""
MACRO CORRELATION TERMINAL
Production-grade macroeconomic correlation analysis.
Powered by FRED API, Streamlit, Plotly, Statsmodels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from io import BytesIO

# ── Optional imports with graceful fallback ──────────────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG & STYLING
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Macro Correlation Terminal",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sober academic dark theme ──
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #131922;
    --bg-card: #161c28;
    --border: #232a3a;
    --border-soft: #1c2330;
    --text-primary: #d8dee9;
    --text-secondary: #7d8597;
    --text-muted: #5a6275;
    --accent: #4c8eda;
    --accent-soft: #06d6a0;
    --accent-warn: #d97757;
}

.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

h1, h2, h3, h4, h5, h6 {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em;
    color: var(--text-primary) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stDateInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stSlider [data-baseweb="slider"] > div > div {
    background: var(--accent) !important;
}

/* ── Buttons ── */
.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    transition: border-color 0.15s ease !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
}

hr { border-color: var(--border) !important; opacity: 0.6 !important; }

.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
}

/* ── Header ── */
.terminal-header {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 22px 28px;
    margin-bottom: 22px;
}
.terminal-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.3rem !important;
    font-weight: 500 !important;
    color: var(--text-primary);
    margin: 0 0 4px 0;
}
.terminal-header .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}

/* ── Metric strip ── */
.metric-strip {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
}
.metric-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.1;
}
.metric-card .sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 6px;
}

/* ── ADF stationarity row (sober) ── */
.adf-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.adf-cell {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 18px;
}
.adf-cell .adf-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.adf-cell .adf-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 4px;
}
.adf-cell .adf-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
}
.adf-ok   { color: var(--accent-soft); }
.adf-fail { color: var(--accent-warn); }
.adf-na   { color: var(--text-muted); }

/* ── Granger causality block (matches ADF styling) ── */
.granger-row {
    display: flex;
    gap: 12px;
    margin: 12px 0 8px 0;
}
.granger-cell {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 18px;
}
.granger-cell .granger-direction {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.granger-cell .granger-verdict {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 4px;
}
.granger-cell .granger-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
}
.granger-causes    { color: var(--accent-soft); }
.granger-nocause   { color: var(--text-muted); }
.granger-warn      { color: var(--accent-warn); }

/* ── Section labels ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 20px 0 8px 0;
}

/* ── Recommendation block (sober) ── */
.rec-block {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    border-radius: 2px;
    padding: 10px 14px;
    margin: 8px 0 12px 0;
}
.rec-block .rec-title {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    font-size: 0.74rem;
    color: var(--accent);
    letter-spacing: 0.02em;
    margin-bottom: 4px;
}
.rec-block .rec-explain {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    line-height: 1.5;
    color: var(--text-secondary);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--bg-secondary);
    border-radius: 4px;
    padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* ── Info note (sober, no colored panel) ── */
.note {
    background: transparent;
    border-left: 2px solid var(--border);
    padding: 8px 14px;
    margin: 12px 0;
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.5;
    font-family: 'Inter', sans-serif;
}

/* ── Selectbox readability (closed state) ── */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    min-height: 44px !important;
    height: auto !important;
    line-height: 1.4 !important;
    padding: 7px 30px 7px 11px !important;
    font-size: 0.82rem !important;
    word-break: break-word !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    word-break: break-word !important;
    line-height: 1.4 !important;
}
div[data-baseweb="popover"] ul li {
    white-space: normal !important;
    word-break: break-word !important;
    padding: 9px 13px !important;
    font-size: 0.8rem !important;
    line-height: 1.4 !important;
    border-bottom: 1px solid var(--border-soft) !important;
}
div[data-baseweb="popover"] ul { max-height: 400px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Tooltip icon ── */
div[data-testid="stTooltipIcon"] svg { color: var(--text-muted); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 1. REFERENCE LIBRARY
# ════════════════════════════════════════════════════════════════════

# Each series carries its FRED ID, category, and a recommended transform
# with an explanatory note (used by the smart-scaling engine).
SERIES_LIBRARY = {
    # Rates & Yields
    "Yield Curve 10Y-2Y Spread": {
        "fred_id": "T10Y2Y", "category": "Rates & Yields",
        "rec_transform": "level",
        "rec_reason": "Already a spread in percentage points. Level preserves sign and zero-crossing signals.",
    },
    "Federal Funds Rate": {
        "fred_id": "FEDFUNDS", "category": "Rates & Yields",
        "rec_transform": "diff",
        "rec_reason": "Policy rate in %. Difference captures basis-point moves between meetings.",
    },
    "10-Year Treasury Yield": {
        "fred_id": "DGS10", "category": "Rates & Yields",
        "rec_transform": "diff",
        "rec_reason": "Interest rate level. Difference converts to basis-point changes for stationarity.",
    },
    "2-Year Treasury Yield": {
        "fred_id": "DGS2", "category": "Rates & Yields",
        "rec_transform": "diff",
        "rec_reason": "Short-term rate. Difference gives rate-of-change in bps.",
    },
    "10-Year Real Interest Rate": {
        "fred_id": "REAINTRATREARAT10Y", "category": "Rates & Yields",
        "rec_transform": "level",
        "rec_reason": "Real rate can be negative. Level preserves sign.",
    },
    "High Yield OAS Spread": {
        "fred_id": "BAMLH0A0HYM2", "category": "Rates & Yields",
        "rec_transform": "level",
        "rec_reason": "Credit spread in bps. Level shows risk appetite; spikes mark stress events.",
    },

    # Inflation & Prices
    "CPI - All Urban Consumers": {
        "fred_id": "CPIAUCSL", "category": "Inflation & Prices",
        "rec_transform": "yoy",
        "rec_reason": "Price index level. YoY % gives the headline inflation rate.",
    },
    "Core CPI (ex Food & Energy)": {
        "fred_id": "CPILFESL", "category": "Inflation & Prices",
        "rec_transform": "yoy",
        "rec_reason": "Core price index. YoY % reads underlying inflation trends.",
    },
    "Personal Consumption Expenditures": {
        "fred_id": "PCE", "category": "Inflation & Prices",
        "rec_transform": "yoy",
        "rec_reason": "Nominal spending level. YoY % shows growth and removes seasonality.",
    },
    "PCE Price Index": {
        "fred_id": "PCEPI", "category": "Inflation & Prices",
        "rec_transform": "yoy",
        "rec_reason": "The Fed's preferred inflation gauge. YoY % is the standard policy metric.",
    },
    "10-Year Breakeven Inflation": {
        "fred_id": "T10YIE", "category": "Inflation & Prices",
        "rec_transform": "level",
        "rec_reason": "Already in % (market-implied inflation). Level shows expectations.",
    },
    "Gold Price (USD/Troy Oz)": {
        "fred_id": "GOLDPMGBD228NLBM", "category": "Inflation & Prices",
        "rec_transform": "pct",
        "rec_reason": "Asset price level. Log Returns remove trend for stationary analysis.",
    },

    # Real Economy
    "Unemployment Rate": {
        "fred_id": "UNRATE", "category": "Real Economy",
        "rec_transform": "level",
        "rec_reason": "Already in %. Level is standard for labor market analysis.",
    },
    "Nonfarm Payrolls": {
        "fred_id": "PAYEMS", "category": "Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Employment count in thousands. YoY % shows job growth pace.",
    },
    "Real GDP": {
        "fred_id": "GDPC1", "category": "Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Quarterly GDP level. YoY % gives the standard economic growth rate.",
    },
    "Industrial Production Index": {
        "fred_id": "INDPRO", "category": "Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Index level with trends. YoY % reveals the cyclical manufacturing signal.",
    },
    "Retail Sales": {
        "fred_id": "RSAFS", "category": "Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Nominal sales in $M. YoY % strips out seasonality and inflation drift.",
    },
    "Housing Starts": {
        "fred_id": "HOUST", "category": "Real Economy",
        "rec_transform": "yoy",
        "rec_reason": "Thousands of units. YoY % reveals the housing cycle.",
    },

    # Sentiment & Markets
    "CBOE VIX Index": {
        "fred_id": "VIXCLS", "category": "Sentiment & Markets",
        "rec_transform": "level",
        "rec_reason": "Implied volatility in %. Level is standard; mean-reverting by nature.",
    },
    "S&P 500 Index": {
        "fred_id": "SP500", "category": "Sentiment & Markets",
        "rec_transform": "pct",
        "rec_reason": "Price index. Log Returns are essential for removing the upward trend.",
    },
    "M2 Money Supply": {
        "fred_id": "WM2NS", "category": "Sentiment & Markets",
        "rec_transform": "yoy",
        "rec_reason": "Monetary aggregate in $B. YoY % shows liquidity growth.",
    },
    "Consumer Sentiment (UMich)": {
        "fred_id": "UMCSENT", "category": "Sentiment & Markets",
        "rec_transform": "level",
        "rec_reason": "Survey index. Level is standard; already bounded and comparable.",
    },
    "Trade-Weighted USD Index": {
        "fred_id": "DTWEXBGS", "category": "Sentiment & Markets",
        "rec_transform": "pct",
        "rec_reason": "Currency index level. Log Returns show dollar momentum.",
    },
    "Initial Jobless Claims": {
        "fred_id": "ICSA", "category": "Sentiment & Markets",
        "rec_transform": "level",
        "rec_reason": "Weekly claims count. Level is the standard read.",
    },
}

TRANSFORM_OPTIONS = {
    "Raw Level": "level",
    "% Change (Log Returns)": "pct",
    "Absolute Difference (Δ)": "diff",
    "Year-over-Year % Change": "yoy",
}
TRANSFORM_CODE_TO_KEY = {v: k for k, v in TRANSFORM_OPTIONS.items()}

# Color palette
C_LINE_A = "#4c8eda"   # blue
C_LINE_B = "#d97757"   # warm orange
C_GRID   = "#1c2330"
C_BG     = "#0d1117"
C_TEXT   = "#d8dee9"
C_MUTED  = "#7d8597"


# ════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def build_series_list():
    return list(SERIES_LIBRARY.keys())

def get_fred_id(friendly_name):
    return SERIES_LIBRARY[friendly_name]["fred_id"]

def get_recommended_transform(friendly_name):
    meta = SERIES_LIBRARY.get(friendly_name, {})
    return {"code": meta.get("rec_transform", "level"),
            "reason": meta.get("rec_reason", "")}

def format_series_option(name):
    """Format selectbox option: 'Friendly Name · FRED_ID'."""
    meta = SERIES_LIBRARY.get(name, {})
    return f"{name}  ·  {meta.get('fred_id', '')}"


def apply_transform(series, mode):
    """Apply the chosen stationarity transformation."""
    if mode == "pct":
        return np.log(series / series.shift(1)).dropna() * 100
    elif mode == "diff":
        return series.diff().dropna()
    elif mode == "yoy":
        freq_guess = pd.infer_freq(series.index)
        if freq_guess and "Q" in freq_guess:
            return series.pct_change(periods=4).dropna() * 100
        return series.pct_change(periods=12).dropna() * 100
    return series


def resample_to_lower_freq(df):
    """Resample both columns to the lower of the two estimated frequencies."""
    freqs = {"D": 1, "W": 7, "MS": 30, "QS": 90, "AS": 365}

    def estimate_freq(s):
        diffs = s.dropna().index.to_series().diff().dt.days.median()
        if diffs is None or np.isnan(diffs): return "MS"
        if diffs < 3: return "D"
        if diffs < 14: return "W"
        if diffs < 75: return "MS"
        return "QS"

    f1, f2 = estimate_freq(df.iloc[:, 0]), estimate_freq(df.iloc[:, 1])
    target = f1 if freqs.get(f1, 30) >= freqs.get(f2, 30) else f2
    return df.resample(target).last().dropna()


def compute_correlation_stats(x, y):
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
            "r": float(np.corrcoef(x_clean, y_clean)[0, 1]),
            "r2": float(model.rsquared),
            "pvalue": float(model.pvalues[1]) if len(model.pvalues) > 1 else np.nan,
            "slope": float(model.params[1]) if len(model.params) > 1 else np.nan,
            "intercept": float(model.params[0]),
            "n": n,
        }
    else:
        from scipy import stats as sp_stats
        r = float(np.corrcoef(x_clean, y_clean)[0, 1])
        slope = np.polyfit(x_clean, y_clean, 1)
        _, pval = sp_stats.pearsonr(x_clean, y_clean)
        return {"r": r, "r2": r**2, "pvalue": float(pval),
                "slope": float(slope[0]), "intercept": float(slope[1]), "n": n}


def adf_test(series):
    """
    Augmented Dickey-Fuller test.
    H0: unit root (non-stationary).
    Reject H0 (i.e. conclude stationarity) when p-value < 0.05.
    """
    clean = series.dropna()
    if len(clean) < 20 or not STATSMODELS_AVAILABLE:
        return {"stat": np.nan, "pvalue": np.nan,
                "stationary": None, "n": len(clean)}
    try:
        result = adfuller(clean.values, autolag="AIC")
        return {"stat": float(result[0]), "pvalue": float(result[1]),
                "stationary": bool(result[1] < 0.05), "n": len(clean)}
    except Exception:
        return {"stat": np.nan, "pvalue": np.nan,
                "stationary": None, "n": len(clean)}


def granger_test(cause, effect, max_lag=4):
    """
    Granger causality test: does `cause` Granger-cause `effect`?
    H0: `cause` does NOT Granger-cause `effect`.
    Reject H0 (i.e. conclude Granger-causality) when p-value < 0.05.

    Statsmodels expects a 2-column array where the FIRST column is the
    series being predicted (effect) and the SECOND column is the
    candidate causal series (cause).

    Returns the minimum p-value across all tested lags (1..max_lag)
    using the F-test from the SSR-based test, along with the lag
    that achieved it and a per-lag breakdown.
    """
    df_test = pd.concat([effect, cause], axis=1).dropna()
    n = len(df_test)
    # Need enough observations to estimate the unrestricted model
    if n < (max_lag * 3 + 10) or not STATSMODELS_AVAILABLE:
        return {"min_pvalue": np.nan, "best_lag": None,
                "causes": None, "per_lag": [], "n": n}

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(
                df_test.values, maxlag=max_lag, verbose=False
            )

        per_lag = []
        for lag, res in results.items():
            # ssr_ftest returns (F, p, df_denom, df_num)
            p = float(res[0]["ssr_ftest"][1])
            f = float(res[0]["ssr_ftest"][0])
            per_lag.append({"lag": int(lag), "pvalue": p, "fstat": f})

        if not per_lag:
            return {"min_pvalue": np.nan, "best_lag": None,
                    "causes": None, "per_lag": [], "n": n}

        best = min(per_lag, key=lambda d: d["pvalue"])
        return {
            "min_pvalue": best["pvalue"],
            "best_lag": best["lag"],
            "best_fstat": best["fstat"],
            "causes": bool(best["pvalue"] < 0.05),
            "per_lag": per_lag,
            "n": n,
        }
    except Exception:
        return {"min_pvalue": np.nan, "best_lag": None,
                "causes": None, "per_lag": [], "n": n}


def significance_label(p):
    if np.isnan(p): return "n/a"
    if p < 0.001: return "p < 0.001 (***)"
    if p < 0.01:  return "p < 0.01 (**)"
    if p < 0.05:  return "p < 0.05 (*)"
    return "not significant"


# ════════════════════════════════════════════════════════════════════
# 3. DATA FETCHING (CACHED)
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(api_key, series_id, start, end):
    """Fetch a single FRED series with caching."""
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start=start, observation_end=end)
    data.name = series_id
    data.index = pd.DatetimeIndex(data.index)
    return data.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recession_periods(api_key, start, end):
    """
    Fetch NBER recession indicator (USREC) and return a list of
    (start_date, end_date) tuples for each recession period.
    USREC equals 1 during recessions, 0 otherwise.
    """
    try:
        fred = Fred(api_key=api_key)
        usrec = fred.get_series("USREC",
                                observation_start=start,
                                observation_end=end).dropna()
        if usrec.empty:
            return []

        periods = []
        in_rec = False
        period_start = None
        for date, value in usrec.items():
            if value == 1 and not in_rec:
                in_rec = True
                period_start = date
            elif value == 0 and in_rec:
                in_rec = False
                periods.append((period_start, date))
        if in_rec:
            periods.append((period_start, usrec.index[-1]))
        return periods
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════════
# 4. PLOTLY CHART BUILDERS
# ════════════════════════════════════════════════════════════════════

PLOTLY_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=C_BG,
    plot_bgcolor=C_BG,
    font=dict(family="Inter, sans-serif", color=C_TEXT, size=11),
    margin=dict(l=60, r=60, t=50, b=45),
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(size=10, color=C_TEXT),
        bgcolor="rgba(0,0,0,0)",
    ),
)


def add_recession_shading(fig, periods, x_min, x_max):
    """Add discreet grey vertical bands for NBER recession periods."""
    if not periods:
        return
    for rec_start, rec_end in periods:
        if rec_end < x_min or rec_start > x_max:
            continue
        fig.add_vrect(
            x0=max(rec_start, x_min),
            x1=min(rec_end, x_max),
            fillcolor="rgba(170, 180, 200, 0.08)",
            line_width=0,
            layer="below",
        )


def build_dual_axis_chart(df, col_a, col_b, label_a, label_b, lag,
                          recession_periods=None):
    """Time-series chart with dual Y axes and recession shading."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    lag_suffix = f" (lag {lag:+d}m)" if lag != 0 else ""

    add_recession_shading(fig, recession_periods, df.index.min(), df.index.max())

    fig.add_trace(
        go.Scatter(x=df.index, y=df[col_a], name=f"{label_a}{lag_suffix}",
                   line=dict(color=C_LINE_A, width=1.6),
                   hovertemplate="%{y:.4f}"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[col_b], name=label_b,
                   line=dict(color=C_LINE_B, width=1.6),
                   hovertemplate="%{y:.4f}"),
        secondary_y=True,
    )

    fig.update_layout(
        **PLOTLY_DEFAULTS,
        title=dict(
            text="Time Series Comparison",
            font=dict(family="JetBrains Mono, monospace", size=13, color=C_TEXT),
            x=0.01,
        ),
        height=420,
    )
    fig.update_xaxes(gridcolor=C_GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(title_text=label_a, secondary_y=False,
                     gridcolor=C_GRID, zeroline=False,
                     title_font=dict(color=C_LINE_A, size=11),
                     tickfont=dict(color=C_LINE_A, size=10))
    fig.update_yaxes(title_text=label_b, secondary_y=True,
                     gridcolor="rgba(28,35,48,0.4)", zeroline=False,
                     title_font=dict(color=C_LINE_B, size=11),
                     tickfont=dict(color=C_LINE_B, size=10))
    return fig


def build_scatter_chart(x, y, label_x, label_y, stats):
    """Scatter plot with OLS regression line."""
    mask = x.notna() & y.notna()
    xc, yc = x[mask], y[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xc, y=yc, mode="markers",
        marker=dict(color=C_LINE_A, size=4, opacity=0.55,
                    line=dict(width=0)),
        name="Observations",
        hovertemplate=f"{label_x}: %{{x:.3f}}<br>{label_y}: %{{y:.3f}}",
    ))

    if not np.isnan(stats["slope"]):
        x_range = np.linspace(xc.min(), xc.max(), 100)
        y_hat = stats["intercept"] + stats["slope"] * x_range
        eq_text = (f"y = {stats['slope']:.4f}·x "
                   f"{'+ ' if stats['intercept'] >= 0 else '− '}"
                   f"{abs(stats['intercept']):.4f}")
        fig.add_trace(go.Scatter(
            x=x_range, y=y_hat, mode="lines",
            line=dict(color=C_LINE_B, width=1.4, dash="dot"),
            name=f"OLS: {eq_text}",
        ))

    fig.update_layout(
        **PLOTLY_DEFAULTS,
        title=dict(text="Correlation Scatter with OLS Regression",
                   font=dict(family="JetBrains Mono, monospace", size=13, color=C_TEXT),
                   x=0.01),
        xaxis=dict(title=label_x, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(title=label_y, gridcolor=C_GRID, zeroline=False),
        height=420,
    )
    return fig


def build_rolling_corr_chart(df, col_a, col_b, recession_periods=None,
                             windows=(30, 90, 252)):
    """Rolling correlation chart for multiple windows."""
    fig = go.Figure()
    add_recession_shading(fig, recession_periods, df.index.min(), df.index.max())

    colors = [C_LINE_A, "#8a9bd0", C_LINE_B]
    for w, c in zip(windows, colors):
        if len(df) > w:
            rolling = df[col_a].rolling(w).corr(df[col_b])
            fig.add_trace(go.Scatter(
                x=df.index, y=rolling, name=f"{w}-period",
                line=dict(color=c, width=1.4),
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="#3a4256", line_width=0.7)
    fig.add_hline(y=0.7, line_dash="dot", line_color="rgba(76,142,218,0.25)", line_width=0.5)
    fig.add_hline(y=-0.7, line_dash="dot", line_color="rgba(217,119,87,0.25)", line_width=0.5)

    fig.update_layout(
        **PLOTLY_DEFAULTS,
        title=dict(text="Rolling Correlation",
                   font=dict(family="JetBrains Mono, monospace", size=13, color=C_TEXT),
                   x=0.01),
        yaxis=dict(title="Pearson r", gridcolor=C_GRID, range=[-1.05, 1.05]),
        xaxis=dict(gridcolor=C_GRID),
        height=340,
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# 5. SIDEBAR
# ════════════════════════════════════════════════════════════════════

series_names = build_series_list()

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:18px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.95rem;
                    font-weight:500; color:#d8dee9; letter-spacing:0.02em;">
            MACRO TERMINAL
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                    color:#5a6275; letter-spacing:0.08em; margin-top:2px;">
            CORRELATION ENGINE · v3.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── FRED API key: prefer st.secrets, fall back to manual entry ──
    api_key = None
    try:
        api_key = st.secrets["FRED_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    if not api_key:
        api_key = st.text_input(
            "FRED API Key", type="password",
            help="Get a free key at fred.stlouisfed.org/docs/api/api_key.html",
            placeholder="Enter your FRED API key",
        )

    if api_key:
        st.session_state["fred_api_key"] = api_key

    st.markdown("---")

    # ── Variable A ──
    st.markdown('<div class="section-label">Variable A</div>', unsafe_allow_html=True)

    sel_a = st.selectbox("Select series A", series_names, index=0, key="sel_a",
                         format_func=format_series_option,
                         help="Curated library, or use custom ID below.")
    custom_a = st.text_input("Or custom FRED ID", key="custom_a",
                             placeholder="e.g. DCOILWTICO")
    is_custom_a = bool(custom_a.strip())
    series_id_a = custom_a.strip().upper() if is_custom_a else get_fred_id(sel_a)

    if not is_custom_a:
        rec_a = get_recommended_transform(sel_a)
        rec_key_a = TRANSFORM_CODE_TO_KEY.get(rec_a["code"], "Raw Level")
        rec_idx_a = list(TRANSFORM_OPTIONS.keys()).index(rec_key_a)
        st.markdown(
            f'<div class="rec-block">'
            f'<div class="rec-title">Recommended: {rec_key_a}</div>'
            f'<div class="rec-explain">{rec_a["reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        rec_idx_a = 0

    transform_a = st.selectbox(
        "Transform A", list(TRANSFORM_OPTIONS.keys()),
        index=rec_idx_a, key="tf_a",
        help="Raw Level: rates and bounded series. Log Returns: prices and indices. "
             "Δ: interest rates (bps moves). YoY %: seasonal economic series.",
    )

    st.markdown("---")

    # ── Variable B ──
    st.markdown('<div class="section-label">Variable B</div>', unsafe_allow_html=True)

    sel_b = st.selectbox("Select series B", series_names, index=5, key="sel_b",
                         format_func=format_series_option,
                         help="Curated library, or use custom ID below.")
    custom_b = st.text_input("Or custom FRED ID", key="custom_b",
                             placeholder="e.g. DEXUSEU")
    is_custom_b = bool(custom_b.strip())
    series_id_b = custom_b.strip().upper() if is_custom_b else get_fred_id(sel_b)

    if not is_custom_b:
        rec_b = get_recommended_transform(sel_b)
        rec_key_b = TRANSFORM_CODE_TO_KEY.get(rec_b["code"], "Raw Level")
        rec_idx_b = list(TRANSFORM_OPTIONS.keys()).index(rec_key_b)
        st.markdown(
            f'<div class="rec-block">'
            f'<div class="rec-title">Recommended: {rec_key_b}</div>'
            f'<div class="rec-explain">{rec_b["reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        rec_idx_b = 0

    transform_b = st.selectbox(
        "Transform B", list(TRANSFORM_OPTIONS.keys()),
        index=rec_idx_b, key="tf_b",
        help="Same options as Transform A. The recommendation depends on the nature "
             "of each series; correlating raw levels of two trending series produces "
             "spurious correlation (Yule, 1926).",
    )

    st.markdown("---")

    # ── Parameters ──
    st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)

    today = datetime.today().date()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Start", value=datetime(2000, 1, 1).date(),
            min_value=datetime(1947, 1, 1).date(),
            max_value=today,
            help="Earliest observation date. FRED coverage begins in 1947 for most series.",
        )
    with col_d2:
        end_date = st.date_input(
            "End", value=today,
            min_value=datetime(1947, 1, 1).date(),
            max_value=today,
            help="Latest observation date. Capped at today (no future data).",
        )

    lag = st.slider("Lag (months)", -24, 24, 0, 1,
                    help="Shift A relative to B. Positive lag tests whether A leads B.")

    auto_resample = st.checkbox("Auto-resample to lower frequency", value=True)
    show_recessions = st.checkbox("Show NBER recession shading", value=True,
                                  help="Overlays grey bands for NBER-dated US recessions.")

    granger_max_lag = st.slider(
        "Granger max lag", 1, 12, 4, 1,
        help="Maximum lag tested in the Granger causality test. "
             "The reported p-value is the minimum across all lags 1..L.",
    )


# ════════════════════════════════════════════════════════════════════
# 6. MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="terminal-header">
    <h1>Macro Correlation Terminal</h1>
    <div class="subtitle">
        FRED-powered bivariate analysis · Correlation · Lead-Lag · OLS Regression · Stationarity Testing
    </div>
</div>
""", unsafe_allow_html=True)


if not FRED_AVAILABLE:
    st.error("`fredapi` is not installed. Run `pip install fredapi`.")
    st.stop()

if not api_key:
    st.markdown("""
    <div class="note">
        Enter your FRED API key in the sidebar to begin. Free keys are available at
        <a href="https://fred.stlouisfed.org/docs/api/api_key.html"
           style="color:#4c8eda;" target="_blank">fred.stlouisfed.org</a>.
    </div>
    <div class="note">
        <strong>Example setups:</strong><br>
        UNRATE (Level) vs CPIAUCSL (YoY %) — Phillips curve.<br>
        T10Y2Y (Level) vs UNRATE (Level) at lag +12 — yield curve as recession leading indicator.<br>
        WM2NS (YoY %) vs CPIAUCSL (YoY %) at lag +18 — quantity theory of money.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Date range validation ──
if start_date >= end_date:
    st.markdown(
        '<div class="note">Start date must be before end date.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Fetch data ──
with st.spinner(f"Fetching {series_id_a}…"):
    try:
        raw_a = fetch_fred_series(api_key, series_id_a,
                                  start_date.strftime("%Y-%m-%d"),
                                  end_date.strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"Could not fetch {series_id_a}: {e}")
        st.stop()

with st.spinner(f"Fetching {series_id_b}…"):
    try:
        raw_b = fetch_fred_series(api_key, series_id_b,
                                  start_date.strftime("%Y-%m-%d"),
                                  end_date.strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"Could not fetch {series_id_b}: {e}")
        st.stop()

# ── Inform the user when requested range exceeds actual data coverage ──
effective_start = max(raw_a.index.min(), raw_b.index.min())
effective_end = min(raw_a.index.max(), raw_b.index.max())
requested_start = pd.Timestamp(start_date)

if effective_start > requested_start:
    st.markdown(
        f'<div class="note">Effective start: {effective_start.date()}. '
        f'Requested {start_date} but earliest common observation for '
        f'{series_id_a} / {series_id_b} is {effective_start.date()}.</div>',
        unsafe_allow_html=True,
    )


# ── Transform ──
mode_a = TRANSFORM_OPTIONS[transform_a]
mode_b = TRANSFORM_OPTIONS[transform_b]
trans_a = apply_transform(raw_a, mode_a)
trans_b = apply_transform(raw_b, mode_b)

label_a = f"{series_id_a} ({mode_a})"
label_b = f"{series_id_b} ({mode_b})"

# ── Lag ──
if lag != 0:
    trans_a = trans_a.shift(lag)

# ── Merge & resample ──
df = pd.DataFrame({label_a: trans_a, label_b: trans_b})
if auto_resample:
    df = resample_to_lower_freq(df)
df = df.dropna()

if len(df) < 3:
    st.warning("Not enough overlapping data points after transformations. "
               "Adjust date range, transforms, or lag.")
    st.stop()


# ── Fetch recession periods (if enabled) ──
recession_periods = []
if show_recessions:
    recession_periods = fetch_recession_periods(
        api_key,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )


# ── Correlation statistics ──
stats = compute_correlation_stats(df[label_a], df[label_b])

r_d  = f"{stats['r']:.4f}"      if not np.isnan(stats['r'])      else "—"
r2_d = f"{stats['r2']:.4f}"     if not np.isnan(stats['r2'])     else "—"
p_d  = f"{stats['pvalue']:.2e}" if not np.isnan(stats['pvalue']) else "—"

st.markdown(f"""
<div class="metric-strip">
    <div class="metric-card">
        <div class="label">Pearson r</div>
        <div class="value">{r_d}</div>
        <div class="sub">Linear correlation coefficient</div>
    </div>
    <div class="metric-card">
        <div class="label">R²</div>
        <div class="value">{r2_d}</div>
        <div class="sub">{(stats['r2']*100 if not np.isnan(stats['r2']) else 0):.1f}% variance explained</div>
    </div>
    <div class="metric-card">
        <div class="label">P-value</div>
        <div class="value">{p_d}</div>
        <div class="sub">{significance_label(stats['pvalue'])}</div>
    </div>
    <div class="metric-card">
        <div class="label">Observations</div>
        <div class="value">{stats['n']:,}</div>
        <div class="sub">Lag: {lag:+d} months</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── ADF stationarity tests ──
adf_a = adf_test(df[label_a])
adf_b = adf_test(df[label_b])

def format_adf_cell(series_id, transform_key, adf_result):
    if adf_result["stationary"] is None:
        status, cls, pval = "N/A", "adf-na", "—"
    elif adf_result["stationary"]:
        status, cls = "Stationary", "adf-ok"
        pval = f"p = {adf_result['pvalue']:.4f}"
    else:
        status, cls = "Non-stationary", "adf-fail"
        pval = f"p = {adf_result['pvalue']:.4f}"

    return (
        f'<div class="adf-cell">'
        f'<div class="adf-label">ADF · {series_id} ({transform_key})</div>'
        f'<div class="adf-value {cls}">{status}</div>'
        f'<div class="adf-meta">{pval} &nbsp;·&nbsp; n = {adf_result["n"]}</div>'
        f'</div>'
    )

adf_html = (
    '<div class="adf-row">'
    + format_adf_cell(series_id_a, transform_a, adf_a)
    + format_adf_cell(series_id_b, transform_b, adf_b)
    + '</div>'
)
st.markdown(adf_html, unsafe_allow_html=True)

# Sober interpretation note when at least one series fails the test
if (adf_a.get("stationary") is False) or (adf_b.get("stationary") is False):
    st.markdown(
        '<div class="note">At least one transformed series fails the ADF test '
        '(H₀: unit root, not rejected at the 5% level). The Pearson r reported '
        'above may reflect spurious correlation between non-stationary processes '
        '(Yule, 1926). Consider re-running with a stronger transform (e.g. '
        'Difference or YoY) or proceeding with caution when interpreting '
        'causality.</div>',
        unsafe_allow_html=True,
    )


# ── Granger causality tests (both directions) ──
granger_ab = granger_test(df[label_a], df[label_b], max_lag=granger_max_lag)
granger_ba = granger_test(df[label_b], df[label_a], max_lag=granger_max_lag)

both_stationary = (adf_a.get("stationary") is True) and (adf_b.get("stationary") is True)

def format_granger_cell(cause_id, effect_id, result, valid):
    """
    Render one direction of the Granger causality test.
    `valid` is False when the underlying ADF tests indicate non-stationarity,
    in which case we display the result but flag it as unreliable.
    """
    direction = f"{cause_id} → {effect_id}"

    if result["causes"] is None:
        verdict, cls, meta = "N/A", "granger-nocause", "Insufficient observations"
    elif result["causes"]:
        verdict, cls = "Granger-causes", "granger-causes"
        meta = (f"min p = {result['min_pvalue']:.4f} at lag {result['best_lag']} "
                f"&nbsp;·&nbsp; n = {result['n']}")
    else:
        verdict, cls = "No causality", "granger-nocause"
        meta = (f"min p = {result['min_pvalue']:.4f} at lag {result['best_lag']} "
                f"&nbsp;·&nbsp; n = {result['n']}")

    if not valid and result["causes"] is not None:
        cls = "granger-warn"
        meta = meta + " &nbsp;·&nbsp; unreliable (non-stationary inputs)"

    return (
        f'<div class="granger-cell">'
        f'<div class="granger-direction">{direction}</div>'
        f'<div class="granger-verdict {cls}">{verdict}</div>'
        f'<div class="granger-meta">{meta}</div>'
        f'</div>'
    )

granger_html = (
    '<div class="granger-row">'
    + format_granger_cell(series_id_a, series_id_b, granger_ab, both_stationary)
    + format_granger_cell(series_id_b, series_id_a, granger_ba, both_stationary)
    + '</div>'
)
st.markdown(granger_html, unsafe_allow_html=True)

# Concise technical interpretation
if granger_ab["causes"] is not None and granger_ba["causes"] is not None:
    if granger_ab["causes"] and granger_ba["causes"]:
        granger_summary = (
            f"Bidirectional Granger-causality detected (feedback): past values of "
            f"{series_id_a} help predict {series_id_b} and vice versa."
        )
    elif granger_ab["causes"]:
        granger_summary = (
            f"Unidirectional Granger-causality: past values of {series_id_a} help "
            f"predict {series_id_b}, but not the reverse. Consistent with "
            f"{series_id_a} acting as a leading indicator."
        )
    elif granger_ba["causes"]:
        granger_summary = (
            f"Unidirectional Granger-causality: past values of {series_id_b} help "
            f"predict {series_id_a}, but not the reverse. Consistent with "
            f"{series_id_b} acting as a leading indicator."
        )
    else:
        granger_summary = (
            f"No Granger-causality detected in either direction at lags 1..{granger_max_lag}. "
            f"Any observed correlation is contemporaneous rather than predictive."
        )

    if not both_stationary:
        granger_summary += (
            " Note: at least one series is non-stationary per the ADF test; "
            "Granger results should be interpreted with caution."
        )

    st.markdown(f'<div class="note">{granger_summary}</div>', unsafe_allow_html=True)

# Per-lag breakdown in an expander (for methodological transparency)
with st.expander("Granger causality — per-lag breakdown", expanded=False):
    st.markdown(
        '<div class="note">F-test p-values for the null hypothesis that the candidate '
        'causal series does NOT Granger-cause the target. p &lt; 0.05 rejects the null.</div>',
        unsafe_allow_html=True,
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(
            f'<div class="section-label">{series_id_a} → {series_id_b}</div>',
            unsafe_allow_html=True,
        )
        if granger_ab["per_lag"]:
            lag_df_ab = pd.DataFrame(granger_ab["per_lag"])
            lag_df_ab["pvalue"] = lag_df_ab["pvalue"].map(lambda v: f"{v:.4f}")
            lag_df_ab["fstat"] = lag_df_ab["fstat"].map(lambda v: f"{v:.3f}")
            lag_df_ab.columns = ["Lag", "p-value", "F-statistic"]
            st.dataframe(lag_df_ab, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="note">No results.</div>', unsafe_allow_html=True)

    with col_g2:
        st.markdown(
            f'<div class="section-label">{series_id_b} → {series_id_a}</div>',
            unsafe_allow_html=True,
        )
        if granger_ba["per_lag"]:
            lag_df_ba = pd.DataFrame(granger_ba["per_lag"])
            lag_df_ba["pvalue"] = lag_df_ba["pvalue"].map(lambda v: f"{v:.4f}")
            lag_df_ba["fstat"] = lag_df_ba["fstat"].map(lambda v: f"{v:.3f}")
            lag_df_ba.columns = ["Lag", "p-value", "F-statistic"]
            st.dataframe(lag_df_ba, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="note">No results.</div>', unsafe_allow_html=True)


# ── Charts ──
tab_ts, tab_scatter, tab_rolling, tab_data = st.tabs([
    "Time Series", "Scatter / OLS", "Rolling Correlation", "Data"
])

with tab_ts:
    fig_ts = build_dual_axis_chart(df, label_a, label_b, label_a, label_b, lag,
                                   recession_periods=recession_periods)
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})
    if recession_periods:
        st.markdown(
            '<div class="note">Grey bands mark NBER-dated US recessions (FRED: USREC).</div>',
            unsafe_allow_html=True,
        )

with tab_scatter:
    fig_sc = build_scatter_chart(df[label_a], df[label_b], label_a, label_b, stats)
    st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    if not np.isnan(stats["slope"]):
        st.markdown(f"""
        <div class="note">
            OLS regression: <code style="color:#d8dee9;">
            {label_b} = {stats['slope']:.6f} · {label_a}
            {'+ ' if stats['intercept'] >= 0 else '− '}{abs(stats['intercept']):.6f}
            </code> &nbsp; · &nbsp; R² = {stats['r2']:.4f} &nbsp; · &nbsp;
            p = {stats['pvalue']:.2e} &nbsp; · &nbsp; n = {stats['n']}
        </div>
        """, unsafe_allow_html=True)

with tab_rolling:
    fig_rc = build_rolling_corr_chart(df, label_a, label_b,
                                      recession_periods=recession_periods)
    st.plotly_chart(fig_rc, use_container_width=True, config={"displayModeBar": False})

with tab_data:
    st.dataframe(df.style.format("{:.4f}"), use_container_width=True, height=400)

    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=True)
    csv_buffer.seek(0)
    st.download_button(
        label="Download merged CSV",
        data=csv_buffer,
        file_name=f"macro_{series_id_a}_{series_id_b}_lag{lag}.csv",
        mime="text/csv",
    )


# ── Lag sensitivity scan ──
with st.expander("Lag Sensitivity Scan (−24 to +24 months)", expanded=False):
    st.markdown("""
    <div class="note">
        Scans all integer lags in the range and reports the lag that maximises |r|.
        Use exploratorily; the global optimum may reflect data mining if used without
        a prior hypothesis about lead-lag direction.
    </div>
    """, unsafe_allow_html=True)

    lag_results = []
    trans_a_base = apply_transform(raw_a, mode_a)
    for test_lag in range(-24, 25):
        shifted = trans_a_base.shift(test_lag)
        temp_df = pd.DataFrame({label_a: shifted, label_b: trans_b}).dropna()
        if len(temp_df) >= 10:
            r_val = float(np.corrcoef(temp_df[label_a].values,
                                      temp_df[label_b].values)[0, 1])
            lag_results.append({"lag": test_lag, "r": r_val,
                                "abs_r": abs(r_val), "n": len(temp_df)})

    if lag_results:
        lag_df = pd.DataFrame(lag_results)
        best = lag_df.loc[lag_df["abs_r"].idxmax()]

        bar_colors = []
        for _, row in lag_df.iterrows():
            if row["lag"] == best["lag"]:
                bar_colors.append(C_LINE_B)
            elif abs(row["r"]) >= 0.4:
                bar_colors.append(C_LINE_A)
            else:
                bar_colors.append("#2a3142")

        fig_lag = go.Figure(go.Bar(
            x=lag_df["lag"], y=lag_df["r"],
            marker_color=bar_colors,
            hovertemplate="Lag %{x}m: r = %{y:.4f}<extra></extra>",
        ))
        fig_lag.update_layout(
            **PLOTLY_DEFAULTS,
            title=dict(text=f"Optimal lag: {int(best['lag']):+d} months  ·  r = {best['r']:.4f}",
                       font=dict(family="JetBrains Mono, monospace", size=12, color=C_TEXT),
                       x=0.01),
            xaxis=dict(title="Lag (months)", gridcolor=C_GRID, dtick=3),
            yaxis=dict(title="Pearson r", gridcolor=C_GRID, range=[-1.05, 1.05]),
            height=320,
        )
        fig_lag.add_hline(y=0, line_dash="dash", line_color="#3a4256", line_width=0.7)
        st.plotly_chart(fig_lag, use_container_width=True, config={"displayModeBar": False})


# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:14px 0;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#5a6275;
                 letter-spacing:0.06em;">
        MACRO CORRELATION TERMINAL · Streamlit · FRED API · Plotly · Statsmodels<br>
        Data: Federal Reserve Bank of St. Louis · For research and educational purposes
    </span>
</div>
""", unsafe_allow_html=True)