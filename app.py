# =============================================================================
# app.py – Sales Forecasting Dashboard  |  Task 7
# Uses REAL logic from Tasks 1‑6 (train.csv, XGBoost, IsolationForest, KMeans)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import sqrt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from dateutil.relativedelta import relativedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS – enterprise emerald system ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,0,0');
:root {
    --bg: #f5f7f4;
    --card: #ffffff;
    --ink: #111827;
    --muted: #667085;
    --line: #dbe5dc;
    --soft: #eef4ee;
    --emerald: #047857;
    --forest: #064e3b;
    --success: #10b981;
    --amber: #d97706;
    --red: #dc2626;
    --shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
    --shadow-soft: 0 8px 22px rgba(15, 23, 42, 0.06);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at top left, #edf8ef 0, #f5f7f4 300px, #f5f7f4 100%); }
header[data-testid="stHeader"] { background: rgba(245, 247, 244, 0.84); backdrop-filter: blur(10px); }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { visibility: hidden; height: 0; }
/* sidebar collapse button hidden once open */
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { display: none !important; }
.block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 4rem; }

.material-symbols-rounded {
    font-family: 'Material Symbols Rounded';
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    direction: ltr;
}

/* Sidebar */
[data-testid="stSidebar"] { 
    background: #ffffff !important; 
    border-right: 1px solid var(--line);
    box-shadow: 10px 0 30px rgba(15, 23, 42, 0.04);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #334155; }
[data-testid="stSidebar"] .stRadio > div { gap: 10px; }
[data-testid="stSidebar"] .stRadio label { 
    padding: 11px 14px; 
    border-radius: 10px; 
    border: 1px solid transparent;
    transition: all 0.18s ease; 
    font-weight: 650;
}
[data-testid="stSidebar"] .stRadio label:hover { 
    background: #eef7f0; 
    border-color: #d5ead9;
    transform: translateX(2px);
}
[data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stMultiSelect, [data-testid="stSidebar"] .stSlider {
    background: #f8fbf8;
    border: 1px solid #e1ebe2;
    border-radius: 12px;
    padding: 10px 10px 8px;
    margin-bottom: 12px;
}

/* KPI Cards */
.kpi-card {
    position: relative;
    overflow: hidden;
    min-height: 168px;
    background: linear-gradient(180deg, #ffffff 0%, #fbfdfb 100%); 
    border-radius: 18px; 
    padding: 22px 22px 18px;
    box-shadow: var(--shadow-soft); 
    border: 1px solid rgba(219, 229, 220, 0.92);
    margin-bottom: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 4px;
    background: linear-gradient(180deg, var(--emerald), var(--success));
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow);
}
.kpi-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.kpi-icon {
    width: 38px; height: 38px; border-radius: 12px;
    display: grid; place-items: center;
    background: #e7f7ed; color: var(--forest);
}
.kpi-label { font-size: 11px; font-weight: 750; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-size: 30px; font-weight: 800; color: var(--ink); margin: 14px 0 8px; letter-spacing: 0; line-height: 1.05; overflow-wrap: anywhere; }
.kpi-delta { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 700; color: #047857; background: #e7f8ef; border-radius: 999px; padding: 5px 9px; }
.kpi-delta.neg { color: #b42318; background: #fee4e2; }
.kpi-spark { position: absolute; right: 16px; bottom: 12px; opacity: 0.9; width: 110px; height: 42px; }

.exec-card {
    background: linear-gradient(135deg, #ffffff 0%, #f7fbf7 100%);
    border: 1px solid rgba(219, 229, 220, 0.98);
    box-shadow: var(--shadow);
    border-radius: 20px;
    padding: 24px;
    margin: 8px 0 28px;
}
.exec-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
.exec-title { color: var(--ink); font-size: 18px; font-weight: 800; letter-spacing: 0; }
.exec-subtitle { color: var(--muted); font-size: 13px; margin-top: 4px; }
.exec-grid { display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 12px; }
.insight-chip {
    background: #ffffff;
    border: 1px solid #e2ebe3;
    border-radius: 14px;
    padding: 14px;
    min-height: 110px;
}
.insight-chip .material-symbols-rounded { color: var(--emerald); margin-bottom: 10px; }
.insight-label { color: var(--muted); font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: 0.05em; }
.insight-value { color: var(--ink); font-size: 15px; font-weight: 800; margin-top: 6px; line-height: 1.25; }

.section-kicker { color: var(--emerald); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 8px; }
.section-title { color: var(--ink); font-size: 21px; font-weight: 800; margin: 2px 0 14px; letter-spacing: 0; }
.context-strip {
    background: #ffffff;
    border: 1px solid #e2ebe3;
    border-radius: 14px;
    padding: 12px 16px;
    color: #475467;
    font-size: 13px;
    margin-bottom: 20px;
    box-shadow: 0 5px 14px rgba(15,23,42,0.04);
}

/* Headers */
h1 { color: var(--ink) !important; font-weight: 800 !important; font-size: 34px !important; letter-spacing: 0 !important; margin-bottom: 6px !important; }
h2, h3 { color: #18231d !important; font-weight: 750 !important; letter-spacing: 0 !important; }

/* Metric Box */
.metric-box {
    background: #ffffff; border-radius: 16px; padding: 24px;
    text-align: center; border: 1px solid #e2ebe3;
    box-shadow: var(--shadow-soft);
}
.metric-box .m-label { font-size: 12px; color: var(--muted); font-weight: 750;
                       letter-spacing: .05em; text-transform: uppercase; }
.metric-box .m-value { font-size: 26px; font-weight: 800; color: var(--ink); margin-top: 8px; letter-spacing: 0; }

[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #ffffff;
    border: 1px solid #e2ebe3;
    border-radius: 18px;
    padding: 12px;
    box-shadow: var(--shadow-soft);
}
[data-testid="stDataFrame"] {
    overflow: hidden;
}

/* Clean up expander styles */
.streamlit-expanderHeader {
    font-weight: 700 !important;
    color: var(--ink) !important;
    border-radius: 12px !important;
}

/* Button styling */
.stButton > button {
    border-radius: 8px;
    font-weight: 700;
    transition: all 0.2s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(4, 120, 87, 0.16);
}
.stDownloadButton > button {
    border-radius: 9px;
    font-weight: 750;
}
@media (max-width: 1100px) {
    .exec-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""", unsafe_allow_html=True)

# ── Force sidebar open on every page load ─────────────────────────────────────
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    // Clear any saved collapsed state for sidebar
    Object.keys(localStorage).forEach(function(k) {
        if (k.toLowerCase().indexOf('sidebar') !== -1) {
            localStorage.removeItem(k);
        }
    });
    // If sidebar is currently collapsed, click the expand arrow
    function tryExpand() {
        var btn = document.querySelector('[data-testid="collapsedControl"]');
        if (btn) { btn.click(); }
    }
    setTimeout(tryExpand, 300);
    setTimeout(tryExpand, 900);
})();
</script>
""", height=0)

# Premium deep emerald and slate palette
EMERALD = ["#064e3b", "#047857", "#059669", "#10b981", "#34d399", "#0f766e", "#14b8a6", "#84cc16"]

# =============================================================================
# ── REAL DATA LOADING (Task 1) ────────────────────────────────────────────────
# =============================================================================

@st.cache_data
def load_raw_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Year"]  = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df

# =============================================================================
# ── REAL FORECAST ENGINE (Task 2/3/4 – XGBoost recursive forecast) ───────────
# =============================================================================

def build_features(ts: pd.Series) -> pd.DataFrame:
    """Add lag + rolling + calendar features."""
    df = ts.to_frame(name="Sales")
    df["Lag1"] = df["Sales"].shift(1)
    df["Lag2"] = df["Sales"].shift(2)
    df["Lag3"] = df["Sales"].shift(3)
    df["RollingMean"] = df["Sales"].rolling(3).mean()
    df["Month"]   = df.index.month
    df["Quarter"] = df.index.quarter
    df["Year"]    = df.index.year
    def season(m):
        if m in [12,1,2]: return 1
        elif m in [3,4,5]: return 2
        elif m in [6,7,8]: return 3
        else: return 4
    df["Season"] = df["Month"].apply(season)
    return df.dropna()


@st.cache_data
def run_forecast(group_by: str, selected_value: str, horizon_months: int):
    """
    Real XGBoost recursive forecast.
    Returns (combined_df, mae, rmse)
    """
    raw = load_raw_data()
    if group_by == "Category":
        seg = raw[raw["Category"] == selected_value]
    else:
        seg = raw[raw["Region"] == selected_value]

    # Monthly aggregation
    monthly = (
        seg.set_index("Order Date")["Sales"]
        .resample("ME").sum()
    )

    full_df  = build_features(monthly)
    features = ["Lag1","Lag2","Lag3","RollingMean","Month","Quarter","Year","Season"]

    # Train on all but last 3 months; evaluate on last 3
    train_df = full_df.iloc[:-3]
    test_df  = full_df.iloc[-3:]

    model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                         max_depth=4, random_state=42)
    model.fit(train_df[features], train_df["Sales"])

    # Evaluate
    y_pred_test = model.predict(test_df[features])
    mae  = mean_absolute_error(test_df["Sales"], y_pred_test)
    rmse = sqrt(mean_squared_error(test_df["Sales"], y_pred_test))

    # Retrain on full data for future forecast
    model.fit(full_df[features], full_df["Sales"])

    # Recursive forecast for horizon_months steps
    last_sales = list(monthly.values)
    last_date  = monthly.index[-1]
    future_preds = []

    for i in range(horizon_months):
        fut_date = last_date + relativedelta(months=i+1)
        lag1 = last_sales[-1]
        lag2 = last_sales[-2] if len(last_sales) >= 2 else lag1
        lag3 = last_sales[-3] if len(last_sales) >= 3 else lag1
        roll = np.mean(last_sales[-3:])
        m = fut_date.month
        q = fut_date.quarter
        y = fut_date.year
        def s(mo):
            if mo in [12,1,2]: return 1
            elif mo in [3,4,5]: return 2
            elif mo in [6,7,8]: return 3
            else: return 4
        row = pd.DataFrame([[lag1,lag2,lag3,roll,m,q,y,s(m)]], columns=features)
        pred = model.predict(row)[0]
        future_preds.append((fut_date, pred))
        last_sales.append(pred)

    # Historical rows
    hist_rows = []
    for d, v in zip(monthly.index, monthly.values):
        hist_rows.append({"ds": d, "yhat": v, "yhat_lower": v*0.97,
                          "yhat_upper": v*1.03, "type": "Historical"})

    # Forecast rows (± 1.5*MAE for simple bounds)
    fut_rows = []
    for d, v in future_preds:
        fut_rows.append({"ds": d, "yhat": v, "yhat_lower": v - 1.5*mae,
                         "yhat_upper": v + 1.5*mae, "type": "Forecast"})

    combined = pd.concat([
        pd.DataFrame(hist_rows),
        pd.DataFrame(fut_rows)
    ], ignore_index=True)

    return combined, round(mae, 2), round(rmse, 2)

# =============================================================================
# ── REAL ANOMALY DETECTION (Task 5) ──────────────────────────────────────────
# =============================================================================

@st.cache_data
def get_anomaly_data():
    """
    Exact reproduction of Task 5: IsolationForest + Rolling Z-Score
    """
    df = load_raw_data()

    weekly = (
        df.set_index("Order Date")
        .resample("W-SUN")["Sales"]
        .sum()
        .to_frame()
    )

    weekly["Month"]   = weekly.index.month
    weekly["Week"]    = weekly.index.isocalendar().week.astype(int)
    weekly["Quarter"] = weekly.index.quarter

    # Isolation Forest
    feats = ["Sales","Month","Week","Quarter"]
    scaler = StandardScaler()
    X = scaler.fit_transform(weekly[feats])
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    weekly["IF_Anomaly"] = iso.fit_predict(X)
    weekly["IF_Status"]  = weekly["IF_Anomaly"].map({1:"Normal", -1:"Anomaly"})

    # Rolling Z-Score
    window = 8
    weekly["RollingMean"] = weekly["Sales"].rolling(window).mean()
    weekly["RollingStd"]  = weekly["Sales"].rolling(window).std()
    weekly["Rolling_Z"]   = (weekly["Sales"] - weekly["RollingMean"]) / weekly["RollingStd"]
    weekly["Z_Anomaly"]   = weekly["Rolling_Z"].abs() > 2

    # Anomaly type classification
    weekly["Anomaly Type"] = np.where(
        weekly["IF_Anomaly"] == -1,
        weekly.apply(lambda r: "High Sales Spike" if r["Sales"] >= r["RollingMean"]
                     else "Low Sales Drop", axis=1),
        "Normal"
    )

    def explanation(row):
        if row["Anomaly Type"] == "Normal":
            return "No anomaly."
        m = row.name.month
        if row["Anomaly Type"] == "High Sales Spike":
            if m == 11:   return "Spike likely corresponds to Black Friday promotions."
            elif m == 12: return "Holiday shopping and year-end sales."
            elif m in [9,10]: return "Possible festive season promotion."
            else:         return "Likely due to promotional campaigns or unusually high demand."
        else:
            if m in [1,2]:   return "Lower demand after the holiday season."
            elif m in [6,7]: return "Possible seasonal slowdown or inventory shortages."
            else:            return "Possible supply chain issue or temporary demand reduction."

    weekly["Possible Explanation"] = weekly.apply(explanation, axis=1)
    return weekly.reset_index()

# =============================================================================
# ── REAL CLUSTERING (Task 6) ─────────────────────────────────────────────────
# =============================================================================

@st.cache_data
def get_cluster_data():
    """
    Exact reproduction of Task 6: KMeans on sub-category features
    """
    df = load_raw_data()

    monthly = (
        df.groupby(["Sub-Category","Year","Month"])["Sales"]
        .sum().reset_index()
    )

    total_sales = monthly.groupby("Sub-Category")["Sales"].sum()
    volatility  = monthly.groupby("Sub-Category")["Sales"].std()
    avg_order   = df.groupby("Sub-Category")["Sales"].mean()

    # Growth Rate (YoY)
    yearly = df.groupby(["Sub-Category","Year"])["Sales"].sum().reset_index()
    growth = []
    for sub in yearly["Sub-Category"].unique():
        temp = yearly[yearly["Sub-Category"]==sub].sort_values("Year")
        if len(temp) >= 2:
            rate = ((temp.iloc[-1]["Sales"] - temp.iloc[0]["Sales"]) / temp.iloc[0]["Sales"]) * 100
        else:
            rate = 0
        growth.append([sub, rate])
    growth = pd.DataFrame(growth, columns=["Sub-Category","Growth Rate"])

    cluster_data = pd.DataFrame({
        "Total Sales":         total_sales,
        "Sales Volatility":    volatility,
        "Average Order Value": avg_order,
    }).reset_index()
    cluster_data = cluster_data.merge(growth, on="Sub-Category")
    cluster_data.fillna(0, inplace=True)

    features = ["Total Sales","Growth Rate","Sales Volatility","Average Order Value"]
    scaler   = StandardScaler()
    X        = scaler.fit_transform(cluster_data[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    cluster_data["Cluster"] = kmeans.fit_predict(X)

    cluster_labels = {
        0: "High Volume, Stable Demand",
        1: "Growing Demand",
        2: "Low Volume, High Volatility",
        3: "Declining Demand",
    }
    cluster_data["Demand Segment"] = cluster_data["Cluster"].map(cluster_labels)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X)
    cluster_data["PC1"] = pca_coords[:, 0]
    cluster_data["PC2"] = pca_coords[:, 1]

    return cluster_data

# =============================================================================
# ── UI HELPERS ────────────────────────────────────────────────────────────────
# =============================================================================

def format_money(value, decimals=0):
    if pd.isna(value):
        return "$0"
    return f"${value:,.{decimals}f}"


def pct_change(current, previous):
    if previous in (0, None) or pd.isna(previous):
        return 0
    return ((current - previous) / abs(previous)) * 100


def sparkline_svg(values, positive=True):
    vals = pd.Series(values).dropna().astype(float).tail(10).tolist()
    if len(vals) < 2:
        vals = [0, 0]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1
    points = []
    for idx, val in enumerate(vals):
        x = 6 + idx * (98 / max(len(vals) - 1, 1))
        y = 36 - ((val - lo) / span) * 28
        points.append(f"{x:.1f},{y:.1f}")
    stroke = "#047857" if positive else "#dc2626"
    fill = "rgba(16,185,129,0.12)" if positive else "rgba(220,38,38,0.10)"
    area_points = " ".join(points + ["104,40", "6,40"])
    line_points = " ".join(points)
    return (
        f"<svg class='kpi-spark' viewBox='0 0 112 44' preserveAspectRatio='none'>"
        f"<polygon points='{area_points}' fill='{fill}'/>"
        f"<polyline points='{line_points}' fill='none' stroke='{stroke}' stroke-width='2.5' "
        f"stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )


def kpi(col, label, value, delta="", neg=False, icon="monitoring", spark=None):
    d_cls = "neg" if neg else ""
    arrow = "south_east" if neg else "north_east"
    d_html = (
        f"<div class='kpi-delta {d_cls}'><span class='material-symbols-rounded'>{arrow}</span>{delta}</div>"
        if delta else ""
    )
    spark_html = sparkline_svg(spark, positive=not neg) if spark is not None else ""
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-top">
            <div class="kpi-label">{label}</div>
            <div class="kpi-icon"><span class="material-symbols-rounded">{icon}</span></div>
        </div>
        <div class="kpi-value">{value}</div>
        {d_html}
        {spark_html}
    </div>""", unsafe_allow_html=True)


def page_intro(title, caption):
    st.title(title)
    st.markdown(f"<div class='context-strip'>{caption}</div>", unsafe_allow_html=True)


def section_heading(kicker, title):
    st.markdown(
        f"<div class='section-kicker'>{kicker}</div><div class='section-title'>{title}</div>",
        unsafe_allow_html=True,
    )


def executive_insights(items, subtitle="Automatically generated from the active filters."):
    chips = []
    for icon, label, value in items:
        chips.append(
            f"<div class='insight-chip'>"
            f"<span class='material-symbols-rounded'>{icon}</span>"
            f"<div class='insight-label'>{label}</div>"
            f"<div class='insight-value'>{value}</div>"
            f"</div>"
        )
    st.markdown(
        f"""
        <div class="exec-card">
            <div class="exec-head">
                <div>
                    <div class="exec-title">Executive summary</div>
                    <div class="exec-subtitle">{subtitle}</div>
                </div>
                <span class="material-symbols-rounded" style="color:#047857;font-size:28px;">auto_awesome</span>
            </div>
            <div class="exec-grid">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_config():
    return {
        "displayModeBar": False,
        "responsive": True,
        "scrollZoom": False,
    }


def show_chart(fig):
    st.plotly_chart(fig, width="stretch", config=chart_config())


def show_table(df, **kwargs):
    kwargs.setdefault("width", "stretch")
    kwargs.setdefault("hide_index", True)
    kwargs.setdefault("height", "auto")
    st.dataframe(df, **kwargs)


def chart_style(fig, title=""):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=13, color="#475467"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=70, b=40, l=18, r=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=12, color="#475467")),
        colorway=EMERALD,
        hoverlabel=dict(bgcolor="#111827", font_size=12, font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color="#667085", size=11), linecolor="#e2ebe3", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef4ee",
                     tickfont=dict(color="#667085", size=11), linecolor="#e2ebe3", zeroline=False)
    if title:
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=17, color="#111827"), x=0.02, y=0.95)
        )
    else:
        fig.update_layout(title_text="")
    return fig

# =============================================================================
# ── PAGE 1: SALES OVERVIEW ────────────────────────────────────────────────────
# =============================================================================

def page_sales_overview():
    page_intro(
        "Sales overview",
        "Commercial performance cockpit with region, category, and year filters applied across every visual.",
    )
    df = load_raw_data()

    with st.sidebar:
        st.markdown("<div style='font-size:12px; font-weight:800; color:#047857; text-transform:uppercase; letter-spacing:.08em; margin:14px 0 10px;'>Data filters</div>", unsafe_allow_html=True)
        regions    = ["All"] + sorted(df["Region"].unique())
        categories = ["All"] + sorted(df["Category"].unique())
        sel_region = st.selectbox("Region", regions)
        sel_cat    = st.selectbox("Category", categories)
        all_years  = sorted(df["Year"].unique())
        sel_years  = st.multiselect("Year(s)", all_years, default=all_years)

    flt = df.copy()
    if sel_region != "All": flt = flt[flt["Region"] == sel_region]
    if sel_cat    != "All": flt = flt[flt["Category"] == sel_cat]
    if sel_years:           flt = flt[flt["Year"].isin(sel_years)]

    monthly = flt.groupby("Month")["Sales"].sum().reset_index()
    yearly = flt.groupby("Year")["Sales"].sum().reset_index()
    category_sales = flt.groupby("Category")["Sales"].sum()
    region_sales = flt.groupby("Region")["Sales"].sum()

    total_sales  = flt["Sales"].sum()
    total_orders = len(flt)
    avg_order    = flt["Sales"].mean()
    top_cat      = category_sales.idxmax() if not category_sales.empty else "N/A"
    top_region   = region_sales.idxmax() if not region_sales.empty else "N/A"
    current_year_sales = yearly["Sales"].iloc[-1] if len(yearly) else 0
    previous_year_sales = yearly["Sales"].iloc[-2] if len(yearly) > 1 else current_year_sales
    sales_delta = pct_change(current_year_sales, previous_year_sales)
    order_delta = pct_change(total_orders, len(df))
    avg_delta = pct_change(avg_order, df["Sales"].mean())

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total sales",     format_money(total_sales), f"{sales_delta:+.1f}% YoY", sales_delta < 0, "payments", monthly["Sales"])
    kpi(c2, "Total orders",    f"{total_orders:,}", f"{order_delta:+.1f}% vs dataset", order_delta < 0, "receipt_long", monthly["Sales"])
    kpi(c3, "Avg order value", format_money(avg_order, 2), f"{avg_delta:+.1f}% vs avg", avg_delta < 0, "shopping_cart", monthly["Sales"].rolling(2).mean())
    kpi(c4, "Top category",    top_cat, "Highest sales mix", False, "category", category_sales.values)

    growth_label = "expanding" if sales_delta >= 0 else "softening"
    recommendation = (
        f"Prioritize {top_cat} in {top_region}; the active view is {growth_label} at {sales_delta:+.1f}% YoY."
        if top_cat != "N/A" else "Broaden the filter selection to surface a recommendation."
    )
    executive_insights([
        ("workspace_premium", "Highest performing category", top_cat),
        ("trending_up", "Sales growth", f"{sales_delta:+.1f}% YoY"),
        ("public", "Best performing region", top_region),
        ("shopping_bag", "Average order value", format_money(avg_order, 2)),
        ("tips_and_updates", "Business recommendation", recommendation),
    ])

    section_heading("Primary forecast surface", "Revenue trajectory")

    # ── Total Sales by Year (Bar Chart) ───────────────────────────────────────
    fig1 = px.bar(yearly, x="Year", y="Sales",
                  text=yearly["Sales"].apply(lambda v: f"${v:,.0f}"),
                  color_discrete_sequence=["#064e3b"])
    fig1.update_traces(textposition="outside", marker_line_width=0, opacity=0.9)
    fig1.update_xaxes(type="category")
    chart_style(fig1, "Total Sales by Year")
    show_chart(fig1)

    # ── Monthly Sales Trend (Line Chart) ──────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly["Sales"],
        mode="lines", name="Monthly Sales",
        line=dict(color="#059669", width=3),
        fill="tozeroy", fillcolor="rgba(5, 150, 105, 0.08)"))
    chart_style(fig2, "Monthly Sales Trend")
    show_chart(fig2)

    # ── By Region & Category ──────────────────────────────────────────────────
    section_heading("Supporting analytics", "Regional and category composition")
    col_a, col_b = st.columns(2)

    with col_a:
        reg_df = flt.groupby("Region")["Sales"].sum().reset_index()
        fig3 = px.bar(reg_df, x="Region", y="Sales",
                      color="Region", color_discrete_sequence=EMERALD,
                      text=reg_df["Sales"].apply(lambda v: f"${v:,.0f}"))
        fig3.update_traces(textposition="outside", showlegend=False)
        chart_style(fig3, "Sales by Region")
        col_a.plotly_chart(fig3, width="stretch", config=chart_config())

    with col_b:
        cat_df = flt.groupby("Category")["Sales"].sum().reset_index()
        fig4 = px.pie(cat_df, names="Category", values="Sales",
                      color_discrete_sequence=EMERALD, hole=0.5)
        fig4.update_traces(textinfo="percent+label", pull=[0.02]*3)
        chart_style(fig4, "Sales by Category")
        col_b.plotly_chart(fig4, width="stretch", config=chart_config())

    # ── Heatmap: Region × Category ────────────────────────────────────────────
    heat = flt.groupby(["Region","Category"])["Sales"].sum().reset_index()
    pivot = heat.pivot(index="Region", columns="Category", values="Sales").fillna(0)
    fig5 = px.imshow(pivot, text_auto=".2s",
                     color_continuous_scale=["#f8fafc", "#34d399", "#064e3b"],
                     aspect="auto")
    chart_style(fig5, "Sales Heatmap: Region vs Category")
    show_chart(fig5)


# =============================================================================
# ── PAGE 2: FORECAST EXPLORER ─────────────────────────────────────────────────
# =============================================================================

def page_forecast_explorer():
    page_intro(
        "Forecast explorer",
        "Scenario view for XGBoost sales projections, confidence intervals, and model quality diagnostics.",
    )
    df = load_raw_data()

    with st.sidebar:
        st.markdown("<div style='font-size:12px; font-weight:800; color:#047857; text-transform:uppercase; letter-spacing:.08em; margin:14px 0 10px;'>Forecast parameters</div>", unsafe_allow_html=True)
        group_by = st.selectbox("Forecast Dimension", ["Category", "Region"])
        if group_by == "Category":
            options = sorted(df["Category"].unique())
        else:
            options = sorted(df["Region"].unique())
        selected = st.selectbox(f"Select {group_by}", options)
        horizon  = st.select_slider(
            "Forecast Horizon (Months)",
            options=[1, 2, 3], value=1)

    st.markdown(f"<div class='context-strip'>Model: <b>XGBoost</b> &nbsp;|&nbsp; Target: <b>{selected}</b> &nbsp;|&nbsp; Horizon: <b>{horizon} Month(s)</b></div>", unsafe_allow_html=True)

    with st.spinner("Executing XGBoost forecast..."):
        forecast_df, mae, rmse = run_forecast(group_by, selected, horizon)

    hist = forecast_df[forecast_df["type"] == "Historical"]
    fut  = forecast_df[forecast_df["type"] == "Forecast"]

    current_value = hist["yhat"].iloc[-1] if not hist.empty else 0
    forecast_value = fut["yhat"].iloc[-1] if not fut.empty else current_value
    forecast_growth = pct_change(forecast_value, current_value)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Latest actual", format_money(current_value), "Historical baseline", False, "history", hist["yhat"])
    kpi(c2, "Forecast close", format_money(forecast_value), f"{forecast_growth:+.1f}% vs latest", forecast_growth < 0, "trending_up", fut["yhat"])
    kpi(c3, "MAE", format_money(mae, 2), "Model error", False, "query_stats", [mae, mae * 0.96, mae * 1.02])
    kpi(c4, "RMSE", format_money(rmse, 2), f"{horizon} month horizon", False, "speed", [rmse, rmse * 0.98, rmse * 1.01])

    executive_insights([
        ("track_changes", "Forecast dimension", f"{group_by}: {selected}"),
        ("trending_up", "Highest forecast growth", f"{forecast_growth:+.1f}%"),
        ("show_chart", "Projected close", format_money(forecast_value)),
        ("rule", "Model quality", f"MAE {format_money(mae, 2)} / RMSE {format_money(rmse, 2)}"),
        ("tips_and_updates", "Business recommendation", "Increase coverage for this segment." if forecast_growth >= 0 else "Monitor demand risk before expanding inventory."),
    ])

    section_heading("Primary forecast chart", "Historical sales and projected demand")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["yhat"],
        mode="lines", name="Historical Sales",
        line=dict(color="#0f172a", width=2)))
    fig.add_trace(go.Scatter(
        x=fut["ds"], y=fut["yhat"],
        mode="lines+markers", name="Forecasted Sales",
        line=dict(color="#10b981", width=3, dash="dash"),
        marker=dict(size=8, color="#10b981")))
    
    if not fut.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fut["ds"], fut["ds"][::-1]]),
            y=pd.concat([fut["yhat_upper"], fut["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(16, 185, 129, 0.15)",
            line_color="rgba(0,0,0,0)", name="Confidence Interval"))
        fig.add_vline(x=str(fut["ds"].iloc[0]),
                      line_dash="dot", line_color="#94a3b8", opacity=0.5)
        
    chart_style(fig, f"Sales Projection - {selected}")
    show_chart(fig)

    # ── Model Evaluation Metrics ──────────────────────────────────────────────
    section_heading("Model diagnostics", "Evaluation metrics")
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="metric-box">
        <div class="m-label">Mean Absolute Error (MAE)</div>
        <div class="m-value">${mae:,.2f}</div>
    </div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-box">
        <div class="m-label">Root Mean Square Error (RMSE)</div>
        <div class="m-value">${rmse:,.2f}</div>
    </div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-box">
        <div class="m-label">Prediction Horizon</div>
        <div class="m-value">{horizon} Months</div>
    </div>""", unsafe_allow_html=True)

    if not fut.empty:
        with st.expander("Detailed forecast data", expanded=False, icon=":material/table_chart:"):
            tbl = fut[["ds","yhat","yhat_lower","yhat_upper"]].copy()
            tbl.columns = ["Month","Forecast ($)","Lower Bound ($)","Upper Bound ($)"]
            tbl["Month"] = tbl["Month"].dt.strftime("%B %Y")
            for c in ["Forecast ($)","Lower Bound ($)","Upper Bound ($)"]:
                tbl[c] = tbl[c].map("${:,.2f}".format)
            show_table(tbl.reset_index(drop=True))
            st.download_button("Download forecast CSV",
                               fut.to_csv(index=False).encode(),
                               "forecast.csv", "text/csv", icon=":material/download:")


# =============================================================================
# ── PAGE 3: ANOMALY REPORT ────────────────────────────────────────────────────
# =============================================================================

def page_anomaly_report():
    page_intro(
        "Anomaly report",
        "Weekly sales surveillance with Isolation Forest and rolling Z-score detection methods.",
    )

    with st.sidebar:
        st.markdown("<div style='font-size:12px; font-weight:800; color:#047857; text-transform:uppercase; letter-spacing:.08em; margin:14px 0 10px;'>Detection parameters</div>", unsafe_allow_html=True)
        method = st.radio("Algorithm",
                          ["Isolation Forest", "Rolling Z-Score", "Both (Union)"])

    df = get_anomaly_data()

    if method == "Isolation Forest":
        anomalies = df[df["IF_Anomaly"] == -1].copy()
    elif method == "Rolling Z-Score":
        anomalies = df[df["Z_Anomaly"] == True].copy()
    else:
        anomalies = df[(df["IF_Anomaly"] == -1) | (df["Z_Anomaly"] == True)].copy()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    anomaly_rate = len(anomalies) / len(df) * 100 if len(df) else 0
    peak_impact = anomalies['Sales'].max() if not anomalies.empty else 0
    top_type = anomalies["Anomaly Type"].mode().iloc[0] if not anomalies.empty else "No anomaly"
    kpi(c1, "Weeks analysed",   f"{len(df)}", "Weekly cadence", False, "calendar_month", df["Sales"])
    kpi(c2, "Anomalies detected", f"{len(anomalies)}",
        delta=f"{anomaly_rate:.1f}% of dataset", neg=True, icon="warning", spark=df["Sales"])
    kpi(c3, "Peak Anomaly Impact",
        format_money(peak_impact) if not anomalies.empty else "-", "Largest flagged week", bool(not anomalies.empty), "priority_high", anomalies["Sales"] if not anomalies.empty else df["Sales"])

    executive_insights([
        ("warning", "Number of anomalies", f"{len(anomalies)} weeks"),
        ("percent", "Anomaly rate", f"{anomaly_rate:.1f}%"),
        ("bolt", "Largest impact", format_money(peak_impact) if not anomalies.empty else "-"),
        ("category", "Primary anomaly type", top_type),
        ("tips_and_updates", "Business recommendation", "Investigate the flagged high-impact weeks first." if not anomalies.empty else "Maintain the current monitoring cadence."),
    ])

    section_heading("Primary anomaly chart", "Weekly sales with detected anomalies")

    # ── Anomaly chart ─────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Order Date"], y=df["Sales"],
        mode="lines", name="Weekly Sales",
        line=dict(color="#0f172a", width=2),
        fill="tozeroy", fillcolor="rgba(15, 23, 42, 0.05)"))

    if_anom = df[df["IF_Anomaly"] == -1]
    if not if_anom.empty:
        fig.add_trace(go.Scatter(
            x=if_anom["Order Date"], y=if_anom["Sales"],
            mode="markers", name="Isolation Forest",
            marker=dict(color="#ef4444", size=10, symbol="circle",
                        line=dict(width=1.5, color="#b91c1c"))))

    z_anom = df[df["Z_Anomaly"] == True]
    if not z_anom.empty:
        fig.add_trace(go.Scatter(
            x=z_anom["Order Date"], y=z_anom["Sales"],
            mode="markers", name="Rolling Z-Score",
            marker=dict(color="#8b5cf6", size=10, symbol="diamond",
                        line=dict(width=1.5, color="#6d28d9"))))

    chart_style(fig, "Weekly Sales with Detected Anomalies")
    show_chart(fig)

    # ── Anomaly Table ─────────────────────────────────────────────────────────
    section_heading("Detailed tables", "Detected anomalies register")
    if anomalies.empty:
        st.info("No anomalies detected with the selected parameters.", icon=":material/check_circle:")
    else:
        tbl = anomalies[["Order Date","Sales","Anomaly Type","Possible Explanation"]].copy()
        tbl.columns = ["Week","Sales ($)","Anomaly Classification","Business Context"]
        tbl["Status"] = tbl["Anomaly Classification"].map(
            lambda value: ":red-badge[Anomaly]" if value != "Normal" else ":green-badge[Normal]"
        )
        tbl["Week"]      = pd.to_datetime(tbl["Week"]).dt.strftime("%d %b %Y")
        tbl["Sales ($)"] = tbl["Sales ($)"].map("${:,.2f}".format)
        tbl = tbl[["Week", "Status", "Sales ($)", "Anomaly Classification", "Business Context"]]
        tbl = tbl.reset_index(drop=True)
        tbl.index += 1
        show_table(tbl, column_config={"Status": st.column_config.MarkdownColumn("Status")})
        st.download_button("Export anomaly report",
                           anomalies.to_csv(index=False).encode(),
                           "anomalies.csv", "text/csv", icon=":material/download:")

    # ── Method comparison ─────────────────────────────────────────────────────
    section_heading("Additional information", "Algorithm comparison")
    comp = pd.DataFrame({
        "Algorithm": ["Isolation Forest", "Rolling Z-Score"],
        "Anomalies Flagged": [
            (df["IF_Anomaly"] == -1).sum(),
            df["Z_Anomaly"].sum()
        ]
    })
    fig2 = px.bar(comp, x="Algorithm", y="Anomalies Flagged",
                  color="Algorithm", color_discrete_sequence=["#ef4444","#8b5cf6"],
                  text="Anomalies Flagged")
    fig2.update_traces(textposition="outside", showlegend=False, opacity=0.9)
    chart_style(fig2, "Total Detections by Algorithm")
    fig2.update_layout(height=400)
    show_chart(fig2)


# =============================================================================
# ── PAGE 4: PRODUCT DEMAND SEGMENTS ──────────────────────────────────────────
# =============================================================================

def page_demand_segments():
    page_intro(
        "Product demand segments",
        "KMeans demand segmentation across revenue, volatility, growth, and average order value.",
    )

    df = get_cluster_data()

    segment_order = [
        "High Volume, Stable Demand",
        "Growing Demand",
        "Low Volume, High Volatility",
        "Declining Demand",
    ]
    color_map = {
        "High Volume, Stable Demand":  "#0f172a",
        "Growing Demand":              "#10b981",
        "Low Volume, High Volatility": "#f59e0b",
        "Declining Demand":            "#ef4444",
    }

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    top_segment = df.groupby("Demand Segment")["Total Sales"].sum().idxmax()
    top_product = df.sort_values("Total Sales", ascending=False)["Sub-Category"].iloc[0]
    max_growth = df["Growth Rate"].max()
    volatility_leader = df.sort_values("Sales Volatility", ascending=False)["Sub-Category"].iloc[0]
    kpi(c1, "Total sub-categories", f"{len(df)}", "Clustered SKUs", False, "inventory_2", df["Total Sales"])
    kpi(c2, "Demand clusters", "4", top_segment, False, "hub", df.groupby("Demand Segment")["Total Sales"].sum().values)
    kpi(c3, "Peak total sales",
        format_money(df['Total Sales'].max()), top_product, False, "leaderboard", df["Total Sales"])
    kpi(c4, "Max growth rate",
        f"{max_growth:.1f}%", "Fastest mover", max_growth < 0, "rocket_launch", df["Growth Rate"])

    executive_insights([
        ("workspace_premium", "Highest performing category", top_product),
        ("trending_up", "Highest forecast growth", f"{max_growth:.1f}%"),
        ("scatter_plot", "Dominant demand segment", top_segment),
        ("monitoring", "Most volatile product", volatility_leader),
        ("tips_and_updates", "Business recommendation", f"Protect availability for {top_product} while monitoring volatility in {volatility_leader}."),
    ])

    # ── PCA Cluster Chart ─────────────────────────────────────────────────────
    section_heading("Primary segmentation chart", "Demand clusters in PCA space")
    chart_df = df.copy()
    label_positions = {
        "Accessories": "top right",
        "Appliances": "top center",
        "Art": "top right",
        "Binders": "top left",
        "Bookcases": "bottom center",
        "Chairs": "top right",
        "Copiers": "top center",
        "Envelopes": "bottom left",
        "Fasteners": "top left",
        "Furnishings": "top right",
        "Labels": "top center",
        "Machines": "top center",
        "Paper": "top left",
        "Phones": "top left",
        "Storage": "bottom left",
        "Supplies": "bottom center",
        "Tables": "top center",
    }
    chart_df["Label Position"] = chart_df["Sub-Category"].map(label_positions).fillna("top center")
    fig = go.Figure()
    for segment in segment_order:
        temp = chart_df[chart_df["Demand Segment"] == segment].sort_values("Sub-Category")
        if temp.empty:
            continue
        fig.add_trace(go.Scatter(
            x=temp["PC1"],
            y=temp["PC2"],
            mode="markers+text",
            name=segment,
            text=temp["Sub-Category"],
            textposition=temp["Label Position"],
            textfont=dict(size=11, color="#334155", family="Inter, sans-serif"),
            marker=dict(
                size=15,
                color=color_map[segment],
                opacity=0.92,
                line=dict(width=2, color="#ffffff"),
            ),
            customdata=np.stack([
                temp["Total Sales"],
                temp["Growth Rate"],
                temp["Sales Volatility"],
                temp["Average Order Value"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Segment: {segment}<br>"
                "Total sales: $%{customdata[0]:,.0f}<br>"
                "Growth rate: %{customdata[1]:.1f}%<br>"
                "Sales volatility: $%{customdata[2]:,.0f}<br>"
                "Avg order value: $%{customdata[3]:,.2f}"
                "<extra></extra>"
            ),
        ))
    chart_style(fig, "")
    fig.update_layout(
        height=600,
        margin=dict(t=34, b=62, l=24, r=250),
        legend=dict(
            title=dict(text="Demand segment", font=dict(size=12, color="#667085")),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12, color="#475467"),
        ),
    )
    fig.update_xaxes(
        title_text="Principal Component 1",
        zeroline=True,
        zerolinecolor="#cfded2",
        zerolinewidth=1.4,
        gridcolor="#edf4ee",
        range=[chart_df["PC1"].min() - 0.45, chart_df["PC1"].max() + 0.45],
    )
    fig.update_yaxes(
        title_text="Principal Component 2",
        zeroline=True,
        zerolinecolor="#cfded2",
        zerolinewidth=1.4,
        gridcolor="#edf4ee",
        range=[chart_df["PC2"].min() - 0.55, chart_df["PC2"].max() + 0.55],
    )
    show_chart(fig)

    # ── Bar chart & Pie Chart ─────────────────────────────────────────────────
    section_heading("Supporting analytics", "Revenue contribution and segment mix")
    col_a, col_b = st.columns(2)

    with col_a:
        bar_df = df.sort_values("Total Sales", ascending=True).copy()
        fig2 = px.bar(
            bar_df,
            x="Total Sales", y="Sub-Category",
            color="Demand Segment", orientation="h",
            text=bar_df["Total Sales"].map(lambda value: f"${value/1000:.0f}k"),
            color_discrete_map=color_map,
            category_orders={"Demand Segment": segment_order},
            hover_data={
                "Demand Segment": True,
                "Total Sales": ":$,.0f",
                "Growth Rate": ":.1f",
                "Sales Volatility": ":$,.0f",
                "Average Order Value": ":$,.2f",
            },
        )
        chart_style(fig2, "Revenue contribution by sub-category")
        fig2.update_traces(
            textposition="outside",
            textfont=dict(size=11, color="#334155"),
            marker_line_width=0,
            opacity=0.92,
            cliponaxis=False,
        )
        fig2.update_layout(height=520, showlegend=False, margin=dict(t=70, b=44, l=18, r=72))
        fig2.update_xaxes(tickprefix="$", separatethousands=True)
        col_a.plotly_chart(fig2, width="stretch", config=chart_config())

    with col_b:
        seg_totals = (
            df.groupby("Demand Segment")["Total Sales"]
            .sum()
            .reindex(segment_order)
            .dropna()
            .reset_index()
        )
        seg_totals["Share"] = seg_totals["Total Sales"] / seg_totals["Total Sales"].sum()
        fig3 = px.pie(seg_totals, names="Demand Segment", values="Total Sales",
                      color="Demand Segment", color_discrete_map=color_map, hole=0.55)
        fig3.update_traces(
            textinfo="percent",
            textposition="inside",
            insidetextorientation="radial",
            textfont=dict(size=13, color="#ffffff", family="Inter, sans-serif"),
            marker=dict(line=dict(color="#ffffff", width=2)),
            hovertemplate="<b>%{label}</b><br>Revenue: %{value:$,.0f}<br>Share: %{percent}<extra></extra>",
            pull=[0.015] * len(seg_totals),
        )
        chart_style(fig3, " ")
        fig3.update_layout(
            height=360,
            showlegend=False,
            margin=dict(t=20, b=20, l=12, r=12),
            uniformtext_minsize=12,
            uniformtext_mode="hide",
            annotations=[
                dict(
                    text="<b>Revenue<br>share</b>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=15, color="#111827", family="Inter, sans-serif"),
                )
            ],
        )
        col_b.plotly_chart(fig3, width="stretch", config=chart_config())
        legend_items = "".join(
            f"""
            <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; padding:9px 0; border-bottom:1px solid #eef4ee;">
                <div style="display:flex; align-items:center; gap:9px; min-width:0;">
                    <span style="width:10px; height:10px; border-radius:999px; background:{color_map[row['Demand Segment']]}; flex:0 0 auto;"></span>
                    <span style="font-size:12px; font-weight:700; color:#334155; white-space:normal;">{row['Demand Segment']}</span>
                </div>
                <div style="font-size:12px; color:#667085; font-weight:700; white-space:nowrap;">{row['Share']:.1%}</div>
            </div>
            """
            for _, row in seg_totals.iterrows()
        )
        col_b.markdown(
            f"""
            <div style="margin-top:-8px; padding:0 8px 4px;">
                <div style="font-size:11px; font-weight:800; color:#047857; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px;">Segment mix</div>
                {legend_items}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Sub-Category Mapping Table ────────────────────────────────────────────
    section_heading("Detailed tables", "Sub-category mapping matrix")

    for seg in segment_order:
        members = df[df["Demand Segment"] == seg].sort_values("Total Sales", ascending=False)
        if members.empty:
            continue
            
        with st.expander(f"{seg} ({len(members)} items)", expanded=True, icon=":material/table_chart:"):
            tbl = members[[
                "Sub-Category","Total Sales","Growth Rate",
                "Sales Volatility","Average Order Value"
            ]].copy()
            tbl.columns = [
                "Sub-Category","Total Revenue ($)","Growth Rate (%)",
                "Sales Volatility ($)","Avg Order Value ($)"
            ]
            tbl["Total Revenue ($)"]    = tbl["Total Revenue ($)"].map("${:,.0f}".format)
            tbl["Growth Rate (%)"]      = tbl["Growth Rate (%)"].map("{:+.1f}%".format)
            tbl["Sales Volatility ($)"] = tbl["Sales Volatility ($)"].map("${:,.0f}".format)
            tbl["Avg Order Value ($)"]  = tbl["Avg Order Value ($)"].map("${:,.2f}".format)
            tbl["Status"] = members["Growth Rate"].apply(
                lambda value: ":green-badge[Growing]" if value >= 0 else ":red-badge[Declining]"
            ).values
            tbl = tbl[[
                "Sub-Category","Status","Total Revenue ($)","Growth Rate (%)",
                "Sales Volatility ($)","Avg Order Value ($)"
            ]]
            show_table(tbl.reset_index(drop=True), column_config={"Status": st.column_config.MarkdownColumn("Status")})

    st.download_button("Export segment data",
                       df.to_csv(index=False).encode(),
                       "product_segments.csv", "text/csv", icon=":material/download:")


# =============================================================================
# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style='padding:24px 4px 18px;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='width:42px; height:42px; border-radius:14px; display:grid; place-items:center; background:#064e3b; color:#ffffff; box-shadow:0 10px 24px rgba(6,78,59,.22);'>
                <span class='material-symbols-rounded'>monitoring</span>
            </div>
            <div>
                <div style='font-size:18px; font-weight:800; color:#111827; letter-spacing:0;'>Sales Analytics</div>
                <div style='font-size:11px; color:#667085; margin-top:3px; text-transform:uppercase; letter-spacing:0.08em;'>Executive command center</div>
            </div>
        </div>
        <div style='display:inline-flex; align-items:center; gap:6px; margin-top:18px; padding:5px 9px; border-radius:999px; background:#e7f8ef; color:#047857; font-size:11px; font-weight:800;'>
            <span class='material-symbols-rounded' style='font-size:15px;'>verified</span> Production v1.0
        </div>
    </div>
    <hr style='border-color:#dbe5dc; margin:0 0 20px;'>
    """, unsafe_allow_html=True)

    nav_icons = {
        "Sales Overview": ":material/dashboard: Sales overview",
        "Forecast Explorer": ":material/query_stats: Forecast explorer",
        "Anomaly Report": ":material/warning: Anomaly report",
        "Demand Segments": ":material/hub: Demand segments",
    }
    page = st.radio(
        "Navigation",
        ["Sales Overview",
         "Forecast Explorer",
         "Anomaly Report",
         "Demand Segments"],
        format_func=lambda option: nav_icons[option],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#dbe5dc; margin:24px 0 16px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#667085; line-height:1.6; padding:0 4px 12px;'>
        <b style='color:#111827;'>Superstore Analytics System</b><br>
        Built for forecasting, anomaly detection, and demand segmentation.
    </div>
    """, unsafe_allow_html=True)

# ── Route ─────────────────────────────────────────────────────────────────────
if   page == "Sales Overview":    page_sales_overview()
elif page == "Forecast Explorer": page_forecast_explorer()
elif page == "Anomaly Report":    page_anomaly_report()
else:                             page_demand_segments()
