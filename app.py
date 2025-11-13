import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components


# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(page_title="Horizon Media | Sustainability Intelligence Dashboard", layout="wide")


st.markdown(
    """
<style>
/* --- Global Page Styling --- */
:root {
    --bg: #000000;
    --card: #1a1a1a;
    --text-title: #ffffff;
    --text-body: #e0e0e0;
    --button-bg: #3a3a3a;
    --button-hover: #4a4a4a;
    --button-text: #f5f5f5;
    --card-border: #333333;
    --accent: #3b82f6;
    --em: #f97316;
    --muted: #9ca3af;
}


html, body, .stApp, .block-container, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text-body) !important;
}


[data-testid="stDecoration"] {
    display: none !important;
}


header[data-testid="stHeader"] {
    background-color: var(--bg) !important;
    border: none !important;
    box-shadow: none !important;
}


.block-container {
    padding-top: 0rem !important;
    margin-top: 0 !important;
}


.main > div:first-child {
    padding-top: 0 !important;
}


h1, h2, h3, h4, h5, h6 {
    color: var(--text-title) !important;
}


p, span, div, label, li {
    color: var(--text-body) !important;
}


.title {
    font-size: 42px;
    font-weight: 800;
    margin: 10px 0 30px 0;
    padding-bottom: 10px;
    color: var(--text-title) !important;
}


.stButton > button,
button[kind="secondary"],
button[kind="primary"],
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
div[data-testid="stFormSubmitButton"] > button {
    background-color: var(--button-bg) !important;
    color: var(--button-text) !important;
    border: 1px solid #4a4a4a !important;
    border-radius: 8px !important;
    transition: all 0.2s ease-in-out;
    height: 42px !important;
}


.stButton > button:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="baseButton-primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: var(--button-hover) !important;
    border-color: #5a5a5a !important;
}


.stSelectbox > div > div,
.stMultiSelect > div > div,
div[data-baseweb="select"] > div {
    background-color: var(--button-bg) !important;
    color: var(--button-text) !important;
    border: 1px solid #4a4a4a !important;
    border-radius: 8px !important;
}


.stSelectbox label,
.stMultiSelect label {
    color: var(--text-body) !important;
}


.stMultiSelect span,
.stSelectbox span,
.stMultiSelect input,
.stSelectbox input {
    color: var(--button-text) !important;
}


[data-baseweb="popover"],
[data-baseweb="popover"] * {
    background-color: var(--button-bg) !important;
}


[data-baseweb="select"] > div,
[data-baseweb="menu"] {
    background-color: var(--button-bg) !important;
}


[data-baseweb="menu"] li {
    color: var(--button-text) !important;
    background-color: var(--button-bg) !important;
}


[data-baseweb="menu"] li:hover {
    background-color: var(--button-hover) !important;
}


[data-testid="stFileUploader"] {
    margin-top: 20px !important;
}


[data-testid="stFileUploader"] > div {
    background-color: transparent !important;
}


[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploader"] section {
    background-color: #2a2a2a !important;
    border: 2px dashed #505050 !important;
    border-radius: 8px !important;
}


[data-testid="stFileUploadDropzone"] *,
[data-testid="stFileUploader"] section * {
    color: var(--text-body) !important;
}


[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button {
    background-color: var(--button-bg) !important;
    color: var(--button-text) !important;
    border: 1px solid #4a4a4a !important;
}


[data-testid="stFileUploader"] label {
    color: var(--text-body) !important;
}


.filters {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: nowrap;
    margin-bottom: 16px;
}


.kpi-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 8px;
}


.kpi-row:first-of-type {
    margin-bottom: 8px;
}


.kpi-row:last-of-type {
    margin-bottom: 30px;
}


.kpi-card {
    background: var(--card);
    border-radius: 12px;
    padding: 10px 12px;
    border: 1px solid var(--card-border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    min-width: 120px;
    max-width: 100%;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}


.kpi-card:hover {
    z-index: 100;
    transform: scale(1.05);
}


.kpi-card-emissions {
    border-left: 4px solid var(--em);
}


.kpi-card-emissions:hover {
    box-shadow: 0 8px 24px rgba(249,115,22,0.4);
}


.kpi-card-biz {
  border-left: 4px solid var(--accent);
}


.kpi-card-biz:hover {
  box-shadow: 0 8px 24px rgba(59,130,246,0.4);
}


.kpi-info-button {
    position: absolute;
    top: 5px;
    right: 5px;
    color: var(--muted);
    font-size: 10px;
    cursor: help;
    padding: 2px;
    border-radius: 50%;
    transition: color 0.2s;
    user-select: none;
}

.kpi-info-button:hover {
    color: var(--accent);
}


.kpi-info-button .tooltip-text {
    visibility: hidden;
    width: 150px;
    background-color: #333;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 101;
    top: 20px;
    right: 0px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 12px;
    line-height: 1.3;
    white-space: normal;
    box-shadow: 0 2px 10px rgba(0,0,0,0.5);
}


.kpi-info-button:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}


.kpi-label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}


.kpi-value {
    color: var(--text-title) !important;
    font-size: 22px;
    font-weight: 800;
    margin-top: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.3s ease;
}


.kpi-card:hover .kpi-value,
.kpi-card:hover .kpi-label {
    white-space: normal;
    overflow: visible;
    word-break: break-word;
}


.kpi-card:hover .kpi-value {
    font-size: 20px;
}


.chart-card {
    background: var(--card);
    border-radius: 10px;
    padding: 10px;
    border: 1px solid var(--card-border);
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    margin-bottom: 20px;
}


.reco-block {
    background: var(--card);
    border-radius: 12px;
    padding: 18px;
    border: 1px solid var(--card-border);
    margin-top: 0px;
    margin-bottom: 30px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}


.reco-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
    border: 1px solid var(--card-border);
}

.reco-table th, .reco-table td {
    border: 1px solid var(--card-border);
    padding: 12px 16px;
    vertical-align: top;
    text-align: left;
    width: 33.33%;
}

.reco-table th {
    background-color: #2a2a2a;
    color: var(--text-title);
    font-size: 16px;
}

.reco-table td {
    background-color: var(--card);
    min-height: 200px;
}

.reco-table ul {
    margin: 0;
    padding-left: 20px;
}

.reco-table li {
    margin-bottom: 12px;
    color: var(--text-body) !important;
    line-height: 1.4;
    list-style-type: disc;
}

.reco-table ul {
    list-style-type: disc;
    color: var(--text-body);
}

.reco-table li > span {
    color: inherit;
}

.small-muted {
    color: var(--muted);
    font-size: 13px;
}

.roas-good { color: #4ade80 !important; }
.roas-bad { color: #f87171 !important; }
.roas-neutral { color: #fbbf24 !important; }

.emiss-good { color: #4ade80 !important; }
.emiss-bad { color: #f87171 !important; }
.emiss-neutral { color: #3b82f6 !important; }

.reco-block + div,
.element-container:has(+ .element-container > div > .reco-block) {
    display: none !important;
}

hr {
    display: none !important;
}

[data-testid="stDataFrame"],
[data-testid="stDataFrame"] * {
    color: var(--text-body) !important;
}

.dataframe {
    color: var(--text-body) !important;
}

.stAlert {
    display: none !important;
}

.title {
    margin-top: 50px !important;
    text-align: center !important;
}
</style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="title">Horizon Media | Sustainability Intelligence Dashboard</div>', unsafe_allow_html=True)


# -------------------------
# Helpers
# -------------------------
def safe_div(a, b):
    if isinstance(a, (pd.Series, np.ndarray)) or isinstance(b, (pd.Series, np.ndarray)):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where((b_arr != 0) & (~pd.isna(b_arr)), a_arr / b_arr, 0.0)
        return res
    try:
        return a / b if b not in (0, None, np.nan) else 0.0
    except Exception:
        return 0.0


def fmt_money(n):
    if pd.isna(n):
        return "$0"
    n = float(n)
    if abs(n) >= 1e9: return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6: return f"${n/1e6:.2f}M"
    if abs(n) >= 1e3: return f"${n/1e3:.1f}K"
    return f"${n:,.0f}"


def fmt_count(n):
    if pd.isna(n):
        return "0"
    n = float(n)
    if abs(n) >= 1e6: return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3: return f"{n/1e3:.1f}K"
    return f"{int(n):,}" if float(n).is_integer() else f"{n:.2f}"


BLUE_PALETTE = ["#c7d2fe","#93c5fd","#60a5fa","#3b82f6","#2563eb","#1d4ed8","#1e40af","#1e3a8a"]
EMISSIONS_COLOR = "#f97316"
EMISSIONS_UNIT_LABEL = "g CO₂e"


def blue_gradient(n):
    seq = BLUE_PALETTE
    if n <= 0:
        return []
    if n <= len(seq):
        step = (len(seq)-1)/max(1,(n-1))
        idxs = [int(round(i*step)) for i in range(n)]
        return [seq[i] for i in idxs]
    return [seq[i % len(seq)] for i in range(n)]


def display_emission(val_g):
    if pd.isna(val_g):
        return f"0 {EMISSIONS_UNIT_LABEL}"
    return f"{float(val_g):,.2f} {EMISSIONS_UNIT_LABEL}"


def display_emission_1000(val_g_1000):
    if pd.isna(val_g_1000):
        return f"0.00 {EMISSIONS_UNIT_LABEL}"
    return f"{float(val_g_1000):,.2f} {EMISSIONS_UNIT_LABEL}"


# -------------------------
# Upload CSV
# -------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to populate the dashboard.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.upper()


# -------------------------
# Cascading Filters & Reset Logic
# -------------------------
filter_order = [
    ("Advertiser ID", "ADVERTISER_ID"),
    ("Campaign ID", "CAMPAIGN_ID"),
    ("Inventory ID", "INVENTORY_ID"),
    ("Industry", "INDUSTRY"),
    ("Device Type", "DEVICE_TYPE"),
    ("Ad Format", "AD_FORMAT"),
    ("Region", "REGION"),
]

for _, col in filter_order:
    key = f"f_{col}"
    if key not in st.session_state:
        st.session_state[key] = []

if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False

def _reset_filters():
    st.session_state.reset_trigger = True

st.markdown('<div class="filters">', unsafe_allow_html=True)
cols = st.columns([1, 1, 1, 1, 1, 1, 1, 0.85], gap="small")

current_df = df.copy()

for (label, col), col_idx in zip(filter_order, range(7)):
    with cols[col_idx]:
        allowed_opts = sorted(current_df[col].dropna().unique()) if col in current_df.columns else []
        sel_key = f"f_{col}"
        existing = st.session_state.get(sel_key, [])
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        safe_default = [v for v in existing if v in allowed_opts]
        st.multiselect(label, options=allowed_opts, default=safe_default, key=sel_key)
        selected = st.session_state.get(sel_key, [])
        if selected and col in current_df.columns:
            current_df = current_df[current_df[col].isin(selected)]

with cols[7]:
    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
    st.button("Reset", key="reset_filters_btn", on_click=_reset_filters, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.reset_trigger:
    for _, c in filter_order:
        key = f"f_{c}"
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.reset_trigger = False
    st.rerun()

filtered = df.copy()
for _, col in filter_order:
    sel = st.session_state.get(f"f_{col}", [])
    if sel and col in filtered.columns:
        filtered = filtered[filtered[col].isin(sel)]

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# -------------------------
# Aggregations & KPIs
# -------------------------
@st.cache_data(show_spinner=False)
def compute_totals(df_local):
    return {
        "TOTAL_IMPRESSIONS": df_local["TOTAL_IMPRESSIONS"].sum() if "TOTAL_IMPRESSIONS" in df_local.columns else 0,
        "TOTAL_CONVERSIONS": df_local["TOTAL_CONVERSIONS"].sum() if "TOTAL_CONVERSIONS" in df_local.columns else 0,
        "TOTAL_CLICKS": df_local["TOTAL_CLICKS"].sum() if "TOTAL_CLICKS" in df_local.columns else 0,
        "REVENUE": df_local["REVENUE"].sum() if "REVENUE" in df_local.columns else 0,
        "SPEND": df_local["SPEND"].sum() if "SPEND" in df_local.columns else 0,
        "TOTAL_EMISSIONS_GRAMS": df_local["TOTAL_EMISSIONS_GRAMS"].sum() if "TOTAL_EMISSIONS_GRAMS" in df_local.columns else 0,
        "UNIQUE_CAMPAIGNS": df_local["CAMPAIGN_ID"].nunique() if "CAMPAIGN_ID" in df_local.columns else 0,
    }

totals = compute_totals(filtered)
total_impressions = totals["TOTAL_IMPRESSIONS"]
total_conversions = totals["TOTAL_CONVERSIONS"]
total_clicks = totals["TOTAL_CLICKS"]
total_revenue = totals["REVENUE"]
total_spend = totals["SPEND"]
total_emissions_g = totals["TOTAL_EMISSIONS_GRAMS"]
total_campaigns = totals["UNIQUE_CAMPAIGNS"]

emissions_per_1000_impr_g = safe_div(total_emissions_g, total_impressions) * 1000.0
emissions_per_1000_click_g = safe_div(total_emissions_g, total_clicks) * 1000.0
emissions_per_1000_conv_g = safe_div(total_emissions_g, total_conversions) * 1000.0
emissions_per_1000_spent_g = safe_div(total_emissions_g, total_spend) * 1000.0
avg_emission_per_campaign_g = safe_div(total_emissions_g, total_campaigns)

avg_cpc = safe_div(total_spend, total_clicks)
avg_cpm = safe_div(total_spend, total_impressions) * 1000.0
engagement_rate = safe_div(total_clicks, total_impressions) * 100.0
conversion_rate = safe_div(total_conversions, total_impressions) * 100.0
impr_per_1000_dollars = safe_div(total_impressions, total_spend) * 1000.0

row1 = [
    ("Emissions / 1k Impr", display_emission_1000(emissions_per_1000_impr_g), "emissions_kpi"),
    ("Emissions / 1k Click", display_emission_1000(emissions_per_1000_click_g), "emissions_kpi"),
    ("Emissions / 1k Conv", display_emission_1000(emissions_per_1000_conv_g), "emissions_kpi"),
    ("Conversion Rate", f"{conversion_rate:.2f}%", "biz_kpi"),
    ("Engagement Rate", f"{engagement_rate:.2f}%", "biz_kpi"),
    ("Impr. / $1k Spent", fmt_count(impr_per_1000_dollars), "biz_kpi"),
]

row2 = [
    ("Total Emissions", display_emission(total_emissions_g), "emissions_kpi"),
    ("Emissions / $1k Spent", display_emission_1000(emissions_per_1000_spent_g), "emissions_kpi"),
    ("Emissions / Campaign", display_emission(avg_emission_per_campaign_g), "emissions_kpi"),
    ("Total Spend", fmt_money(total_spend), "biz_kpi"),
    ("CPC", fmt_money(avg_cpc), "biz_kpi"),
    ("CPM", fmt_money(avg_cpm), "biz_kpi"),
]

def render_kpi_row(items):
    tooltips = {
        "CPC": "Cost Per Click",
        "CPM": "Cost Per Mille (1,000 Impressions)",
    }

    cols_kpi = st.columns(len(items), gap="small")
    for c, (label, value, kpi_type) in zip(cols_kpi, items):
        if kpi_type == "emissions_kpi":
            card_class = "kpi-card kpi-card-emissions"
        elif kpi_type == "biz_kpi":
            card_class = "kpi-card kpi-card-biz"
        else:
            card_class = "kpi-card"

        info_button_html = ""
        if label in tooltips:
            tooltip_text = tooltips[label]
            info_button_html = f"<div class='kpi-info-button'>ⓘ<span class='tooltip-text'>{tooltip_text}</span></div>"

        c.markdown(f"<div class='{card_class}' title='{label}'>{info_button_html}<div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>", unsafe_allow_html=True)

render_kpi_row(row1)
st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
render_kpi_row(row2)

st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)


# -------------------------
# Chart Utilities
# -------------------------
def ordered_cats_by_value(df_local, x_col, value_col, top_n=12):
    if x_col not in df_local.columns or value_col not in df_local.columns:
        return []
    return df_local.groupby(x_col)[value_col].sum().sort_values(ascending=False).head(top_n).index.tolist()


def stacked_with_emissions(df_local, x_col, stack_col, value_col, emissions_col, title, top_n=12):
    if x_col not in df_local.columns or value_col not in df_local.columns:
        return None

    tmp = df_local.copy()
    is_datetime_like = False
    try:
        if np.issubdtype(tmp[x_col].dtype, np.datetime64):
            tmp[x_col] = pd.to_datetime(tmp[x_col], errors="coerce")
            is_datetime_like = True
        else:
            coerced = pd.to_datetime(tmp[x_col], errors="coerce")
            if coerced.notna().any():
                tmp[x_col] = coerced
                is_datetime_like = True
    except Exception:
        is_datetime_like = False

    if is_datetime_like:
        order_index = tmp.groupby(tmp[x_col])[value_col].sum().sort_index().index.tolist()
        cats = order_index[-top_n:]
        cats = sorted(cats)
        x_labels = [pd.to_datetime(c).strftime("%Y-%m-%d") for c in cats]
    else:
        cats = tmp.groupby(x_col)[value_col].sum().sort_values(ascending=False).head(top_n).index.tolist()
        x_labels = [str(c) for c in cats]

    if not cats:
        return None

    d = tmp[tmp[x_col].isin(cats)].copy()
    groups = sorted(d[stack_col].dropna().unique()) if stack_col in d.columns else []
    colors = blue_gradient(len(groups) if groups else 1)

    fig = go.Figure()

    for i, g in enumerate(groups):
        ys = []
        for c in cats:
            mask = (d[x_col] == c) & (d[stack_col] == g)
            ys.append(d.loc[mask, value_col].sum())
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=ys,
                name=str(g),
                marker=dict(color=colors[i] if i < len(colors) else None),
                hovertemplate="%{y:,.0f}<br>%{x}<extra></extra>",
            )
        )

    emiss_raw = []
    for c in cats:
        emiss_raw.append(d.loc[d[x_col] == c, emissions_col].sum())
    emiss_vals = [float(v) for v in emiss_raw]

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=emiss_vals,
            name=f"Emissions ({EMISSIONS_UNIT_LABEL})",
            mode="lines+markers",
            line=dict(color=EMISSIONS_COLOR, width=2, shape="spline"),
            marker=dict(size=6),
            connectgaps=True,
            yaxis="y2",
            hovertemplate="%{y:.3f} " + EMISSIONS_UNIT_LABEL + "<br>%{x}<extra></extra>",
        )
    )

    numeric_emiss = [v for v in emiss_vals if v is not None]
    min_e = float(min(numeric_emiss)) if numeric_emiss else 0.0
    max_e = float(max(numeric_emiss)) if numeric_emiss else 0.0
    y2_range = [max(0, min_e * 0.9), max_e * 1.1] if max_e > 0 else None

    fig.update_layout(
        title=title or "",
        title_font=dict(color="#ffffff"),
        barmode="stack",
        hovermode="x unified",
        xaxis=dict(tickangle=-30, automargin=True, tickfont=dict(size=10, color="#e0e0e0")),
        yaxis=dict(title=dict(text=value_col, font=dict(color="#e0e0e0")), automargin=True, tickfont=dict(color="#e0e0e0")),
        yaxis2=dict(
            title=dict(text=f"Emissions ({EMISSIONS_UNIT_LABEL})", font=dict(color="#e0e0e0")),
            overlaying="y",
            side="right",
            showgrid=False,
            title_standoff=80,
            tickfont=dict(color="#e0e0e0"),
        ),
        margin=dict(l=64, r=200, t=60, b=100),
        legend=dict(orientation="h", y=-0.35, xanchor="center", x=0.5, font=dict(color="#e0e0e0")),
        height=520,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,26,0.5)',
        font=dict(color='#e0e0e0')
    )
    if y2_range:
        fig.layout.yaxis2.update(range=y2_range)

    return fig


def bar_with_emissions(df_local, x_col, bar_col, emissions_col, title, top_n=12):
    if x_col not in df_local.columns or bar_col not in df_local.columns:
        return None

    cats = ordered_cats_by_value(df_local, x_col, bar_col, top_n)
    if not cats:
        return None

    d = df_local[df_local[x_col].isin(cats)].copy()
    bar_vals = [d[d[x_col] == c][bar_col].sum() for c in cats]
    emiss_vals = [float(d[d[x_col] == c][emissions_col].sum()) for c in cats]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=cats, y=bar_vals, name=bar_col, marker_color=BLUE_PALETTE[3], hovertemplate="%{y:,.0f}<extra></extra>")
    )
    fig.add_trace(
        go.Scatter(
            x=cats,
            y=emiss_vals,
            name=f"Emissions ({EMISSIONS_UNIT_LABEL})",
            mode="lines+markers",
            line=dict(color=EMISSIONS_COLOR, width=2),
            marker=dict(size=6),
            yaxis="y2",
            hovertemplate="%{y:.3f} " + EMISSIONS_UNIT_LABEL + "<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or "",
        title_font=dict(color="#ffffff"),
        yaxis=dict(title=dict(text=bar_col, font=dict(color="#e0e0e0")), automargin=True, tickfont=dict(color="#e0e0e0")),
        yaxis2=dict(
            title=dict(text=f"Emissions ({EMISSIONS_UNIT_LABEL})", font=dict(color="#e0e0e0")),
            overlaying="y",
            side="right",
            showgrid=False,
            title_standoff=80,
            tickfont=dict(color="#e0e0e0"),
        ),
        margin=dict(l=64, r=200, t=60, b=100),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.35, xanchor="center", x=0.5, font=dict(color="#e0e0e0")),
        xaxis=dict(tickangle=-30, automargin=True, tickfont=dict(size=10, color="#e0e0e0")),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,26,0.5)",
        font=dict(color="#e0e0e0")
    )
    return fig


def pie_blue(df_local, name_col, value_col, top_n=12, title=None):
    if isinstance(top_n, str) and title is None:
        title = top_n
        top_n = 12
    if name_col not in df_local.columns or value_col not in df_local.columns:
        return None

    agg = df_local.groupby(name_col)[value_col].sum().reset_index()
    agg = agg.sort_values(value_col, ascending=False).head(top_n)
    if agg.empty:
        return None

    palette = blue_gradient(len(agg))
    fig = px.pie(agg, names=name_col, values=value_col, color_discrete_sequence=palette, title=title, hole=0.4)
    fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,.0f}<extra></extra>")
    fig.update_layout(
        margin=dict(l=60, r=140, t=40, b=40),
        height=520,
        width=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,26,0.5)',
        font=dict(color='#e0e0e0'),
        title_font=dict(color="#ffffff"),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, font=dict(color="#e0e0e0"))
    )
    return fig


def two_per_row(fig1, fig2):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if fig1 is not None:
            st.plotly_chart(fig1, use_container_width=True)
    with c2:
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<div style='margin-bottom:30px;'></div>", unsafe_allow_html=True)


# -------------------------
# Chart Definitions
# -------------------------
chart1  = stacked_with_emissions(filtered,"INVENTORY_ID","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Inventory Performance: Impressions vs. Emissions")
chart2  = stacked_with_emissions(filtered,"INDUSTRY","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Industry Performance: Impressions vs. Emissions")
chart3  = stacked_with_emissions(filtered,"REGION","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Regional Performance: Impressions vs. Emissions")
chart4  = stacked_with_emissions(filtered,"DEVICE_TYPE","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Device Performance: Impressions vs. Emissions")
if "DATE" in filtered.columns:
    tmp = filtered.copy()
    tmp["DATE"] = pd.to_datetime(tmp["DATE"], errors="coerce")
    tmp = tmp.dropna(subset=["DATE"])
    tmp["WEEK"] = tmp["DATE"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    chart5 = stacked_with_emissions(tmp,"WEEK","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Weekly Trend: Impressions vs. Emissions")
else:
    chart5 = None
chart6  = pie_blue(filtered,"INDUSTRY","TOTAL_EMISSIONS_GRAMS","Total Emissions Breakdown by Industry")
chart7  = bar_with_emissions(filtered,"REGION","TOTAL_CONVERSIONS","TOTAL_EMISSIONS_GRAMS","Regional Conversions vs. Emissions")
chart8  = bar_with_emissions(filtered,"DEVICE_TYPE","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Device Impressions vs. Emissions")
chart9  = stacked_with_emissions(filtered,"INDUSTRY","DEVICE_TYPE","TOTAL_CONVERSIONS","TOTAL_EMISSIONS_GRAMS","Industry Conversions vs. Emissions")
chart10 = bar_with_emissions(filtered,"ADVERTISER_ID","SPEND","TOTAL_EMISSIONS_GRAMS","Advertiser Spend vs. Emissions")
chart11 = pie_blue(filtered,"REGION","SPEND","Total Spend Distribution by Region")
chart12 = stacked_with_emissions(filtered,"ADVERTISER_ID","AD_FORMAT","SPEND","TOTAL_EMISSIONS_GRAMS","Advertiser Spend vs. Emissions")
chart13 = stacked_with_emissions(filtered,"AD_FORMAT","DEVICE_TYPE","TOTAL_CLICKS","TOTAL_EMISSIONS_GRAMS","Ad Format Clicks vs. Emissions")
chart14 = pie_blue(filtered,"DEVICE_TYPE","TOTAL_CONVERSIONS","Total Conversions Distribution by Device")
chart15 = stacked_with_emissions(filtered,"AD_FORMAT","DEVICE_TYPE","TOTAL_CONVERSIONS","TOTAL_EMISSIONS_GRAMS","Ad Format Conversions vs. Emissions")


# -------------------------
# Chart Layout
# -------------------------
two_per_row(chart1, chart2)
two_per_row(chart3, chart4)
two_per_row(chart5, chart12)
two_per_row(chart7, chart8)
two_per_row(chart9, chart13)
two_per_row(chart10, chart15)
two_per_row(chart6, chart11)
two_per_row(chart14, None)


# -------------------------
# Recommendations Block
# -------------------------
def format_list_as_html_table_cell(list_items):
    html = "<ul>"
    for item in list_items:
        html += f"<li>{item}</li>"
    html += "</ul>"
    return html


def recommendations_block(df_local):
    if df_local.empty or not {"SPEND","REVENUE","TOTAL_EMISSIONS_GRAMS"}.issubset(df_local.columns):
        st.warning("Not enough data for recommendations.")
        return

    d = df_local.copy()
    d["ROAS"] = safe_div(d["REVENUE"], d["SPEND"])
    d["EMISSIONS_G_PER_$"] = safe_div(d["TOTAL_EMISSIONS_GRAMS"], d["SPEND"])
    d["EMISSIONS_G_PER_$1K"] = d["EMISSIONS_G_PER_$"] * 1000.0
    d["CONV_RATE"] = safe_div(d.get("TOTAL_CONVERSIONS", 0), d.get("TOTAL_IMPRESSIONS", 0)) * 100.0

    med_roas = d["ROAS"].median()
    med_emiss_g_per_dollar = d["EMISSIONS_G_PER_$"].median()

    ROAS_SCALE_THRESHOLD = max(2.5, med_roas * 1.1)
    ROAS_REDUCE_THRESHOLD = 1.5
    EMISSIONS_SCALE_THRESHOLD = med_emiss_g_per_dollar * 0.75
    EMISSIONS_REDUCE_THRESHOLD = med_emiss_g_per_dollar * 1.25

    scale_df = d[
        (d["ROAS"] >= ROAS_SCALE_THRESHOLD) &
        (d["EMISSIONS_G_PER_$"] <= EMISSIONS_SCALE_THRESHOLD)
    ]

    reduce_df = d[
        (d["ROAS"] <= ROAS_REDUCE_THRESHOLD) &
        (d["EMISSIONS_G_PER_$"] >= EMISSIONS_REDUCE_THRESHOLD)
    ]

    all_filtered_indices = scale_df.index.union(reduce_df.index)
    optimize_df = d[~d.index.isin(all_filtered_indices)]

    def format_list(sub, category):
        if sub.empty:
            return ["No candidates found for this category."]
        out = []
        key_cols = ["CAMPAIGN_ID", "AD_FORMAT", "DEVICE_TYPE", "REGION", "INVENTORY_ID"]

        for _, r in sub.sort_values("SPEND", ascending=False).head(6).iterrows():
            seg_id = next((str(r[c]) for c in key_cols if c in r and pd.notna(r[c])), "General Segment")

            roas = r.get('ROAS', 0)
            emissions_1k = r.get('EMISSIONS_G_PER_$1K', 0)
            conv_rate = r.get('CONV_RATE', 0)

            if roas >= 2.5:
                roas_class = "roas-good"
            elif roas <= 1.5:
                roas_class = "roas-bad"
            else:
                roas_class = "roas-neutral"

            if emissions_1k <= med_emiss_g_per_dollar * 0.75:
                emiss_class = "emiss-good"
            elif emissions_1k >= med_emiss_g_per_dollar * 1.25:
                emiss_class = "emiss-bad"
            else:
                emiss_class = "emiss-neutral"

            if category == "SCALE":
                roas_class = "roas-good"
                emiss_class = "emiss-good"
            elif category == "REDUCE":
                roas_class = "roas-bad"
                emiss_class = "emiss-bad"

            line1 = (
                f"<span class='{roas_class}'>ROAS {roas:.2f}x</span> "
                f"&nbsp;<span style='color:#9ca3af;'>&middot;</span>&nbsp; "
                f"<span class='{emiss_class}'>{emissions_1k:,.0f} g / $1k</span>"
            )

            line2 = f"<span style='font-size:12px; color: #9ca3af;'>{seg_id} (Conv {conv_rate:.2f}%)</span>"
            out.append(f"{line1}<br>{line2}")

        return out

    s_list = format_list(scale_df, "SCALE")
    r_list = format_list(reduce_df, "REDUCE")
    o_list = format_list(optimize_df, "OPTIMIZE")

    s_html = format_list_as_html_table_cell(s_list)
    r_html = format_list_as_html_table_cell(r_list)
    o_html = format_list_as_html_table_cell(o_list)

    table_html = f"""
    <div class='reco-block'>
        <h3>🎯 Next Best Placement Recommendations</h3>
        <p class='small-muted'>Thresholds are dynamically set based on the filtered dataset's median ROAS and Emissions/$1k.</p>
        <table class="reco-table">
            <thead>
                <tr>
                    <th>🟢 SCALE / INVEST (High ROAS, Low Carbon Intensity)</th>
                    <th>🔴 REDUCE / AVOID (Low ROAS, High Carbon Intensity)</th>
                    <th>🟡 OPTIMIZE / REVIEW (Trade-Offs or Moderate)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>{s_html}</td>
                    <td>{r_html}</td>
                    <td>{o_html}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


recommendations_block(filtered)


# -------------------------
# Final Layout - DATA PREVIEW TABLE WITH IMPROVED ALIGNMENT & SMALLER INDEX
# -------------------------
st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
st.markdown("### 📈 Filtered Data Preview", unsafe_allow_html=True)

table_html = filtered.head(200).to_html(classes="custom-table", index=True, border=0, justify="center")

# Dark-themed HTML with BETTER SPACING and HORIZONTAL SCROLL
html = f"""
<style>
.custom-table-container {{
  background: #2a2a2a;
  padding: 18px;
  border-radius: 10px;
  overflow-x: auto;
  overflow-y: auto;
  max-height: 600px;
  border: 1px solid #3a3a3a;
}}
.custom-table {{
  width: 100%;
  border-collapse: collapse;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  min-width: 1200px;
}}
.custom-table thead th {{
  background: #3a3a3a;
  color: #ffffff;
  font-weight: 700;
  padding: 14px 16px;
  text-align: center;
  position: sticky;
  top: 0;
  z-index: 2;
  border: 1px solid #444444;
  white-space: nowrap;
  min-width: 100px;
}}
/* First column (serial/index) - SMALLER */
.custom-table thead th:first-child {{
  min-width: 50px;
  padding: 14px 8px;
}}
.custom-table tbody td:first-child {{
  min-width: 50px;
  padding: 12px 8px;
  font-size: 12px;
  font-weight: 600;
}}
.custom-table tbody th {{
  background: #3a3a3a;
  color: #cfcfcf;
  font-weight: 600;
  padding: 12px 8px;
  text-align: center;
  border: 1px solid #444444;
  min-width: 50px;
  font-size: 12px;
}}
.custom-table tbody td {{
  background: #2a2a2a;
  color: #ffffff;
  padding: 12px 16px;
  border: 1px solid #3a3a3a;
  vertical-align: middle;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.custom-table tbody tr:nth-child(even) td {{
  background: #313131;
}}
.custom-table tbody tr:hover td {{
  background: #3a3a3a;
  cursor: pointer;
}}
.wrap {{
  width: 100%;
}}
.custom-table-container::-webkit-scrollbar {{
  height: 10px;
  width: 10px;
}}
.custom-table-container::-webkit-scrollbar-track {{
  background: #1a1a1a;
  border-radius: 10px;
}}
.custom-table-container::-webkit-scrollbar-thumb {{
  background: #444444;
  border-radius: 5px;
}}
.custom-table-container::-webkit-scrollbar-thumb:hover {{
  background: #555555;
}}
</style>

<div class="wrap">
  <div class="custom-table-container">
    {table_html}
  </div>
</div>
"""

components.html(html, height=650, scrolling=True)