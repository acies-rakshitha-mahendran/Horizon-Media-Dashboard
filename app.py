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

/* Apply globally - Force dark background everywhere */
html, body, .stApp, .block-container, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text-body) !important;
}

/* Remove white line above title */
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

/* --- Headings --- */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-title) !important;
}

/* --- Global Text --- */
p, span, div, label, li {
    color: var(--text-body) !important;
}

/* --- Title --- */
.title {
    font-size: 42px;
    font-weight: 800;
    margin: 10px 0 30px 0;
    padding-bottom: 10px;
    color: var(--text-title) !important;
}

/* --- Buttons (All types) --- */
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

/* --- Filters, Dropdowns, and Multiselect --- */
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

/* Dropdown options menu */
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

/* --- File Uploader --- */
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

/* --- Filters Container --- */
.filters {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 16px;
}

/* --- KPI Cards --- */
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
    box-shadow: 0 8px 24px rgba(59,130,246,0.4);
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

/* --- Chart Cards --- */
.chart-card {
    background: var(--card);
    border-radius: 10px;
    padding: 10px;
    border: 1px solid var(--card-border);
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    margin-bottom: 20px;
}

/* --- Recommendation Block --- */
.reco-block {
    background: var(--card);
    border-radius: 12px;
    padding: 18px;
    border: 1px solid var(--card-border);
    margin-top: 0px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}

/* Remove box below recommendations */
.reco-block + div,
.element-container:has(+ .element-container > div > .reco-block) {
    display: none !important;
}

/* Remove thin line below recommendations */
hr {
    display: none !important;
}

/* --- Dataframe --- */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] * {
    color: var(--text-body) !important;
}

.dataframe {
    color: var(--text-body) !important;
}

/* --- Misc --- */
.small-muted {
    color: var(--muted);
    font-size: 13px;
}

/* Hide widget warnings */
.stAlert {
    display: none !important;
}
/* ------------------------- */
/* TITLE POSITION FIX        */
/* ------------------------- */
.title {
    margin-top: 50px !important;  /* move title down */
    text-align: center !important; /* center align title */
}

/* ------------------------- */
/* DATAFRAME COLOR FIX       */
/* ------------------------- */
[data-testid="stDataFrame"] table {
    background-color: #ffffff !important;  /* white background */
    color: #333333 !important;             /* dark grey text */
    border-radius: 10px !important;
}

[data-testid="stDataFrame"] thead tr th {
    background-color: #f1f1f1 !important;  /* light grey header */
    color: #000000 !important;
    font-weight: 600 !important;
    text-align: center !important;
}

[data-testid="stDataFrame"] tbody tr td {
    background-color: #ffffff !important;
    color: #333333 !important;
    text-align: center !important;
}

[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
    background-color: #f8f8f8 !important;  /* alternate row striping */
}
/* ------------------------- */
/* FILTERED DATA PREVIEW FIX */
/* ------------------------- */
[data-testid="stDataFrame"] {
    background-color: #2a2a2a !important;  /* dark grey container */
    border-radius: 10px !important;
}

/* Table background and text color */
[data-testid="stDataFrame"] table {
    background-color: #2a2a2a !important;  /* grey background */
    color: #ffffff !important;             /* white text */
}

/* Header styling */
[data-testid="stDataFrame"] thead tr th {
    background-color: #3a3a3a !important;  /* slightly darker header */
    color: #ffffff !important;             /* white text */
    font-weight: 600 !important;
    text-align: center !important;
}

/* Body cells */
[data-testid="stDataFrame"] tbody tr td {
    background-color: #2a2a2a !important;  /* grey background */
    color: #ffffff !important;
    text-align: center !important;
}

/* Alternate row subtle contrast */
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
    background-color: #333333 !important;
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

def blue_gradient(n):
    seq = BLUE_PALETTE
    if n <= 0:
        return []
    if n <= len(seq):
        step = (len(seq)-1)/max(1,(n-1))
        idxs = [int(round(i*step)) for i in range(n)]
        return [seq[i] for i in idxs]
    return [seq[i % len(seq)] for i in range(n)]

UNIT_FACTORS = {"g":1.0, "kg":1.0/1000.0, "mg":1000.0, "µg":1_000_000.0, "t":1.0/1_000_000.0}
UNIT_LABELS = {"g":"g CO₂e","kg":"kg CO₂e","mg":"mg CO₂e","µg":"µg CO₂e","t":"tonnes CO₂e"}

def convert_from_grams(val_g, unit):
    return val_g * UNIT_FACTORS.get(unit, 1.0)

def human_emission_str(val_g):
    if pd.isna(val_g):
        return "0 g"
    kg = val_g / 1000.0
    if kg >= 1000:
        return f"{kg/1000:.2f} t"
    if kg >= 1:
        return f"{kg:.2f} kg"
    if val_g >= 1:
        return f"{val_g:.2f} g"
    return f"{val_g*1000:.2f} mg"

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
# Robust cascading filters with stable dynamic options and safe reset
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

# initialize persistent keys
for _, col in filter_order:
    key = f"f_{col}"
    if key not in st.session_state:
        st.session_state[key] = []

if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False

def _reset_filters():
    st.session_state.reset_trigger = True

st.markdown('<div class="filters">', unsafe_allow_html=True)
cols = st.columns([1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.6])

# precompute full option space for each filter column
all_options = {
    col: sorted(df[col].dropna().unique()) if col in df.columns else []
    for _, col in filter_order
}

# build filters sequentially but derive allowed values logically each pass
current_df = df.copy()
for (label, col), holder in zip(filter_order, cols[:-1]):
    with holder:
        # derive allowed options using selections in previous filters
        # (based on currently filtered subset)
        allowed_opts = sorted(current_df[col].dropna().unique()) if col in current_df.columns else []
        sel_key = f"f_{col}"

        # sanitize defaults (keep only values that exist in allowed_opts)
        existing = st.session_state.get(sel_key, [])
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        safe_default = [v for v in existing if v in allowed_opts]

        st.multiselect(label, options=allowed_opts, default=safe_default, key=sel_key)

        # filter a *fresh* copy for downstream logic (don't mutate in place)
        selected = st.session_state.get(sel_key, [])
        if selected and col in current_df.columns:
            current_df = df[df[col].isin(selected)]  # filter from original df for stability

# Reset button
with cols[-1]:
    st.button("Reset Filters", key="reset_filters_btn", on_click=_reset_filters)

st.markdown('</div>', unsafe_allow_html=True)

# handle reset outside callback
if st.session_state.reset_trigger:
    for _, c in filter_order:
        key = f"f_{c}"
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.reset_trigger = False
    st.rerun()

# final filtered dataset
filtered = df.copy()
for _, col in filter_order:
    sel = st.session_state.get(f"f_{col}", [])
    if sel and col in filtered.columns:
        filtered = filtered[filtered[col].isin(sel)]

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# -------------------------
# Emissions unit selector
# -------------------------
emissions_unit = st.selectbox("Emissions unit", options=["kg","g","mg","µg","t"], index=0)
emissions_unit_label = UNIT_LABELS.get(emissions_unit, "kg CO₂e")

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
    }

totals = compute_totals(filtered)
total_impressions = totals["TOTAL_IMPRESSIONS"]
total_conversions = totals["TOTAL_CONVERSIONS"]
total_clicks = totals["TOTAL_CLICKS"]
total_revenue = totals["REVENUE"]
total_spend = totals["SPEND"]
total_emissions_g = totals["TOTAL_EMISSIONS_GRAMS"]

emissions_per_conv_g = safe_div(total_emissions_g, total_conversions)
emissions_per_impr_g = safe_div(total_emissions_g, total_impressions)
emissions_per_dollar_mg = safe_div(total_emissions_g, total_spend) * 1000.0

avg_cpc = safe_div(total_spend, total_clicks)
avg_cpm = safe_div(total_spend, total_impressions) * 1000.0
engagement_rate = safe_div(total_clicks, total_impressions) * 100.0
conversion_rate = safe_div(total_conversions, total_impressions) * 100.0
roas = safe_div(total_revenue, total_spend)

def display_emission(val_g, unit=emissions_unit):
    return f"{convert_from_grams(val_g, unit):,.2f} {UNIT_LABELS.get(unit)}"

# KPI rows (emissions-focused first)
row1 = [
    ("Total Emissions", display_emission(total_emissions_g), "sum(TOTAL_EMISSIONS_GRAMS)"),
    ("Emissions / Conversion", f"{convert_from_grams(emissions_per_conv_g, emissions_unit):.4f} {UNIT_LABELS.get(emissions_unit)}", ""),
    ("Emissions / Impression", f"{convert_from_grams(emissions_per_impr_g, emissions_unit):.4f} {UNIT_LABELS.get(emissions_unit)}", ""),
    ("Emissions / $ Spent", f"{emissions_per_dollar_mg:,.0f} mg/$", ""),
    ("Engagement Rate", f"{engagement_rate:.2f}%", ""),
    ("Conversion Rate", f"{conversion_rate:.2f}%", ""),
]

row2 = [
    ("Total Impressions (M)", f"{total_impressions/1e6:.2f}M", ""),
    ("Total Conversions", fmt_count(total_conversions), ""),
    ("Total Revenue", fmt_money(total_revenue), ""),
    ("Total Spend", fmt_money(total_spend), ""),
    ("Avg. CPC", fmt_money(avg_cpc), ""),
    ("Avg. CPM", fmt_money(avg_cpm), ""),
]

def render_kpi_row(items):
    cols_kpi = st.columns(len(items), gap="small")
    for c, (label, value, tooltip) in zip(cols_kpi, items):
        c.markdown(f"<div class='kpi-card' title='{tooltip}'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>", unsafe_allow_html=True)

render_kpi_row(row1)
st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
render_kpi_row(row2)

# -------------------------
# Chart utilities (final stacked_with_emissions replacement)
# -------------------------
def ordered_cats_by_value(df_local, x_col, value_col, top_n=12):
    if x_col not in df_local.columns or value_col not in df_local.columns:
        return []
    return df_local.groupby(x_col)[value_col].sum().sort_values(ascending=False).head(top_n).index.tolist()

def stacked_with_emissions(df_local, x_col, stack_col, value_col, emissions_col, title, top_n=12):
    if x_col not in df_local.columns or value_col not in df_local.columns:
        return None

    tmp = df_local.copy()
    # detect datetime-like
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
        # take most recent top_n weeks/dates by value or last top_n dates
        order_index = tmp.groupby(tmp[x_col])[value_col].sum().sort_index().index.tolist()
        # keep last top_n (chronological)
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

    emiss_vals = [convert_from_grams(v, emissions_unit) for v in emiss_raw]

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=emiss_vals,
            name=f"Emissions ({UNIT_LABELS.get(emissions_unit)})",
            mode="lines+markers",
            line=dict(color=EMISSIONS_COLOR, width=2, shape="spline"),
            marker=dict(size=6),
            connectgaps=True,
            yaxis="y2",
            hovertemplate="%{y:.3f} " + UNIT_LABELS.get(emissions_unit) + "<br>%{x}<extra></extra>",
        )
    )

    numeric_emiss = [v for v in emiss_vals if v is not None]
    min_e = float(min(numeric_emiss)) if numeric_emiss else 0.0
    max_e = float(max(numeric_emiss)) if numeric_emiss else 0.0
    if max_e > 0:
        buffer_low = min_e * 0.08 if min_e > 0 else 0
        y2_range = [max(0, min_e - buffer_low), max_e * 1.12]
    else:
        y2_range = None

    fig.update_layout(
        title=title or "",
        title_font=dict(color="#ffffff"),
        barmode="stack",
        hovermode="x unified",
        xaxis=dict(tickangle=-30, automargin=True, tickfont=dict(size=10, color="#e0e0e0")),
        yaxis=dict(
            title=dict(text=value_col, font=dict(color="#e0e0e0")),
            automargin=True,
            tickfont=dict(color="#e0e0e0")
        ),
        yaxis2=dict(
            title=dict(text=f"Emissions ({UNIT_LABELS.get(emissions_unit)})", font=dict(color="#e0e0e0")),
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
    emiss_vals = [convert_from_grams(d[d[x_col] == c][emissions_col].sum(), emissions_unit) for c in cats]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=cats,
            y=bar_vals,
            name=bar_col,
            marker_color=blue_gradient(1)[0],
            hovertemplate="%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cats,
            y=emiss_vals,
            name=f"Emissions ({UNIT_LABELS.get(emissions_unit)})",
            mode="lines+markers",
            line=dict(color=EMISSIONS_COLOR, width=2),
            marker=dict(size=6),
            yaxis="y2",
            hovertemplate="%{y:.3f} " + UNIT_LABELS.get(emissions_unit) + "<extra></extra>",
        )
    )

    fig.update_layout(
        title=title or "",
        title_font=dict(color="#ffffff"),
        yaxis=dict(
            title=dict(text=bar_col, font=dict(color="#e0e0e0")),
            automargin=True,
            tickfont=dict(color="#e0e0e0")
        ),
        yaxis2=dict(
            title=dict(text=f"Emissions ({UNIT_LABELS.get(emissions_unit)})", font=dict(color="#e0e0e0")),
            overlaying="y",
            side="right",
            showgrid=False,
            title_standoff=80,
            tickfont=dict(color="#e0e0e0"),
        ),
        margin=dict(l=64, r=200, t=60, b=100),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(color="#e0e0e0")
        ),
        xaxis=dict(
            tickangle=-30,
            automargin=True,
            tickfont=dict(size=10, color="#e0e0e0")
        ),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,26,0.5)",
        font=dict(color="#e0e0e0")
    )

    return fig

def pie_blue(df_local, name_col, value_col, top_n=12, title=None):
    # allow calling as pie_blue(df, "INDUSTRY", "TOTAL_EMISSIONS_GRAMS", "Title")
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
# Build the 15 charts
# -------------------------
chart1  = stacked_with_emissions(filtered,"INVENTORY_ID","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Emissions & Impressions — Inventory (stacked)")
chart2  = stacked_with_emissions(filtered,"INDUSTRY","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Emissions & Impressions — Industry (stacked)")
chart3  = stacked_with_emissions(filtered,"REGION","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Emissions & Impressions — Region (stacked)")
chart4  = stacked_with_emissions(filtered,"DEVICE_TYPE","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Emissions & Impressions — Device (stacked)")

if "DATE" in filtered.columns:
    tmp = filtered.copy()
    tmp["DATE"] = pd.to_datetime(tmp["DATE"], errors="coerce")
    tmp = tmp.dropna(subset=["DATE"])
    tmp["WEEK"] = tmp["DATE"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    chart5 = stacked_with_emissions(tmp,"WEEK","AD_FORMAT","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Emissions & Impressions — Over Time (Weekly)")
else:
    chart5 = None

chart6  = pie_blue(filtered,"INDUSTRY","TOTAL_EMISSIONS_GRAMS","Emissions Share by Industry")
chart7  = bar_with_emissions(filtered,"REGION","TOTAL_CONVERSIONS","TOTAL_EMISSIONS_GRAMS","Emissions per Conversion by Region")
chart8  = bar_with_emissions(filtered,"DEVICE_TYPE","TOTAL_IMPRESSIONS","TOTAL_EMISSIONS_GRAMS","Emissions per Impression by Device Type")
chart9  = stacked_with_emissions(filtered,"INDUSTRY","DEVICE_TYPE","TOTAL_CONVERSIONS","TOTAL_EMISSIONS_GRAMS","Conversions by Industry (stacked by Device) + Emissions")
chart10 = bar_with_emissions(filtered,"ADVERTISER_ID","SPEND","TOTAL_EMISSIONS_GRAMS","Emissions per $ Spend by Advertiser")
chart11 = pie_blue(filtered,"REGION","SPEND","Spend Distribution by Region")
chart12 = stacked_with_emissions(filtered,"ADVERTISER_ID","AD_FORMAT","SPEND","TOTAL_EMISSIONS_GRAMS","Revenue vs Spend by Advertiser + Emissions")
chart13 = stacked_with_emissions(filtered,"AD_FORMAT","DEVICE_TYPE","TOTAL_CLICKS","TOTAL_EMISSIONS_GRAMS","Average CPC by Ad Format + Emissions")
chart14 = pie_blue(filtered,"DEVICE_TYPE","TOTAL_CONVERSIONS","Conversions Share by Device Type")
chart15 = stacked_with_emissions(filtered,"AD_FORMAT","DEVICE_TYPE","TOTAL_CONVERSIONS","TOTAL_EMISSIONS_GRAMS","Combined Efficiency View — Emissions per Impression & Conversion")

# -------------------------
# Layout: two charts per row, in the requested order
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
# Recommendations (polished)
# -------------------------
def recommendations_block(df_local):
    if df_local.empty or not {"SPEND","REVENUE","TOTAL_EMISSIONS_GRAMS"}.issubset(df_local.columns):
        st.warning("Not enough data for recommendations.")
        return

    d = df_local.copy()
    d["ROAS"] = safe_div(d["REVENUE"], d["SPEND"])
    d["EMISSIONS_MG_PER_$"] = safe_div(d["TOTAL_EMISSIONS_GRAMS"], d["SPEND"]) * 1000.0
    d["CONV_RATE"] = safe_div(d.get("TOTAL_CONVERSIONS", 0), d.get("TOTAL_IMPRESSIONS", 0)) * 100.0

    med_roas = d["ROAS"].median()
    med_emiss = d["EMISSIONS_MG_PER_$"].median()

    scale_df = d[(d["ROAS"] > max(2.5, med_roas)) & (d["EMISSIONS_MG_PER_$"] < 3000)]
    reduce_df = d[(d["ROAS"] < 1.5) & (d["EMISSIONS_MG_PER_$"] > 5000)]
    optimize_df = d[((d["ROAS"].between(1.5, 2.5)) | (d["EMISSIONS_MG_PER_$"] > 3000))]

    def format_list(sub):
        if sub.empty:
            return ["No candidates"]
        out = []
        key_cols = ["CAMPAIGN_ID", "AD_FORMAT", "DEVICE_TYPE", "REGION"]
        for _, r in sub.sort_values("SPEND", ascending=False).head(6).iterrows():
            seg = next((str(r[c]) for c in key_cols if c in r and pd.notna(r[c])), "Segment")
            out.append(f"{seg} — ROAS {r.get('ROAS',0):.2f}x · {r.get('EMISSIONS_MG_PER_$',0):.0f} mg/$ · Conv {r.get('CONV_RATE',0):.2f}%")
        return out

    s_list = format_list(scale_df)
    r_list = format_list(reduce_df)
    o_list = format_list(optimize_df)

    st.markdown("### 🎯 Next Best Placement Recommendations", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("#### 🟢 SCALE / INVEST", unsafe_allow_html=True)
        for i in s_list: st.markdown(f"- {i}")
    with c2:
        st.markdown("#### 🔴 REDUCE / AVOID", unsafe_allow_html=True)
        for i in r_list: st.markdown(f"- {i}")
    with c3:
        st.markdown("#### 🟡 OPTIMIZE / REVIEW", unsafe_allow_html=True)
        for i in o_list: st.markdown(f"- {i}")

recommendations_block(filtered)

# -------------------------
# Filtered data preview
# -------------------------
st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
st.markdown("#### Filtered Data Preview")

table_html = filtered.head(200).to_html(classes="custom-table", index=True, border=0, justify="center")

html = f"""
<style>
/* styling (grey background + white text) */
.custom-table-container {{
  background: #2a2a2a;
  padding: 18px;
  border-radius: 10px;
  overflow: auto;
}}
.custom-table {{
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
}}
.custom-table thead th {{
  background: #3a3a3a;
  color: #ffffff;
  font-weight: 700;
  padding: 12px 10px;
  text-align: left;
  position: sticky;
  top: 0;
  z-index: 2;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.custom-table tbody td {{
  background: #2a2a2a;
  color: #ffffff;
  padding: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  vertical-align: middle;
}}
.custom-table tbody tr:nth-child(even) td {{
  background: #333333;
}}
.custom-table tbody th {{
  background: transparent;
  color: #cfcfcf;
  font-weight: 600;
  padding: 10px;
  text-align: left;
}}
.wrap {{
  width: 100%;
}}
</style>

<div class="wrap">
  <div class="custom-table-container">
    {table_html}
  </div>
</div>
"""

components.html(html, height=520, scrolling=True)
