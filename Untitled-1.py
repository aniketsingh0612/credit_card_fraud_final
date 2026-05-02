"""
💳 Credit Card Fraud Detection Dashboard
========================================
Run with: streamlit run fraud_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ──────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL THEME & STYLE
# ──────────────────────────────────────────────
SAFE_COLOR   = "#00C897"   # emerald green  → legitimate
FRAUD_COLOR  = "#FF4B6E"   # vivid red      → fraudulent
ACCENT_COLOR = "#7B61FF"   # violet         → highlights
BG_CARD      = "#1A1D2E"   # dark card bg
BG_CHART     = "#12152A"   # darker chart bg
TEXT_MAIN    = "#E8EAF6"   # off-white
GRID_COLOR   = "#2A2D45"

PALETTE = {0: SAFE_COLOR, 1: FRAUD_COLOR}

st.markdown(
    f"""
    <style>
    /* ─── Google Font ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Grotesk:wght@600;700&display=swap');

    /* ─── Root overrides ──────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: #0E1121;
        color: {TEXT_MAIN};
    }}

    /* ─── Sidebar ─────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {BG_CARD};
        border-right: 1px solid {GRID_COLOR};
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT_MAIN} !important; }}

    /* ─── Main container ──────────────────────── */
    .block-container {{ padding: 2rem 2.5rem 3rem; }}

    /* ─── KPI cards ───────────────────────────── */
    .kpi-card {{
        background: {BG_CARD};
        border: 1px solid {GRID_COLOR};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }}
    .kpi-label {{
        font-size: 0.78rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: #8890B5;
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        color: #8890B5;
        margin-top: 0.4rem;
    }}

    /* ─── Section headers ─────────────────────── */
    .section-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
        color: {TEXT_MAIN};
        border-left: 3px solid {ACCENT_COLOR};
        padding-left: 0.75rem;
        margin-bottom: 1rem;
    }}

    /* ─── Insight pills ───────────────────────── */
    .insight {{
        background: rgba(123,97,255,.12);
        border: 1px solid rgba(123,97,255,.3);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        color: #C5BCFF;
        margin-top: 0.6rem;
    }}

    /* ─── Metric delta override ───────────────── */
    [data-testid="stMetricDelta"] {{ color: {FRAUD_COLOR}; }}

    /* ─── Hide Streamlit chrome ───────────────── */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ─── Divider ─────────────────────────────── */
    hr {{ border: none; border-top: 1px solid {GRID_COLOR}; margin: 1.5rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# CHART DEFAULTS  (dark background)
# ──────────────────────────────────────────────
def apply_dark_style(ax, fig):
    fig.patch.set_facecolor(BG_CHART)
    ax.set_facecolor(BG_CHART)
    ax.tick_params(colors=TEXT_MAIN, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MAIN)
    ax.yaxis.label.set_color(TEXT_MAIN)
    ax.title.set_color(TEXT_MAIN)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def make_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    apply_dark_style(ax, fig)
    return fig, ax

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("credit_card_fraud_10k.csv")
    df.drop_duplicates(inplace=True)
    df["transaction_hour"] = df["transaction_hour"].astype(int)
    return df

try:
    df = load_data()
except FileNotFoundError:
    # Try absolute path for local testing
    try:
        df = pd.read_csv("/mnt/user-data/uploads/credit_card_fraud_10k.csv")
        df.drop_duplicates(inplace=True)
        df["transaction_hour"] = df["transaction_hour"].astype(int)
    except Exception as e:
        st.error(f"⚠️ Could not load data: {e}\n\nPlace `credit_card_fraud_10k.csv` in the same folder as this script.")
        st.stop()

# ──────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Space Grotesk;font-size:1.1rem;letter-spacing:.08em;'>"
        "🔍 FILTERS</h2>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Amount slider
    amt_min, amt_max = float(df["amount"].min()), float(df["amount"].max())
    selected_amount = st.slider(
        "Transaction Amount (USD)",
        min_value=amt_min, max_value=amt_max,
        value=(amt_min, amt_max), step=10.0,
        format="$%.0f",
    )

    st.markdown(" ")

    # Age slider
    age_min, age_max = int(df["cardholder_age"].min()), int(df["cardholder_age"].max())
    selected_age = st.slider(
        "Cardholder Age",
        min_value=age_min, max_value=age_max,
        value=(age_min, age_max),
    )

    st.markdown(" ")

    # Merchant category
    cats = ["All"] + sorted(df["merchant_category"].unique().tolist())
    selected_cat = st.selectbox("Merchant Category", cats)

    st.markdown(" ")

    # Transaction type
    txn_type = st.radio(
        "Transaction Type",
        ["All", "Domestic Only", "Foreign Only"],
        horizontal=False,
    )

    st.markdown("---")

    # Apply filters
    mask = (
        (df["amount"] >= selected_amount[0]) &
        (df["amount"] <= selected_amount[1]) &
        (df["cardholder_age"] >= selected_age[0]) &
        (df["cardholder_age"] <= selected_age[1])
    )
    if selected_cat != "All":
        mask &= df["merchant_category"] == selected_cat
    if txn_type == "Domestic Only":
        mask &= df["foreign_transaction"] == 0
    elif txn_type == "Foreign Only":
        mask &= df["foreign_transaction"] == 1

    fdf = df[mask].copy()

    n_shown  = len(fdf)
    n_fraud  = fdf["is_fraud"].sum()
    pct      = (n_fraud / n_shown * 100) if n_shown > 0 else 0

    st.markdown(
        f"<div style='background:rgba(0,200,151,.08);border:1px solid rgba(0,200,151,.3);"
        f"border-radius:10px;padding:.8rem 1rem;font-size:.82rem;color:#8DFFD8;'>"
        f"<b>Showing:</b> {n_shown:,} transactions<br>"
        f"<b>Fraud in view:</b> {n_fraud:,} ({pct:.2f}%)"
        f"</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Space Grotesk;font-size:2rem;font-weight:700;"
    "background:linear-gradient(90deg,#7B61FF,#FF4B6E);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;margin-bottom:0.2rem;'>"
    "💳 Fraud Risk Intelligence</h1>"
    "<p style='color:#8890B5;font-size:.9rem;margin-top:0;'>Credit Card Anomaly Detection · 10,000 Transactions</p>",
    unsafe_allow_html=True,
)

if fdf.empty:
    st.warning("No transactions match your filters. Try widening the ranges.")
    st.stop()

# ──────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)

total_fraud_pct  = df["is_fraud"].mean() * 100
avg_fraud_amount = fdf[fdf["is_fraud"] == 1]["amount"].mean() if n_fraud > 0 else 0
avg_safe_amount  = fdf[fdf["is_fraud"] == 0]["amount"].mean()

kpis = [
    (k1, "Total Transactions", f"{n_shown:,}", f"of {len(df):,} total", TEXT_MAIN),
    (k2, "Fraud Cases",        f"{n_fraud:,}", f"{pct:.2f}% of filtered", FRAUD_COLOR),
    (k3, "Avg Fraud Amount",   f"${avg_fraud_amount:,.0f}", "per fraudulent txn", FRAUD_COLOR),
    (k4, "Avg Safe Amount",    f"${avg_safe_amount:,.0f}", "per legitimate txn", SAFE_COLOR),
]

for col, label, value, sub, color in kpis:
    with col:
        st.markdown(
            f"<div class='kpi-card'>"
            f"  <div class='kpi-label'>{label}</div>"
            f"  <div class='kpi-value' style='color:{color};'>{value}</div>"
            f"  <div class='kpi-sub'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────
# ROW 1 : Fraud Distribution  &  Amount Boxplot
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📊 Fraud Risk Analysis</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

# — Chart 1 : Fraud vs Non-Fraud (bar) —
with c1:
    fig, ax = make_fig(6, 4)
    counts = fdf["is_fraud"].value_counts().sort_index()
    bars = ax.bar(
        ["Legitimate", "Fraudulent"],
        counts.values,
        color=[SAFE_COLOR, FRAUD_COLOR],
        width=0.45,
        edgecolor="none",
    )
    # value labels
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values) * 0.01,
            f"{val:,}", ha="center", va="bottom", color=TEXT_MAIN, fontsize=10, fontweight="bold",
        )
    ax.set_title("Transaction Legitimacy Breakdown", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Transactions")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<div class='insight'>💡 Fraud cases represent only ~1.5% of all transactions — "
        "this class imbalance is a key modelling challenge.</div>",
        unsafe_allow_html=True,
    )

# — Chart 2 : Amount Boxplot —
with c2:
    fig, ax = make_fig(6, 4)
    data_safe  = fdf[fdf["is_fraud"] == 0]["amount"]
    data_fraud = fdf[fdf["is_fraud"] == 1]["amount"]

    bp = ax.boxplot(
        [data_safe, data_fraud],
        patch_artist=True,
        widths=0.4,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=GRID_COLOR, linewidth=1.2),
        capprops=dict(color=GRID_COLOR, linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    bp["boxes"][0].set_facecolor(SAFE_COLOR + "55")
    bp["boxes"][0].set_edgecolor(SAFE_COLOR)
    bp["fliers"][0].set_markerfacecolor(SAFE_COLOR)
    bp["boxes"][1].set_facecolor(FRAUD_COLOR + "55")
    bp["boxes"][1].set_edgecolor(FRAUD_COLOR)
    bp["fliers"][1].set_markerfacecolor(FRAUD_COLOR)

    ax.set_xticklabels(["Legitimate", "Fraudulent"])
    ax.set_title("Transaction Amount Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Amount (USD)")
    # cap y-axis at 99th percentile for readability
    ax.set_ylim(0, fdf["amount"].quantile(0.99) * 1.1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<div class='insight'>💡 Fraudulent transactions tend to have <b>higher median amounts</b> "
        "— large purchases are a key fraud signal.</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# ROW 2 : Transaction Hours  &  Foreign Txn
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🕐 Transaction Behavior Insights</div>", unsafe_allow_html=True)

c3, c4 = st.columns(2)

# — Chart 3 : Hour distribution —
with c3:
    fig, ax = make_fig(7, 4)
    safe_hrs  = fdf[fdf["is_fraud"] == 0]["transaction_hour"]
    fraud_hrs = fdf[fdf["is_fraud"] == 1]["transaction_hour"]
    bins = np.arange(0, 25, 1)
    ax.hist(safe_hrs,  bins=bins, color=SAFE_COLOR,  alpha=0.7, label="Legitimate", edgecolor="none")
    ax.hist(fraud_hrs, bins=bins, color=FRAUD_COLOR, alpha=0.85, label="Fraudulent", edgecolor="none")
    ax.set_title("Hourly Transaction Activity", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (0 = midnight)")
    ax.set_ylabel("Number of Transactions")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(facecolor=BG_CARD, edgecolor=GRID_COLOR, labelcolor=TEXT_MAIN, fontsize=9)
    # Shade night hours
    ax.axvspan(0, 6,  alpha=0.08, color=FRAUD_COLOR, zorder=0, label="_nolegend_")
    ax.axvspan(22, 24, alpha=0.08, color=FRAUD_COLOR, zorder=0, label="_nolegend_")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<div class='insight'>💡 Fraud spikes in the <b>early morning hours (0–5 AM)</b> "
        "— highlighted in red. Automated scripts often strike when monitoring is minimal.</div>",
        unsafe_allow_html=True,
    )

# — Chart 4 : Foreign transaction risk —
with c4:
    fig, ax = make_fig(7, 4)
    groups = fdf.groupby("foreign_transaction")["is_fraud"].agg(["sum", "count"])
    groups["rate"] = groups["sum"] / groups["count"] * 100
    bars = ax.bar(
        ["Domestic", "International"],
        groups["rate"].values,
        color=[SAFE_COLOR, FRAUD_COLOR],
        width=0.45,
        edgecolor="none",
    )
    for bar, val in zip(bars, groups["rate"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}%", ha="center", va="bottom", color=TEXT_MAIN, fontsize=11, fontweight="bold",
        )
    ax.set_title("Fraud Rate: Domestic vs International", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_ylim(0, max(groups["rate"].values) * 1.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<div class='insight'>💡 <b>Foreign / international transactions carry significantly higher fraud risk.</b> "
        "Geo-based rules are among the most effective first-line defences.</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# ROW 3 : Device Trust Score  &  Merchant Risk
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🔐 Device & Merchant Risk Profile</div>", unsafe_allow_html=True)

c5, c6 = st.columns(2)

# — Chart 5 : Device Trust Score —
with c5:
    fig, ax = make_fig(6, 4)
    safe_dts  = fdf[fdf["is_fraud"] == 0]["device_trust_score"]
    fraud_dts = fdf[fdf["is_fraud"] == 1]["device_trust_score"]
    ax.hist(safe_dts,  bins=20, color=SAFE_COLOR,  alpha=0.65, label="Legitimate", edgecolor="none")
    ax.hist(fraud_dts, bins=20, color=FRAUD_COLOR, alpha=0.85, label="Fraudulent", edgecolor="none")
    ax.set_title("Device Trust Score Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Device Trust Score (0 – 100)")
    ax.set_ylabel("Number of Transactions")
    ax.legend(facecolor=BG_CARD, edgecolor=GRID_COLOR, labelcolor=TEXT_MAIN, fontsize=9)
    ax.axvline(x=50, color="white", linestyle="--", linewidth=1, alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<div class='insight'>💡 Fraudulent transactions cluster at <b>lower device trust scores</b>. "
        "Scores below 40 should trigger step-up authentication.</div>",
        unsafe_allow_html=True,
    )

# — Chart 6 : Fraud rate by merchant category —
with c6:
    fig, ax = make_fig(6, 4)
    cat_stats = (
        fdf.groupby("merchant_category")["is_fraud"]
        .agg(["sum", "count"])
        .assign(rate=lambda x: x["sum"] / x["count"] * 100)
        .sort_values("rate", ascending=True)
    )
    colors = [
        FRAUD_COLOR if r > cat_stats["rate"].median() else SAFE_COLOR
        for r in cat_stats["rate"]
    ]
    bars = ax.barh(cat_stats.index, cat_stats["rate"], color=colors, edgecolor="none", height=0.5)
    for bar, val in zip(bars, cat_stats["rate"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color=TEXT_MAIN, fontsize=9, fontweight="bold")
    ax.set_title("Fraud Rate by Merchant Category", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_xlim(0, cat_stats["rate"].max() * 1.35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<div class='insight'>💡 Certain merchant categories (highlighted in red) carry <b>above-median fraud rates</b>. "
        "Category-specific thresholds improve detection precision.</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# ROW 4 : Velocity & Location Mismatch heatmap
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📈 Velocity & Correlation Signals</div>", unsafe_allow_html=True)

c7, c8 = st.columns([1.1, 0.9])

# — Chart 7 : Velocity last 24h —
with c7:
    fig, ax = make_fig(7, 4)
    safe_v  = fdf[fdf["is_fraud"] == 0]["velocity_last_24h"]
    fraud_v = fdf[fdf["is_fraud"] == 1]["velocity_last_24h"]
    vbins = np.arange(0, max(fdf["velocity_last_24h"].max(), 1) + 2, 1)
    ax.hist(safe_v,  bins=vbins, color=SAFE_COLOR,  alpha=0.7, label="Legitimate", edgecolor="none")
    ax.hist(fraud_v, bins=vbins, color=FRAUD_COLOR, alpha=0.85, label="Fraudulent", edgecolor="none")
    ax.set_title("Transaction Velocity (last 24 h)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Transactions in Previous 24 Hours")
    ax.set_ylabel("Count")
    ax.legend(facecolor=BG_CARD, edgecolor=GRID_COLOR, labelcolor=TEXT_MAIN, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<div class='insight'>💡 High velocity (many transactions in 24 h) is a strong fraud signal — "
        "card-testing attacks often fire many small charges rapidly.</div>",
        unsafe_allow_html=True,
    )

# — Chart 8 : Numeric feature correlation heatmap —
with c8:
    fig, ax = make_fig(5.5, 4)
    num_cols = ["amount", "device_trust_score", "velocity_last_24h",
                "foreign_transaction", "location_mismatch", "is_fraud"]
    corr = fdf[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(145, 10, s=80, l=40, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8, "color": TEXT_MAIN},
        linewidths=0.5, linecolor=BG_CHART, square=True,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    # style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_MAIN)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_MAIN, fontsize=7)
    cbar.outline.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<div class='insight'>💡 <code>location_mismatch</code>, <code>foreign_transaction</code>, and "
        "<code>device_trust_score</code> show the strongest correlation with fraud.</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4A4F72;font-size:.78rem;'>"
    "💳 Fraud Risk Intelligence Dashboard · Built with Streamlit · "
    "Dataset: 10,000 synthetic credit card transactions</p>",
    unsafe_allow_html=True,
)