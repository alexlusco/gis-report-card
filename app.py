import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

ART = Path("artifacts")
FTE_BUNDLE_PATH = ART / "fte_bootstrap_bundle.joblib"
SHARE_BUNDLE_PATH = ART / "share_bootstrap_bundle.joblib"

PROVINCES_TERRITORIES = [
    "BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL", "YT", "NT", "NU"
]

POP_RANGE_OPTIONS = ["< 25k", "25k–99k", "100k–249k", "250k–999k", "1M+"]
BUDGET_RANGE_OPTIONS = ["< $250M", "$250M–$999M", "$1B–$4.9B", "$5B+"]

Q_LO = 0.25
Q_HI = 0.75


def predict_band(bundle, X_df, q_lo=Q_LO, q_hi=Q_HI):
    central = bundle["central_model"]
    boots = bundle["bootstrap_models"]

    central_pred = float(central.predict(X_df)[0])
    boot_preds = np.array([m.predict(X_df)[0] for m in boots], dtype=float)

    lo = float(np.quantile(boot_preds, q_lo))
    hi = float(np.quantile(boot_preds, q_hi))
    lo, hi = min(lo, hi), max(lo, hi)

    return lo, central_pred, hi


def position_label(cur, lo, hi):
    if cur is None:
        return None
    if cur < lo:
        return "Below peer expected range"
    if cur > hi:
        return "Above peer expected range"
    return "Within peer expected range"


def band_bar(value_lo, value_mid, value_hi, title, units, current_value=None, x0=None, x_max=None):
    """
    Draw:
      - thick horizontal band from lo..hi (peer expected range)
      - red vertical line at median
      - optional orange diamond for current value
      - legend outside plot
      - auto x-limits padded around (band + current) unless x0/x_max explicitly provided
      - if units contains '%', use 0.1 tick increments
    """
    fig, ax = plt.subplots(figsize=(7.8, 1.6))

    # Band
    ax.hlines(
        y=0,
        xmin=value_lo,
        xmax=value_hi,
        linewidth=10,
        label="Peer expected range (25th–75th)"
    )

    # Median (red vertical line)
    ax.vlines(
        x=value_mid,
        ymin=-0.45,
        ymax=0.45,
        colors="red",
        linewidth=3,
        label="Median"
    )

    # Current marker (orange)
    if current_value is not None:
        ax.plot(
            [current_value], [0],
            marker="D",
            linestyle="None",
            color="orange",
            markeredgecolor="orange",
            label="Current"
        )

    ax.set_title(title, fontsize=11, loc="left")
    ax.set_yticks([])
    ax.set_xlabel(units)

    # Auto x-limits around band + current with padding
    span = max(1e-9, value_hi - value_lo)
    vals = [value_lo, value_hi, value_mid]
    if current_value is not None:
        vals.append(current_value)

    left = min(vals) - 0.25 * span
    right = max(vals) + 0.25 * span

    # Optional overrides ONLY if passed
    if x0 is not None:
        left = max(x0, left)
    if x_max is not None:
        right = x_max

    ax.set_xlim(left, right)

    # Spend-share ticks every 0.1% without snapping to 0
    if "%" in units:
        tick_start = np.floor(left * 10) / 10
        tick_end = np.ceil(right * 10) / 10
        ticks = np.arange(tick_start, tick_end + 0.1001, 0.1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks])

    # Legend outside (right)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8
    )
    fig.subplots_adjust(right=0.72)

    return fig


st.set_page_config(page_title="GIS FTE/$ Report Card", layout="centered")
st.title("GIS FTE/$ Report Card")


@st.cache_resource
def load_bundles():
    fte = joblib.load(FTE_BUNDLE_PATH)
    share = joblib.load(SHARE_BUNDLE_PATH)
    return fte, share


fte_bundle, share_bundle = load_bundles()

with st.sidebar:
    st.header("Client features (inputs)")

    province = st.selectbox("Province / Territory", PROVINCES_TERRITORIES, index=4)  # ON default
    population_range = st.selectbox("Population range", POP_RANGE_OPTIONS, index=2)
    budget_range = st.selectbox("Municipal budget range", BUDGET_RANGE_OPTIONS, index=1)

    annual_retention_rate = st.slider("Annual retention rate", 0.60, 0.99, 0.88, 0.01)
    training_hours = st.slider("Avg annual training hours per GIS employee", 0.0, 60.0, 18.0, 0.5)

    has_strategy = st.selectbox("Has GIS strategy?", ["No", "Yes"], index=1)
    tickets_per_month = st.slider("GIS service requests per month (tickets)", 10, 2500, 180, 10)

    avg_years_experience = st.slider("Avg years of experience (GIS staff)", 0.5, 25.0, 7.0, 0.5)
    pct_reactive = st.slider("% of work that is reactive", 0.05, 0.95, 0.55, 0.01)

    num_domains = st.slider("Number of domains served (e.g., police, health)", 1, 20, 8, 1)

    st.divider()
    st.header("Client current levels (overlay)")

    show_current = st.checkbox("Show current levels on visuals", value=True)
    current_fte = st.number_input("Current GIS FTE", min_value=0.0, max_value=2000.0, value=0.0, step=0.5)
    current_share = st.number_input("Current GIS spend share (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.01)

cur_fte = float(current_fte) if show_current and current_fte > 0 else None
cur_share = float(current_share) if show_current and current_share > 0 else None

X = pd.DataFrame([{
    "province_territory": province,
    "population_range": population_range,
    "municipal_budget_range": budget_range,
    "annual_retention_rate": float(annual_retention_rate),
    "avg_training_hours_per_employee": float(training_hours),
    "has_gis_strategy": 1 if has_strategy == "Yes" else 0,
    "tickets_per_month": int(tickets_per_month),
    "avg_years_experience": float(avg_years_experience),
    "pct_work_reactive": float(pct_reactive),
    "num_domains_served": int(num_domains),
}])

st.subheader("Peer expected ranges")

fte_lo, fte_mid, fte_hi = predict_band(fte_bundle, X, Q_LO, Q_HI)
sh_lo, sh_mid, sh_hi = predict_band(share_bundle, X, Q_LO, Q_HI)

fte_pos = position_label(cur_fte, fte_lo, fte_hi)
sh_pos = position_label(cur_share, sh_lo, sh_hi)

c1, c2 = st.columns(2)

with c1:
    st.metric("GIS FTE (median)", f"{fte_mid:.1f}")
    st.write(f"**Peer expected range:** {fte_lo:.1f} to {fte_hi:.1f}")
    if fte_pos is not None:
        st.write(f"**Current position:** {fte_pos}")

    st.pyplot(
        band_bar(
            fte_lo, fte_mid, fte_hi,
            "GIS FTE: peer expected range",
            "FTE",
            current_value=cur_fte
        ),
        clear_figure=True
    )

with c2:
    st.metric("GIS spend share (median)", f"{sh_mid:.3f}%")
    st.write(f"**Peer expected range:** {sh_lo:.3f}% to {sh_hi:.3f}%")
    if sh_pos is not None:
        st.write(f"**Current position:** {sh_pos}")

    st.pyplot(
        band_bar(
            sh_lo, sh_mid, sh_hi,
            "GIS spend share: peer expected range",
            "% of total budget",
            current_value=cur_share
        ),
        clear_figure=True
    )

st.divider()