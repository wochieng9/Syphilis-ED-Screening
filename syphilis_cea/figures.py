"""Matplotlib figure builders for the Streamlit application."""

from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

from .config import MarkovConfig
from .markov import markov_state_occupancy
from .reporting import _fmt_param_value, evpi_curve, psa_convergence
from .utils import ci_ellipse, dollar_fmt

def fig_ce_plane(dal, inc, title, wtp_lines=(50_000, 100_000, 150_000), subtitle=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(dal, inc / 1e6, s=.5, alpha=0.25, color="steelblue", rasterized=True)
    x_lim = np.array([min(dal.min() * 1.1, -50), dal.max() * 1.1])
    for wtp, col in zip(wtp_lines, ["#2a9d8f", "#e9c46a", "#e76f51"]):
        ax.plot(x_lim, wtp * x_lim / 1e6, ls="--", lw=1.2, color=col,
                label=f"${wtp/1000:.0f}K/DALY")
    ci_ellipse(ax, dal, inc / 1e6)
    ax.axhline(0, color="k", lw=0.6, zorder=3)
    ax.axvline(0, color="k", lw=0.6, zorder=3)
    # Shade quadrant labels
    xl, xr = ax.get_xlim() if ax.get_xlim()[0] != 0 else (x_lim[0], x_lim[1])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("ΔDALYs prevented (positive = intervention better)", fontsize=10)
    ax.set_ylabel("ΔCost  [positive = intervention costs more]", fontsize=9)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    subtitle_text = subtitle or ""
    if subtitle_text:
        ax.set_title(subtitle_text, fontsize=11, fontweight="bold")
    ax.legend(title="WTP threshold", fontsize=8, framealpha=0.7)
    ax.grid(alpha=0.15); ax.spines[["top","right"]].set_visible(False)
    return fig

def fig_ceac(dal_hs, ic_hs, dal_soc, ic_soc, wtp_max=200_000, subtitle=None):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    lam = np.arange(0, wtp_max + 1_000, 1_000)
    for dal_, ic_, label, col in [
        (dal_hs,  ic_hs,  "Health sector",  "steelblue"),
        (dal_soc, ic_soc, "Societal (nonmedical care + productivity)", "darkorange"),
    ]:
        probs = (lam[None, :] * dal_[:, None] - ic_[:, None] > 0).mean(axis=0)
        ax.plot(lam, probs, lw=2, label=label, color=col)
    for vline, col in [(50_000,"#2a9d8f"),(100_000,"#e9c46a"),(150_000,"#e76f51")]:
        ax.axvline(vline, ls=":", lw=1, color=col, alpha=0.8, label=f"${vline//1000}K")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.15)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.set_xlabel("WTP threshold ($/DALY)"); ax.set_ylabel("P(cost-effective)")
    subtitle_text = subtitle or ""
    if subtitle_text:
        ax.set_title(subtitle_text, fontweight="bold")
    fig.suptitle("Cost-Effectiveness Acceptability Curve", fontweight="bold")
    ax.tick_params(axis = "x", which="major", labelsize=8)
    ax.tick_params(axis = "y", which="major", labelsize=8)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[["top","right"]].set_visible(False)
    return fig

def fig_evpi(dal, ic, wtp_max=200_000):
    lam, evpi = evpi_curve(dal, ic, wtp_max)
    fig, ax   = plt.subplots(figsize=(7.5, 4))
    ax.plot(lam, evpi / 1e6, color="purple", lw=2)
    ax.fill_between(lam, evpi / 1e6, alpha=0.15, color="purple")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.2f}M"))
    ax.set_xlabel("WTP threshold ($/DALY)")
    ax.set_ylabel("EVPI per cohort ($M)")
    ax.set_title("Expected Value of Perfect Information", fontweight="bold")
    ax.grid(alpha=0.15); ax.spines[["top","right"]].set_visible(False)
    return fig

def fig_convergence(dal, ic, step=500):
    ns, rolling = psa_convergence(dal, ic, step)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    finite = np.isfinite(rolling)
    if not finite.any():
        ax.text(0.5, 0.5, "ICER undefined: cumulative mean DALYs are nonpositive",
                ha="center", va="center", transform=ax.transAxes)
    else:
        base = rolling[np.where(finite)[0][-1]]
        ax.plot(ns, rolling, color="steelblue", lw=2, label="Ratio of cumulative means")
        ax.axhline(base, color="k", ls="--", lw=1, label=f"Final: ${base:,.0f}")
        ax.fill_between(ns, base * 0.95, base * 1.05, alpha=0.12,
                        color="steelblue", label="+/-5% tolerance band")
        ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("PSA iterations")
    ax.set_ylabel("ICER ($/DALY) - health sector")
    ax.set_title("PSA Convergence Diagnostic", fontweight="bold")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    return fig

def fig_tornado(owsa_df: pd.DataFrame, base_icer: float, top_n: int = 12):
    df = owsa_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * max(len(df), 1) + 1)))
    if not np.isfinite(base_icer):
        ax.text(0.5, 0.5, "Base-case ICER is undefined; use the NMB tornado below.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    plotted_labels = []
    plot_index = 0
    for _, row in df.iterrows():
        low = float(row["ICER (low)"])
        high = float(row["ICER (high)"])
        if not (np.isfinite(low) and np.isfinite(high)):
            continue
        ax.barh(plot_index, high - base_icer, left=base_icer,
                height=0.55, color="#4a90d9", edgecolor="k", alpha=0.85)
        ax.barh(plot_index, low - base_icer, left=base_icer,
                height=0.55, color="#e08050", edgecolor="k", alpha=0.85)
        plotted_labels.append(row["Parameter"])
        plot_index += 1
    if not plotted_labels:
        ax.text(0.5, 0.5, "No finite ICER ranges; use the NMB tornado below.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    ax.axvline(base_icer, color="k", lw=1.5, ls="--", zorder=4,
               label=f"Base: ${base_icer:,.0f}")
    ax.set_yticks(range(len(plotted_labels)))
    ax.set_yticklabels(plotted_labels, fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.set_xlabel("ICER ($/DALY) - health-sector perspective")
    ax.set_title("One-Way Sensitivity Analysis (ICER)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    return fig

def fig_tornado_nmb(
    owsa_nmb_df: pd.DataFrame,
    base_nmb:    float,
    wtp:         float,
    perspective: str,
    top_n:       int = 12,
) -> plt.Figure:
    """
    Tornado chart for NMB-based OWSA.

    Bars extend from the base-case NMB toward the low/high parameter values.
    A vertical dashed line marks base-case NMB; a dotted line marks NMB = 0
    (break-even), making cost-effectiveness threshold visually explicit.

    Colour convention (consistent with fig_tornado):
        Blue  (#4a90d9) — direction that increases NMB
        Orange (#e08050) — direction that decreases NMB
    """
    df = owsa_nmb_df.head(top_n)
    n  = len(df)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * n + 1.2)))

    for idx, (_, row) in enumerate(df.iterrows()):
        nmb_lo = float(row["NMB (low param)"])
        nmb_hi = float(row["NMB (high param)"])

        # Determine which direction increases NMB for consistent colouring
        lo_delta = nmb_lo - base_nmb
        hi_delta = nmb_hi - base_nmb

        ax.barh(idx, lo_delta, left=base_nmb, height=0.55,
                color="#e08050" if lo_delta < hi_delta else "#4a90d9",
                edgecolor="k", linewidth=0.5, alpha=0.85)
        ax.barh(idx, hi_delta, left=base_nmb, height=0.55,
                color="#4a90d9" if hi_delta > lo_delta else "#e08050",
                edgecolor="k", linewidth=0.5, alpha=0.85)

        # Parameter-value annotations at bar tips
        x_right = max(nmb_lo, nmb_hi)
        x_left  = min(nmb_lo, nmb_hi)
        pad     = (owsa_nmb_df["NMB max"].max() - owsa_nmb_df["NMB min"].min()) * 0.005

        ax.text(x_right + pad, idx,
                _fmt_param_value(row["High param value"]),
                va="center", fontsize=7, color="#4a90d9")
        ax.text(x_left - pad, idx,
                _fmt_param_value(row["Low param value"]),
                va="center", ha="right", fontsize=7, color="#e08050")

    # Reference lines
    ax.axvline(base_nmb, color="k", lw=1.5, ls="--", zorder=4,
               label=f"Base NMB: ${base_nmb/1e6:,.2f}M")
    ax.axvline(0, color="dimgray", lw=0.9, ls=":", zorder=3,
               label="NMB = 0  (break-even)")

    # Shade negative NMB region as a visual cue
    x_min = ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else (
        owsa_nmb_df["NMB min"].min() * 1.15)
    if x_min < 0:
        ax.axvspan(x_min, 0, color="salmon", alpha=0.06, zorder=0,
                   label="NMB < 0 (not cost-effective at this WTP)")

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["Parameter"].tolist(), fontsize=8)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${x/1e6:,.1f}M"))
    ax.set_xlabel(
        f"Net Monetary Benefit ($M)  |  WTP = ${wtp/1000:.0f}K/DALY  |  "
        f"{'Health sector' if perspective == 'hs' else 'Societal'} perspective",
        fontsize=9)
    ax.set_title("One-Way Sensitivity Analysis — NMB Tornado", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(alpha=0.15, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

def fig_nmb_surface(prev_grid, tx_grid, nmb_G, wtp):
    fig, ax = plt.subplots(figsize=(8, 5))
    PP, TT  = np.meshgrid(prev_grid * 100, tx_grid * 100)
    vmax    = np.percentile(np.abs(nmb_G), 97); vmin = -vmax
    cf = ax.contourf(PP, TT, nmb_G / 1e6, levels=30,
                     cmap="RdYlGn", vmin=vmin/1e6, vmax=vmax/1e6)
    if np.nanmin(nmb_G) <= 0.0 <= np.nanmax(nmb_G):
        ax.contour(PP, TT, nmb_G, levels=[0], colors="k", linewidths=2)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("NMB ($M)", fontsize=9)
    ax.set_xlabel("Active syphilis prevalence (%)", fontsize=10)
    ax.set_ylabel("Same-day treatment rate (%)",   fontsize=10)
    ax.set_title(f"Net Monetary Benefit  |  WTP = ${wtp/1000:.0f}K/DALY\n"
                 "Black contour = break-even (NMB = 0)", fontsize=10, fontweight="bold")
    return fig

def fig_budget_bars(df_bi: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    yrs = df_bi["Year"].values
    ax.bar(yrs - 0.2, df_bi["Program cost ($)"] / 1e6, 0.4,
           label="Program cost",    color="#4a90d9", alpha=0.85)
    ax.bar(yrs + 0.2, df_bi["Outcome savings ($)"] / 1e6, 0.4,
           label="Outcome savings", color="#2a9d8f", alpha=0.85)
    ax.plot(yrs, df_bi["Cumulative net ($)"] / 1e6, "k--o",
            ms=5, lw=1.5, label="Cumulative net impact")
    ax.axhline(0, color="k", lw=0.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax.set_xlabel("Year"); ax.set_title("Annual Budget Impact", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.15)
    ax.spines[["top","right"]].set_visible(False)
    return fig

def fig_markov_states(r_disc, LE, markov: MarkovConfig, q_progress: float):
    del r_disc  # discounting does not affect state occupancy
    horizon = max(int(LE), 1)
    states = ["Healthy", "Mild sequelae", "Severe sequelae", "Dead"]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51", "#6c757d"]
    histories = [
        markov_state_occupancy(markov, q_progress, horizon, complicated=True),
        markov_state_occupancy(markov, q_progress, horizon, complicated=False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    years = np.arange(horizon)
    for ax, occupancy, title in zip(
        axes, histories, ("CS Complicated", "CS Uncomplicated")
    ):
        ax.stackplot(
            years,
            occupancy[:, 0],
            occupancy[:, 1],
            occupancy[:, 2],
            occupancy[:, 3],
            labels=states,
            colors=colors,
            alpha=0.85,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Years after birth")
        ax.grid(alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("State occupancy probability")
    axes[1].legend(loc="center right", fontsize=8, framealpha=0.7)
    fig.suptitle("Infant Markov State Occupancy (mean parameters)", fontweight="bold")
    plt.tight_layout()
    return fig

def fig_markov_daly_dist(df_psa):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    total_cs  = (df_psa["d_cs_comp"] + df_psa["d_cs_uncomp"]).replace(0, np.nan)
    per_case  = (df_psa["mk_dal"] / total_cs).dropna()
    per_case  = per_case[np.isfinite(per_case) & (per_case > 0)]
    axes[0].hist(per_case, bins=60, color="#4a90d9", alpha=0.85, edgecolor="white")
    axes[0].axvline(per_case.mean(), color="k", lw=1.5, ls="--",
                    label=f"Mean = {per_case.mean():.2f} DALYs/case")
    axes[0].set_xlabel("Lifetime DALYs averted per CS case")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Per-Case Lifetime DALY (Markov: YLD + excess YLL)", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.15)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].hist(df_psa["mk_cst"] / 1e3, bins=60, color="#2a9d8f", alpha=0.85, edgecolor="white")
    axes[1].axvline(df_psa["mk_cst"].mean()/1e3, color="k", lw=1.5, ls="--",
                    label=f"Mean = ${df_psa['mk_cst'].mean()/1e3:,.0f}K")
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    axes[1].set_xlabel("Markov lifetime cost saving per cohort")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Lifetime Cost Saving (Markov)", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.15)
    axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout(); return fig

def fig_prod_loss_breakdown(df_psa):
    """Stacked histogram of productivity-loss saving components — societal tab."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_psa["prod_sav"] / 1e6, bins=60, color="#8e44ad",
            alpha=0.80, edgecolor="white")
    ax.axvline(df_psa["prod_sav"].mean()/1e6, color="k", lw=1.5, ls="--",
               label=f"Mean = ${df_psa['prod_sav'].mean()/1e6:,.0f}M")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.set_xlabel("Productivity-loss savings averted per cohort ($M)")
    ax.set_ylabel("Frequency (PSA iterations)")
    ax.set_title("Productivity Loss Savings — Human Capital Method (PSA Distribution)",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.15)
    ax.tick_params(axis = "x", which="major", labelsize=7)
    ax.tick_params(axis = "y", which="major", labelsize=7)
    ax.spines[["top","right"]].set_visible(False)
    return fig

def fig_evppi_bar(
    evppi_results: Dict[str, float],
    evpi_total: float,
    wtp: float,
    perspective: str,
) -> plt.Figure:
    """
    Horizontal bar chart of EVPPI by parameter group.
    Bars are sorted descending. EVPI (upper bound) shown as a dashed line.
    Values shown in $K per cohort.
    """
    labels = list(evppi_results.keys())
    values = [evppi_results[k] / 1e3 for k in labels]   # → $K
    evpi_k = evpi_total / 1e3

    # Sort descending
    order  = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    n   = len(labels)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * n + 1.2)))

    colors = plt.cm.Blues(np.linspace(0.40, 0.80, n))[::-1]
    bars = ax.barh(range(n), values, color=colors, edgecolor="k",
                   alpha=0.88, height=0.6)

    # Value labels
    for bar_, val in zip(bars, values):
        ax.text(bar_.get_width() + evpi_k * 0.01, bar_.get_y() + bar_.get_height() / 2,
                f"${val:,.1f}K", va="center", fontsize=8)

    ax.axvline(evpi_k, color="crimson", lw=1.8, ls="--",
               label=f"EVPI (upper bound): ${evpi_k:,.1f}K")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("EVPPI per cohort ($K)", fontsize=10)
    ax.set_title(
        f"Expected Value of Partially Perfect Information\n"
        f"Perspective: {'Health sector' if perspective == 'hs' else 'Societal'}  |  "
        f"WTP = ${wtp/1000:.0f}K/DALY",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=9, framealpha=0.7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax.grid(alpha=0.15, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig
