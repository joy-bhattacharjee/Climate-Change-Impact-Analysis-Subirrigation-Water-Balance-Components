# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
================================================================================
  Climate-Change Impact Analysis: Subirrigation & Water-Balance Components
================================================================================

Description
-----------
This script analyses growing-season water-balance components (precipitation,
subirrigation demand, evapotranspiration, drain discharge, surface runoff, and
groundwater flow) under historical and future climate scenarios (SSP2-4.5 and
SSP5-8.5) derived from CMIP6 climate model output.

The analysis produces three publication-quality figures:

  Figure 1  Multi-component cumulative seasonal ensemble (all scenarios)
  Figure 2  Subirrigation deep-dive: annual totals, intra-block trends,
            and anomaly boxplots vs. historical baseline
  Figure 3  Growing-season variance distributions for key water-balance
            components, per scenario (one figure per component)

Input data format
-----------------
Tab-separated log files (*.txt) produced by a SWAP/HYDRUS-style field-scale
water-balance model. Each file contains hourly flux columns (m³ h⁻¹) whose
names follow a prefix convention:

  oPr*   — precipitation fluxes
  sIrr*  — subirrigation input fluxes
  sEt*   — soil evapotranspiration fluxes
  oEv*   — open-water / pond evaporation fluxes
  sDra_0, sDra_1 — tile-drain discharge
  sDra_2, sDra_3 — deep groundwater exchange
  sDi*, oDi*     — surface runoff / diversion fluxes

Fluxes are divided by the field area (AREA_M2, m²) to convert to mm h⁻¹.

File-naming conventions
-----------------------
Baseline (ERA5-driven):
  <BASELINE_DIR>/log_all_Subirrigation.txt

Climate-model periods (preferred, one file per period):
  <BASE_DIR>/<MODEL>/<SCENARIO>/log_<PERIOD_LABEL>_Subirrigation.txt

Fallback (single long-run file, period sliced from it):
  <BASE_DIR>/<MODEL>/<SCENARIO>/log_all_Subirrigation.txt

Required Python packages
------------------------
  numpy >= 1.24
  pandas >= 2.0
  matplotlib >= 3.7
  scipy >= 1.10

Usage
-----
  1. Edit the "USER CONFIGURATION" section below (Section 1) to point to
     your data directory, set the field area, and define the scenario periods.
  2. Run:  python climate_subirrigation_analysis.py
  3. Figures are saved as 300 dpi PNG + vector PDF in SAVE_DIR.

Citation
--------
If you use this script, please cite the associated journal article (see README).

License
-------
MIT License — see LICENSE file in the repository root.
================================================================================
"""

import os
import json
import warnings
import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress, mannwhitneyu


# =============================================================================
# HELPER — Cohen's d effect size (pure NumPy, no extra dependency)
# =============================================================================

def cohens_d(a, b):
    """
    Compute pooled Cohen's d between two 1-D arrays.

    Parameters
    ----------
    a, b : array-like
        Sample arrays to compare.

    Returns
    -------
    float
        Cohen's d (positive when mean(a) > mean(b)), or NaN if either
        array has fewer than 2 observations.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_sd = np.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
        / (n1 + n2 - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else np.nan


# =============================================================================
# SECTION 1 — USER CONFIGURATION  ← edit this section for your dataset
# =============================================================================

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

# Directory that contains the ERA5-driven baseline log file.
# The baseline file must be named:  log_all_Subirrigation.txt
BASELINE_DIR = r"path/to/your/baseline/ERA5"

# Root directory for climate-model output.
# Expected sub-tree:  <BASE_DIR>/<MODEL_NAME>/<SCENARIO>/
BASE_DIR = r"path/to/your/model/output"

# Directory where figures will be saved (created automatically).
SAVE_DIR = r"path/to/output/figures"

# Fall-back filename used when period-specific log files are absent.
FALLBACK_LOG_FILENAME = "log_all_Subirrigation.txt"

# ------------------------------------------------------------------
# Field parameters
# ------------------------------------------------------------------

# Total simulated field area in m² (used to convert m³ fluxes → mm).
AREA_M2 = 768          # <-- replace with your field area

# Growing-season definition (month numbers, inclusive).
GROW_START_MM = 5      # May
GROW_END_MM   = 9      # September

# Minimum fraction of expected hourly time-steps in a season required
# to include that year in the analysis (0–1).  Years below this
# threshold are skipped and a warning is printed.
MIN_SEASON_COMPLETENESS = 0.90

# ------------------------------------------------------------------
# Climate models and scenarios
# ------------------------------------------------------------------

# List of CMIP6 model identifiers (must match sub-folder names in BASE_DIR).
ALL_MODELS = ["MODEL-NAME"]            # e.g. ["EC-Earth3", "MPI-ESM1-2-HR"]

# Scenarios to process.
SCENARIOS = ["historical", "ssp245", "ssp585"]

# Simulation start date for each scenario (first date in the log file).
SCENARIO_SIM_START = {
    "historical": "2000-01-01",
    "ssp245":     "2030-01-01",
    "ssp585":     "2030-01-01",
}

# Baseline ERA5-driven run start date.
BASELINE_SIM_START = "2000-01-01"

# ------------------------------------------------------------------
# Period definitions
# ------------------------------------------------------------------
# Each entry specifies a labelled analysis period within a scenario.
#   label     : short string used in figure legends and file names
#   start_yr  : first calendar year of the period (inclusive)
#   end_yr    : last  calendar year of the period (inclusive)
#   dir       : override BASE_DIR for this period (optional; set to
#               None to use BASE_DIR)

PERIODS = {
    "historical": [
        {
            "label":    "Hist-Past",
            "start_yr": 2000,
            "end_yr":   2014,
            "dir":      None,          # uses BASE_DIR
        },
    ],
    "ssp245": [
        {"label": "Near-Cent", "start_yr": 2020, "end_yr": 2039, "dir": None},
        {"label": "Mid-Cent",  "start_yr": 2040, "end_yr": 2069, "dir": None},
        {"label": "End-Cent",  "start_yr": 2070, "end_yr": 2099, "dir": None},
    ],
    "ssp585": [
        {"label": "Near-Cent", "start_yr": 2020, "end_yr": 2039, "dir": None},
        {"label": "Mid-Cent",  "start_yr": 2040, "end_yr": 2069, "dir": None},
        {"label": "End-Cent",  "start_yr": 2070, "end_yr": 2099, "dir": None},
    ],
}

# ------------------------------------------------------------------
# Output options
# ------------------------------------------------------------------
SAVE_FIGURES = True    # set False to display interactively without saving


# =============================================================================
# SECTION 2 — COLOR PALETTE (Okabe-Ito — colorblind safe)
# =============================================================================
# Eight-color Okabe-Ito set; safe for deuteranopia and protanopia.

OI = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "pink":       "#CC79A7",
}

# Mapping from period display name → fill/line color.
PERIOD_COLOR = {
    "Hist-Past":             OI["black"],
    "Near-Cent (SSP2-4.5)": OI["sky_blue"],
    "Mid-Cent (SSP2-4.5)":  OI["blue"],
    "End-Cent (SSP2-4.5)":  "#003f7f",        # darker blue
    "Near-Cent (SSP5-8.5)": OI["orange"],
    "Mid-Cent (SSP5-8.5)":  OI["vermillion"],
    "End-Cent (SSP5-8.5)":  "#7f2e00",        # darker vermillion
}

# Line styles (secondary differentiation; helps with greyscale printing).
PERIOD_LS = {
    "Hist-Past":             "-",
    "Near-Cent (SSP2-4.5)":  "--",
    "Mid-Cent (SSP2-4.5)":   "-.",
    "End-Cent (SSP2-4.5)":   ":",
    "Near-Cent (SSP5-8.5)":  "--",
    "Mid-Cent (SSP5-8.5)":   "-.",
    "End-Cent (SSP5-8.5)":   ":",
}

BL_COLOR = "#666666"   # color for the ERA5 baseline line


# =============================================================================
# SECTION 3 — WATER-BALANCE COMPONENT DEFINITIONS
# =============================================================================

# All components derived from the model output.
COMPONENTS = [
    "Precipitation",
    "Subirrigation_Input",
    "ET",
    "Drain_Discharge",
    "Surface_Runoff",
    "Groundwater_Flow",
]

# Axis labels (used in Figure 1 y-axes).
COMP_LABEL = {
    "Precipitation":       "Precipitation (mm)",
    "Subirrigation_Input": "Subirrigation demand (mm)",
    "ET":                  "Evapotranspiration (mm)",
    "Drain_Discharge":     "Drain discharge (mm)",
    "Surface_Runoff":      "Surface runoff (mm)",
    "Groundwater_Flow":    "Groundwater flow (mm)",
}

# Short labels (used in Figure 3 y-axes).
COMP_SHORT = {
    "Precipitation":       "Precip.",
    "Subirrigation_Input": "Sub-irri.",
    "ET":                  "ET",
    "Drain_Discharge":     "Drain Q",
    "Surface_Runoff":      "Surf. runoff",
    "Groundwater_Flow":    "GW flow",
}

# Components shown in Figure 3 variance plots.
KEY_COMPONENTS = [
    "Subirrigation_Input",
    "ET",
    "Drain_Discharge",
    "Groundwater_Flow",
]

# Create output directory if it does not exist.
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# SECTION 4 — MATPLOTLIB STYLE  (journal-ready, 9 pt, single-column width)
# =============================================================================

mpl.rcParams.update({
    "font.family":        "Arial",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "legend.frameon":     False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "lines.linewidth":    1.4,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
})

FW = 7.0    # figure width in inches (≈ one journal column)


def _subfig_label(ax, letter, x=-0.12, y=1.05):
    """Add a bold '(a)', '(b)', … panel label to an axes object."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right")


def save_fig(fig, name):
    """Save figure as 300 dpi PNG and vector PDF, then close it."""
    if SAVE_FIGURES:
        fig.savefig(os.path.join(SAVE_DIR, f"{name}.png"),
                    dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(SAVE_DIR, f"{name}.pdf"),
                    bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SECTION 5 — DATA LOADING AND PREPROCESSING
# =============================================================================

def load_log(filepath, sim_start):
    """
    Load a model output log file and attach a hourly DatetimeIndex.

    Parameters
    ----------
    filepath  : str  — absolute path to the tab-separated log file.
    sim_start : str  — ISO-format start date, e.g. '2000-01-01'.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame indexed by hourly timestamps, or None if the file is
        missing or cannot be parsed.
    """
    if not os.path.isfile(filepath):
        return None
    try:
        df = pd.read_csv(filepath, sep="\t", index_col=0)
        df.index = pd.date_range(sim_start, periods=len(df), freq="h")
        return df
    except Exception as exc:
        print(f"  [load_log] Could not read {filepath}: {exc}")
        return None


def compute_components(df):
    """
    Aggregate raw model flux columns into named water-balance components.

    Fluxes are in m³ h⁻¹; dividing by AREA_M2 converts to mm h⁻¹.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw hourly model output (as returned by load_log).

    Returns
    -------
    pandas.DataFrame
        Hourly DataFrame with columns matching COMPONENTS.
    """
    wb = pd.DataFrame(index=df.index)

    wb["Precipitation"] = (
        df.filter(like="oPr").sum(axis=1) / AREA_M2
    )
    wb["Subirrigation_Input"] = (
        df.filter(like="sIrr").sum(axis=1) / AREA_M2
    )
    wb["ET"] = (
        df.filter(like="sEt").sum(axis=1) +
        df.filter(like="oEv").sum(axis=1)
    ) / AREA_M2

    drain_cols = [c for c in ["sDra_0", "sDra_1"] if c in df.columns]
    gw_cols    = [c for c in ["sDra_2", "sDra_3"] if c in df.columns]

    wb["Drain_Discharge"] = (
        df[drain_cols].sum(axis=1) if drain_cols
        else pd.Series(0.0, index=df.index)
    ) / AREA_M2

    wb["Surface_Runoff"] = (
        df.filter(like="sDi").sum(axis=1) +
        df.filter(like="oDi").sum(axis=1)
    ) / AREA_M2

    wb["Groundwater_Flow"] = (
        df[gw_cols].sum(axis=1) if gw_cols
        else pd.Series(0.0, index=df.index)
    ) / AREA_M2

    return wb


def slice_wb_by_years(wb, start_yr, end_yr):
    """
    Subset a water-balance DataFrame to a calendar-year range.

    Parameters
    ----------
    wb       : pandas.DataFrame  — hourly water-balance time series.
    start_yr : int               — first year to include.
    end_yr   : int               — last  year to include.

    Returns
    -------
    pandas.DataFrame
    """
    return wb.loc[f"{start_yr}-01-01" : f"{end_yr}-12-31 23:00:00"]


def _expected_hours(yr):
    """
    Return the number of hourly time-steps expected in the growing season
    (GROW_START_MM to GROW_END_MM) for a given calendar year.
    """
    t0 = pd.Timestamp(yr, GROW_START_MM, 1)
    t1 = pd.Timestamp(yr, GROW_END_MM, 30, 23)
    return int((t1 - t0).total_seconds() / 3600) + 1


def growing_season_cumulative(wb):
    """
    Extract within-season cumulative anomaly traces for every year.

    For each year, the trace is the running sum of hourly fluxes relative
    to the first hour of the growing season (1 May), in mm.

    Years with fewer than MIN_SEASON_COMPLETENESS of expected hourly
    observations (after dropping NaN rows) are excluded.

    Parameters
    ----------
    wb : pandas.DataFrame — hourly water-balance DataFrame.

    Returns
    -------
    dict  {year (int): pandas.DataFrame}
        Each DataFrame has the same columns as wb.  Row index is the
        intra-season integer timestep (starting at 0).
    """
    seasons = {}
    for yr in sorted(wb.index.year.unique()):
        t0  = pd.Timestamp(yr, GROW_START_MM, 1)
        t1  = pd.Timestamp(yr, GROW_END_MM, 30, 23)
        sub = wb.loc[t0:t1].copy()

        completeness = sub.notna().all(axis=1).sum() / _expected_hours(yr)
        if completeness < MIN_SEASON_COMPLETENESS:
            print(
                f"  [completeness] year {yr}: {completeness:.0%} valid "
                f"— skipped (threshold {MIN_SEASON_COMPLETENESS:.0%})."
            )
            continue

        # Cumulative anomaly relative to season start (convert m to mm ×1000)
        seasons[yr] = (sub - sub.iloc[0]) * 1000.0

    return seasons


def seasonal_totals(wb):
    """
    Compute end-of-season minus start-of-season totals (mm) for each year.

    Parameters
    ----------
    wb : pandas.DataFrame — hourly water-balance DataFrame.

    Returns
    -------
    pandas.DataFrame
        One row per year, columns matching COMPONENTS, index = Year.
        Returns an empty DataFrame if no valid seasons are found.
    """
    rows = []
    for yr in sorted(wb.index.year.unique()):
        t0  = pd.Timestamp(yr, GROW_START_MM, 1)
        t1  = pd.Timestamp(yr, GROW_END_MM, 30, 23)
        sub = wb.loc[t0:t1].dropna(how="any")

        if len(sub) / _expected_hours(yr) < MIN_SEASON_COMPLETENESS:
            continue

        delta     = (sub.iloc[-1] - sub.iloc[0]) * 1000.0
        row       = delta.to_dict()
        row["Year"] = int(yr)
        rows.append(row)

    return (
        pd.DataFrame(rows).set_index("Year")
        if rows
        else pd.DataFrame(columns=COMPONENTS)
    )


# =============================================================================
# SECTION 6 — DATA INITIALISATION
# =============================================================================
# Load the ERA5 baseline run and all climate-model scenario periods.
# For each combination of (scenario, period, model) the script first looks
# for a period-specific log file; if absent it falls back to a single
# long-run log and slices the requested year range.

# --- Baseline (ERA5-driven) ---
_baseline_path = os.path.join(BASELINE_DIR, "log_all_Subirrigation.txt")
_df_bl = load_log(_baseline_path, BASELINE_SIM_START)

if _df_bl is not None:
    baseline_wb       = compute_components(_df_bl)
    baseline_seasonal = growing_season_cumulative(baseline_wb)
    baseline_totals   = seasonal_totals(baseline_wb)
    print("  Baseline loaded successfully.")
else:
    baseline_wb = None
    baseline_seasonal = {}
    baseline_totals   = pd.DataFrame(columns=COMPONENTS)
    print("  Warning: baseline file not found — baseline will be omitted from plots.")

# --- Climate-model scenario periods ---
# all_seasonal[scenario][period_label][model_name] → dict {year: DataFrame}
# all_totals  [scenario][period_label][model_name] → DataFrame (year × component)
all_seasonal = {s: {} for s in SCENARIOS}
all_totals   = {s: {} for s in SCENARIOS}

for scenario in SCENARIOS:
    for p_cfg in PERIODS[scenario]:
        p_lbl = p_cfg["label"]
        all_seasonal[scenario][p_lbl] = {}
        all_totals[scenario][p_lbl]   = {}

        active_base_dir = p_cfg.get("dir") or BASE_DIR

        for model in ALL_MODELS:
            # Preferred: period-specific log
            period_filepath = os.path.join(
                active_base_dir, model, scenario,
                f"log_{p_lbl}_Subirrigation.txt"
            )
            # Fallback: single long-run log
            fallback_filepath = os.path.join(
                active_base_dir, model, scenario, FALLBACK_LOG_FILENAME
            )

            if os.path.isfile(period_filepath):
                target_file    = period_filepath
                sim_start_date = f"{p_cfg['start_yr']}-01-01"
                is_split_file  = True
            elif os.path.isfile(fallback_filepath):
                target_file    = fallback_filepath
                sim_start_date = SCENARIO_SIM_START[scenario]
                is_split_file  = False
            else:
                print(
                    f"  Warning: no log file found for "
                    f"{model} / {scenario} / {p_lbl}  "
                    f"(searched in {active_base_dir})"
                )
                continue

            df = load_log(target_file, sim_start_date)
            if df is None:
                continue

            full_wb = compute_components(df)

            # Slice to the requested year range if using the long-run file
            sliced_wb = (
                full_wb if is_split_file
                else slice_wb_by_years(full_wb, p_cfg["start_yr"], p_cfg["end_yr"])
            )

            if not sliced_wb.empty:
                all_seasonal[scenario][p_lbl][model] = \
                    growing_season_cumulative(sliced_wb)
                all_totals[scenario][p_lbl][model] = \
                    seasonal_totals(sliced_wb)

            print(f"  Loaded: {model} / {scenario} / {p_lbl}")


# =============================================================================
# SECTION 7 — ENSEMBLE STATISTICS HELPERS
# =============================================================================

def get_period_display_name(scenario, period_label):
    """
    Build the legend/title string for a scenario–period combination.

    Examples
    --------
    get_period_display_name("historical", "Hist-Past") → "Hist-Past"
    get_period_display_name("ssp245",     "Mid-Cent")  → "Mid-Cent (SSP2-4.5)"
    """
    if scenario == "historical":
        return period_label
    scen_name = "SSP2-4.5" if scenario == "ssp245" else "SSP5-8.5"
    return f"{period_label} ({scen_name})"


def ens_stats_p(scenario, period_label, component):
    """
    Compute ensemble mean and 10th/90th percentile seasonal traces.

    Collects individual-year seasonal traces from all models and years
    within the specified scenario–period, then stacks them into a matrix
    (rows = traces, columns = intra-season timesteps).

    Parameters
    ----------
    scenario     : str — one of SCENARIOS.
    period_label : str — period label as defined in PERIODS.
    component    : str — water-balance component name (key of COMPONENTS).

    Returns
    -------
    tuple (x, mean, p10, p90)  or  None if no data are available.
        x    : 1-D array of intra-season integer timesteps.
        mean : 1-D array of ensemble mean values (mm).
        p10  : 1-D array of 10th-percentile values (mm).
        p90  : 1-D array of 90th-percentile values (mm).
    """
    traces = []
    for model_dict in all_seasonal[scenario].get(period_label, {}).values():
        for df in model_dict.values():
            if component in df.columns:
                traces.append(df[component].values)

    if not traces:
        return None

    max_len = max(len(t) for t in traces)
    mat     = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        mat[i, :len(t)] = t

    return (
        np.arange(max_len),
        np.nanmean(mat, axis=0),
        np.nanpercentile(mat, 10, axis=0),
        np.nanpercentile(mat, 90, axis=0),
    )


def annual_ens_p(scenario, period_label, component):
    """
    Assemble ensemble annual seasonal totals (mm season⁻¹).

    Parameters
    ----------
    scenario     : str
    period_label : str
    component    : str

    Returns
    -------
    pandas.DataFrame
        Columns: mean, p10, p90.  Index: Year.
        Empty DataFrame if no data are available.
    """
    model_totals = [
        t for t in all_totals[scenario].get(period_label, {}).values()
        if not t.empty and component in t.columns
    ]
    if not model_totals:
        return pd.DataFrame()

    combined = pd.concat(model_totals, axis=0)
    rows     = []

    for yr in sorted(combined.index.unique()):
        raw  = combined.loc[yr, component]
        vals = (
            raw.values if isinstance(raw, pd.Series) else np.array([raw])
        ).astype(float)
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            continue
        rows.append({
            "Year": int(yr),
            "mean": float(np.mean(vals)),
            "p10":  float(np.percentile(vals, 10)),
            "p90":  float(np.percentile(vals, 90)),
        })

    return (
        pd.DataFrame(rows).set_index("Year")
        if rows
        else pd.DataFrame()
    )


# =============================================================================
# SECTION 8 — STATISTICAL HELPERS
# =============================================================================

def sig_stars(p_value):
    """
    Convert a p-value to a significance-star string.

    Returns
    -------
    '***'  if p < 0.001
    '**'   if p < 0.01
    '*'    if p < 0.05
    'n.s.' otherwise
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


# =============================================================================
# SECTION 9 — FIGURE 1: Multi-component growing-season ensemble
# =============================================================================

def fig1_ensemble_seasonal():
    """
    Plot cumulative growing-season anomaly traces for all six water-balance
    components across all scenario periods.

    Layout : 6 stacked subpanels (one per component), shared x-axis.
    Lines  : ensemble mean per scenario period.
    Shading: 10th–90th percentile spread.
    Dashed grey line: ERA5 baseline mean.
    """
    n_rows = len(COMPONENTS)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(FW, n_rows * 2.0),
        sharex=True,
        constrained_layout=True,
    )
    letters = "abcdef"

    for idx, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        _subfig_label(ax, letters[idx])
        legend_handles = []

        # ── Scenario period traces ──────────────────────────────────────────
        for scen in SCENARIOS:
            for p_cfg in PERIODS[scen]:
                p_lbl = p_cfg["label"]
                res   = ens_stats_p(scen, p_lbl, comp)
                if res is None:
                    continue

                x, mn, p10, p90 = res
                disp_lbl = get_period_display_name(scen, p_lbl)
                color    = PERIOD_COLOR.get(disp_lbl, "#999999")
                ls       = PERIOD_LS.get(disp_lbl, "-")

                ax.fill_between(x, p10, p90, color=color, alpha=0.10,
                                linewidth=0)
                line, = ax.plot(x, mn, color=color, lw=1.4, ls=ls,
                                label=disp_lbl)
                legend_handles.append(line)

        # ── ERA5 baseline mean ──────────────────────────────────────────────
        if baseline_seasonal:
            season_series = [
                df_s[comp].reset_index(drop=True)
                for df_s in baseline_seasonal.values()
                if comp in df_s.columns
            ]
            if season_series:
                bl_mat   = pd.concat(season_series, axis=1)
                bl_mean  = bl_mat.mean(axis=1)
                line_bl, = ax.plot(
                    np.arange(len(bl_mat)), bl_mean,
                    color=BL_COLOR, lw=1.1, ls="--", label="ERA5 Baseline"
                )
                legend_handles.append(line_bl)

        ax.set_ylabel(COMP_LABEL[comp])
        ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")

        # Add legend only on the top panel (avoid repetition)
        if idx == 0:
            seen, unique_handles = set(), []
            for h in legend_handles:
                if h.get_label() not in seen:
                    seen.add(h.get_label())
                    unique_handles.append(h)
            ax.legend(handles=unique_handles, ncol=3, loc="upper left",
                      borderpad=0.4, labelspacing=0.3)

    axes[-1].set_xlabel("Day within growing season (1 May = day 0)")
    save_fig(fig, "F1_multi_period_seasonal_components")
    print("  Figure 1 saved.")


# =============================================================================
# SECTION 10 — FIGURE 2: Subirrigation deep-dive (3 panels)
# =============================================================================

def fig2_subirrigation_deepdive():
    """
    Three-panel figure focusing on subirrigation demand.

    Panel (a) — Annual totals bar chart
        One bar per simulated year, colored by scenario period.
        Error bars show the 10th–90th percentile ensemble spread.

    Panel (b) — Intra-block linear trends
        Scatter of annual means with fitted regression lines,
        colored and styled by scenario period.

    Panel (c) — Anomaly boxplots vs. historical baseline
        Distributions of future seasonal anomalies (relative to the
        Hist-Past ensemble mean).  Significance of difference from the
        historical reference is assessed by the Mann-Whitney U test.
    """
    comp = "Subirrigation_Input"

    fig = plt.figure(figsize=(FW * 1.6, 7.0), constrained_layout=True)
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    for ax, letter in zip([ax_a, ax_b, ax_c], "abc"):
        _subfig_label(ax, letter)

    # ── (a) Annual totals bar chart ─────────────────────────────────────────
    bar_width  = 0.6
    x_position = 0
    tick_pos   = []
    tick_lbls  = []

    for scen in SCENARIOS:
        for p_cfg in PERIODS[scen]:
            p_lbl  = p_cfg["label"]
            df_ann = annual_ens_p(scen, p_lbl, comp)
            if df_ann.empty:
                continue

            disp_lbl = get_period_display_name(scen, p_lbl)
            color    = PERIOD_COLOR.get(disp_lbl, "#999999")

            for yr in df_ann.index:
                mean_val = df_ann.loc[yr, "mean"]
                ax_a.bar(x_position, mean_val, width=bar_width,
                         color=color, alpha=0.80, linewidth=0)
                ax_a.errorbar(
                    x_position, mean_val,
                    yerr=[[mean_val - df_ann.loc[yr, "p10"]],
                          [df_ann.loc[yr, "p90"] - mean_val]],
                    fmt="none", color="#555555", lw=0.7, capsize=1.5,
                )
                x_position += 1

            tick_pos.append(x_position - len(df_ann) / 2)
            tick_lbls.append(p_lbl)
            x_position += 1   # gap between period groups

    ax_a.set_xticks(tick_pos)
    ax_a.set_xticklabels(tick_lbls, rotation=20, ha="right")
    ax_a.set_ylabel("Subirrigation (mm season⁻¹)")

    # ── (b) Intra-block trends ──────────────────────────────────────────────
    for scen in SCENARIOS:
        for p_cfg in PERIODS[scen]:
            p_lbl  = p_cfg["label"]
            df_ann = annual_ens_p(scen, p_lbl, comp)
            if df_ann.empty or len(df_ann) < 3:
                continue

            years  = df_ann.index.values.astype(float)
            values = df_ann["mean"].values
            disp_lbl = get_period_display_name(scen, p_lbl)
            color    = PERIOD_COLOR.get(disp_lbl, "#999999")
            ls       = PERIOD_LS.get(disp_lbl, "-")

            slope, intercept, *_ = linregress(years, values)
            ax_b.plot(years, values, "o", color=color, ms=2.5, zorder=3)
            ax_b.plot(years, intercept + slope * years, ls=ls,
                      color=color, lw=1.1, zorder=2)

    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("Subirrigation (mm season⁻¹)")

    # ── (c) Anomaly boxplots ────────────────────────────────────────────────
    # Collect historical reference values
    hist_ref = []
    for model_totals in all_totals["historical"].get("Hist-Past", {}).values():
        if not model_totals.empty and comp in model_totals.columns:
            hist_ref.extend(model_totals[comp].dropna().tolist())

    hist_mean = np.mean(hist_ref) if hist_ref else 0.0

    anom_data = []
    anom_lbls = []
    anom_cols = []

    for scen in ["ssp245", "ssp585"]:
        for p_cfg in PERIODS[scen]:
            p_lbl   = p_cfg["label"]
            fut_vals = []
            for model_totals in all_totals[scen].get(p_lbl, {}).values():
                if not model_totals.empty and comp in model_totals.columns:
                    fut_vals.extend(model_totals[comp].dropna().tolist())
            if fut_vals:
                anom_data.append(np.array(fut_vals) - hist_mean)
                disp_name = get_period_display_name(scen, p_lbl)
                anom_lbls.append(disp_name)
                anom_cols.append(PERIOD_COLOR.get(disp_name, "#999999"))

    if anom_data:
        bp = ax_c.boxplot(
            anom_data,
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="white", lw=1.5),
            whiskerprops=dict(lw=0.7, color="#555555"),
            capprops=dict(lw=0.7, color="#555555"),
            boxprops=dict(linewidth=0.5),
            flierprops=dict(marker="o", markersize=2, alpha=0.35,
                            markeredgewidth=0),
        )
        for patch, col in zip(bp["boxes"], anom_cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.70)

        ax_c.set_xticks(range(1, len(anom_lbls) + 1))
        ax_c.set_xticklabels(anom_lbls, rotation=35, ha="right")
        ax_c.axhline(0, color="#555555", lw=0.7, ls="--")
        ax_c.set_ylabel("Δ subirrigation (mm season⁻¹)")

        # Significance stars (Mann-Whitney U vs. historical reference)
        if hist_ref and len(hist_ref) > 1:
            y_top = ax_c.get_ylim()[1]
            pad   = (ax_c.get_ylim()[1] - ax_c.get_ylim()[0]) * 0.05
            for i, anom_vals in enumerate(anom_data):
                raw_future = anom_vals + hist_mean
                if len(raw_future) < 2:
                    continue
                _, p_val = mannwhitneyu(
                    hist_ref, raw_future, alternative="two-sided"
                )
                ax_c.text(
                    i + 1, y_top + pad,
                    sig_stars(p_val),
                    ha="center", va="bottom", fontsize=8, color="#333333",
                )

    save_fig(fig, "F2_subirrigation_deepdive")
    print("  Figure 2 saved.")


# =============================================================================
# SECTION 11 — FIGURE 3: Variance distributions (KEY_COMPONENTS)
# =============================================================================

def fig3_variance_blocks():
    """
    For each key water-balance component, produce a three-panel figure
    (one panel per scenario) showing growing-season total distributions
    as boxplots.

    Statistical comparison
    ----------------------
    Within each scenario, periods after the first are compared to the
    first period using the Mann-Whitney U test.  Significance stars are
    annotated above the relevant box.

    One PNG + PDF file is saved per key component.
    """
    scenario_labels = {
        "historical": "Historical",
        "ssp245":     "SSP2-4.5",
        "ssp585":     "SSP5-8.5",
    }

    for comp in KEY_COMPONENTS:
        fig, axes = plt.subplots(
            1, len(SCENARIOS),
            figsize=(FW * 1.4, 3.2),
            sharey=True,
            constrained_layout=True,
        )

        for i, scen in enumerate(SCENARIOS):
            ax = axes[i]
            _subfig_label(ax, "abc"[i])

            data_list = []
            labels    = []
            colors    = []

            for p_cfg in PERIODS[scen]:
                p_lbl = p_cfg["label"]
                vals  = []
                for model_totals in all_totals[scen].get(p_lbl, {}).values():
                    if not model_totals.empty and comp in model_totals.columns:
                        vals.extend(model_totals[comp].dropna().tolist())

                if vals:
                    data_list.append(vals)
                    labels.append(p_lbl)
                    colors.append(
                        PERIOD_COLOR.get(
                            get_period_display_name(scen, p_lbl), "#999999"
                        )
                    )

            if data_list:
                bp = ax.boxplot(
                    data_list,
                    patch_artist=True,
                    widths=0.45,
                    medianprops=dict(color="white", lw=1.5),
                    whiskerprops=dict(lw=0.7, color="#555555"),
                    capprops=dict(lw=0.7, color="#555555"),
                    boxprops=dict(linewidth=0.5),
                    flierprops=dict(marker="o", markersize=2, alpha=0.35,
                                   markeredgewidth=0),
                )
                for patch, col in zip(bp["boxes"], colors):
                    patch.set_facecolor(col)
                    patch.set_alpha(0.65)

                # Significance annotations vs. first period
                if len(data_list) > 1 and len(data_list[0]) > 1:
                    y_top = ax.get_ylim()[1]
                    pad   = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                    for j in range(1, len(data_list)):
                        if len(data_list[j]) < 2:
                            continue
                        _, p_val = mannwhitneyu(
                            data_list[0], data_list[j],
                            alternative="two-sided",
                        )
                        ax.text(
                            j + 1, y_top + pad,
                            sig_stars(p_val),
                            ha="center", va="bottom",
                            fontsize=8, color="#333333",
                        )

                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=15, ha="right")

            ax.set_title(scenario_labels.get(scen, scen))

        axes[0].set_ylabel(f"{COMP_SHORT[comp]}  (mm season⁻¹)")
        save_fig(fig, f"F3_variance_blocks_{comp}")
        print(f"  Figure 3 saved: {comp}")


# =============================================================================
# SECTION 12 — RUN METADATA EXPORT
# =============================================================================

def export_run_metadata():
    """
    Write a JSON file recording the run configuration and library versions.
    Useful for reproducing figures and for archiving alongside published data.
    """
    import sys

    meta = {
        "run_timestamp":      datetime.datetime.now().isoformat(),
        "python_version":     sys.version,
        "numpy_version":      np.__version__,
        "pandas_version":     pd.__version__,
        "matplotlib_version": mpl.__version__,
        "configuration": {
            "AREA_M2":                 AREA_M2,
            "GROW_START_MM":           GROW_START_MM,
            "GROW_END_MM":             GROW_END_MM,
            "MIN_SEASON_COMPLETENESS": MIN_SEASON_COMPLETENESS,
            "ALL_MODELS":              ALL_MODELS,
            "SCENARIOS":               SCENARIOS,
            "PERIODS":                 PERIODS,
        },
    }

    meta_path = os.path.join(SAVE_DIR, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Run metadata saved → {meta_path}")


# =============================================================================
# SECTION 13 — MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Climate-Change Impact Analysis — Subirrigation & Water Balance")
    print(f"  Models    : {ALL_MODELS}")
    print(f"  Scenarios : {SCENARIOS}")
    print(f"  Save dir  : {SAVE_DIR}")
    print("=" * 70)

    fig1_ensemble_seasonal()
    fig2_subirrigation_deepdive()
    fig3_variance_blocks()

    export_run_metadata()

    print()
    print("=" * 70)
    print("  Done.  All figures exported as 300 dpi PNG + vector PDF.")
    print(f"  Output directory: {SAVE_DIR}")
    print("=" * 70)