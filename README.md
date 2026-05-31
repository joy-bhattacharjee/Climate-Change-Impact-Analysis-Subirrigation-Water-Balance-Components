# Climate-Change Impact Analysis: Subirrigation & Water-Balance Components

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Analysis code accompanying the journal article:

> **[Article title]**  
> *[Author list]*  
> *[Journal name]*, [Year]. DOI: [https://doi.org/...]

---

## Overview

This repository contains the Python script used to generate all publication
figures analysing growing-season water-balance components under historical and
future (SSP2-4.5, SSP5-8.5) CMIP6 climate scenarios.

The script produces three figure types:

| Figure | Description |
|--------|-------------|
| **F1** | Cumulative growing-season anomaly traces for six water-balance components across all scenario periods |
| **F2** | Subirrigation deep-dive: annual totals, intra-block linear trends, and anomaly boxplots vs. historical baseline |
| **F3** | Growing-season variance distributions (boxplots) for key components, per scenario |

---

## Repository structure

```
.
├── climate_subirrigation_analysis.py   ← main analysis script
├── README.md
└── LICENSE
```

---

## Requirements

| Package | Minimum version |
|---------|----------------|
| Python  | 3.9 |
| NumPy   | 1.24 |
| pandas  | 2.0 |
| Matplotlib | 3.7 |
| SciPy   | 1.10 |

Install all dependencies with:

```bash
pip install numpy pandas matplotlib scipy
```

---

## Input data format

The script reads **tab-separated hourly model output files** (`.txt`) produced
by a SWAP/HYDRUS-style field-scale water-balance model.

### Column naming convention

| Prefix pattern | Water-balance flux |
|----------------|--------------------|
| `oPr*`         | Precipitation |
| `sIrr*`        | Subirrigation input |
| `sEt*`         | Soil evapotranspiration |
| `oEv*`         | Open-water / pond evaporation |
| `sDra_0`, `sDra_1` | Tile-drain discharge |
| `sDra_2`, `sDra_3` | Deep groundwater exchange |
| `sDi*`, `oDi*` | Surface runoff / diversion |

All fluxes are in **m³ h⁻¹**.  The script divides by `AREA_M2` (m²) to
convert to **mm h⁻¹**.

### Expected directory tree

```
<BASE_DIR>/
└── <MODEL_NAME>/
    ├── historical/
    │   └── log_Hist-Past_Subirrigation.txt   (or log_all_Subirrigation.txt)
    ├── ssp245/
    │   ├── log_Near-Cent_Subirrigation.txt
    │   ├── log_Mid-Cent_Subirrigation.txt
    │   └── log_End-Cent_Subirrigation.txt
    └── ssp585/
        ├── log_Near-Cent_Subirrigation.txt
        ├── log_Mid-Cent_Subirrigation.txt
        └── log_End-Cent_Subirrigation.txt

<BASELINE_DIR>/
└── log_all_Subirrigation.txt    ← ERA5-driven baseline run
```

If a period-specific file is absent the script automatically falls back to
`log_all_Subirrigation.txt` and slices the requested year range.

---

## Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-org>/<your-repo>.git
   cd <your-repo>
   ```

2. **Edit the configuration block** (`SECTION 1 — USER CONFIGURATION`) at
   the top of `climate_subirrigation_analysis.py`:

   ```python
   # Paths
   BASELINE_DIR = r"path/to/your/baseline/ERA5"
   BASE_DIR     = r"path/to/your/model/output"
   SAVE_DIR     = r"path/to/output/figures"

   # Field parameters
   AREA_M2 = 768           # your field area in m²

   # Models
   ALL_MODELS = ["EC-Earth3"]   # list of CMIP6 model identifiers

   # Period year ranges — adjust to match your simulation files
   PERIODS = { ... }
   ```

3. **Run the script**

   ```bash
   python climate_subirrigation_analysis.py
   ```

4. **Outputs** are written to `SAVE_DIR`:

   | File | Description |
   |------|-------------|
   | `F1_multi_period_seasonal_components.png/.pdf` | Figure 1 |
   | `F2_subirrigation_deepdive.png/.pdf` | Figure 2 |
   | `F3_variance_blocks_<component>.png/.pdf` | Figure 3 (one per component) |
   | `run_metadata.json` | Run timestamp, library versions, full configuration |

---

## Key design choices

### Colorblind-safe palette
All figures use the **Okabe-Ito** 8-color set, which is safe for
deuteranopia and protanopia.  Line styles additionally distinguish periods
to support greyscale printing.

### Growing-season completeness filter
Years with fewer than `MIN_SEASON_COMPLETENESS` (default: 90 %) of expected
hourly observations are excluded from all analyses.  A warning is printed for
each excluded year.

### Statistical tests
- **Mann-Whitney U** (non-parametric) is used to compare future period
  distributions to the historical reference.
- Significance is annotated as `*` (p < 0.05), `**` (p < 0.01),
  `***` (p < 0.001), or `n.s.`.

### Ensemble statistics
- Seasonal traces: ensemble **mean** ± 10th/90th percentile shading.
- Annual totals: **mean** with p10–p90 error bars.

---

## License

This code is released under the [MIT License](LICENSE).

---

## Contact

For questions about the code please open an issue in this repository.  
For questions about the underlying model or data please contact the
corresponding author (see the journal article).
