# LA-EWS

Code and processed data for *Land-Atmosphere Coupling as an Early Warning for Critical Vegetation Transition* (Lu, Yang, He, Fan and Chen).

Three regions are analysed: the Amazon (AM), the North American boreal forest (NB) and the Tibetan Plateau (TP). Early warning signals (EWS) are computed from three land–atmosphere coupling indices (SM–ET, T–ET and Rn–λE–Ts) and from NDVI, and the lead of the land–atmosphere EWS (LA-EWS) over vegetation EWS is tested with lagged correlation and convergent cross mapping (CCM).

## Requirements

Python 3.10 and the packages in `requirements.txt`:

```shell
python -m pip install -r requirements.txt
```

## Usage

```shell
python run.py
```

This recomputes the lag correlations of Fig. 4, Tables S5–S7 and the spatial statistics, and compares them, together with the saved CCM and trend-test results, with the values reported in the paper. Results are written to `outputs/`.

Individual steps:

| Command | Output |
|---|---|
| `python code/recompute_ews.py` | Component EWS, LA-EWS and the six Fig. 4 pairs computed from the coupling residuals |
| `python code/figure4.py --resample --plot` | Fig. 4, Tables S5–S7, block-bootstrap null tests and bootstrap lag ranges |
| `python code/run_temporal.py` | Table S3 (trend tests of the Fig. 2 segments) |
| `python code/plot_figure5.py` | Fig. 5 from the saved CCM curves |
| `python code/spatial.py --plot` | Spatial τ maps with Benjamini–Yekutieli adjustment (Fig. 7, Figs. S7–S8, Table S4) |
| `python code/run_ccm.py` | Full CCM run, 120 mappings × 1,000 surrogates (several hours); `--group AM_DFA --rank 1` runs a single mapping |
| `python code/run_monthly_ccm.py` | CCM between monthly residuals (24 mappings) |
| `python code/run_model.py` | Minimal model, Figs. S9–S10 |

## Methods in brief

- **Residuals**: STL decomposition of the monthly regional series (settings in `parameters/components.json`).
- **AR1**: lag-1 Pearson correlation in 120-month rolling windows.
- **DFA**: `fathon.DFA` applied to the residual window, scales 10–120, forward and reverse segmentation, third-order detrending; the exponent is the slope of log F(n) against log n.
- **LA-EWS**: mean of the SM–ET, T–ET and Rn–λE–Ts EWS (EWS are computed for each index first, then averaged).
- **Lead–lag test**: the maximum Pearson correlation over the lag grid is compared with stationary block-bootstrap surrogates of the lagged LA-EWS (mean block length n/10, n/5 and n/3; 1,000 draws each), with the lag search repeated for every surrogate; P = (k + 1)/1,001.
- **CCM**: simplex cross mapping as in pyEDM 2.5.0, 30 random libraries per library size, 1,000 phase-randomized surrogates of the target series.

## Data

| Folder | Content |
|---|---|
| `data/residuals/` | STL residuals of the coupling indices, `{AM,NB,TP}_{tci,tet,pai}.npy` (tci = SM–ET, tet = T–ET, pai = Rn–λE–Ts) |
| `data/la_ews/` | LA-EWS series, keys `{region}_{DFA,AR1}` |
| `data/ndvi/` | AM monthly NDVI and its EWS (`AM_monthly.csv`, `AM_ews.csv`, 1982–2022); NB and TP NDVI residuals |
| `data/figure4/` | Fig. 4 input pairs and source data |
| `data/tables/` | Inputs and results for Tables S5–S7, bootstrap distributions |
| `data/ccm/` | CCM results for the 120 mappings, including the 1,000 surrogate curves of each mapping |
| `data/monthly_ccm/` | Inputs and results of the monthly-residual CCM |
| `data/temporal/` | Fig. 2 rising segments and Table S3 results |
| `data/spatial/` | Pixel-level Kendall τ and P values and GeoTIFF maps (EPSG:4326) for the Amazon |
| `data/boundaries/` | Region boundaries and raster masks |
| `models/minimal/` | Minimal model script, parameters and outputs |
| `source_scripts/` | Original scripts for the coupling indices, regional series and spatial maps; they read the raw gridded data from local paths |

`data/figure4/inputs.npz` contains, for each group `AM_DFA`, `AM_AR1`, `NB_DFA`, `NB_AR1`, `TP_DFA` and `TP_AR1`:

- `{group}_y`: NDVI EWS, 240 values;
- `{group}_X`: lagged LA-EWS, 240 × number of lags, with `X[i, j] = LA[A + i − lag[j]]` and A given in `parameters/alignment.json`;
- `{group}_lags_months`: lags in months; positive values mean that LA-EWS leads NDVI-EWS.

Boundaries (`data/boundaries/`, WGS84): `AM.geojson`, Amazon rainforest (the AM NDVI series is the mean over its bounding rectangle); `NB.geojson`, North American boreal zone (Brandt, J. P. 2009, *Environmental Reviews* 17, 101–161; simplified); `TP.geojson`, Tibetan Plateau (Zhang, Y., Li, B. & Zheng, D., https://doi.org/10.3974/geodb.2014.01.12.V1). `masks/` contains the raster masks used to extract the regional series.

Monte Carlo P values in the saved results are (1 + number of surrogate statistics ≥ observed)/1,001.

The raw satellite and reanalysis data (GIMMS NDVI, MODIS, VODCA, ERA5-Land, CRU scPDSI) are not included; sources are listed in the Data Availability section of the paper.
