# John Henry Mine lidar volumes

Volumes of the spoil dumps and stockpiles at the **John Henry No. 1 Mine**, a
reclaimed surface coal mine near Black Diamond, Washington, measured from
airborne lidar point clouds, with an uncertainty on every number.

The pipeline goes from raw points to volumes: it fetches the point clouds,
removes noise, classifies ground, builds a TIN terrain model, detects and
flattens the pit lakes, finds each pile's toe, interpolates the ground
beneath it, integrates the volume and propagates the errors. It also
differences two surveys to measure change. Every step is scripted, tested
against shapes with known volumes, and reproducible with one command.

## Results

![Site overview with the four measured features](results/figures/usgs2021_overview.png)

Volume above the reconstructed ground, from the 2021 USGS 3DEP survey. The ±
is a 95 % interval covering the base-surface interpolator, a ±2 m toe
tolerance and survey error. Full tables are in
[`results/SUMMARY.md`](results/SUMMARY.md).

| Feature | Volume (m³) | ± 95 % (m³) | Footprint (ha) | Max height (m) |
|---|---:|---:|---:|---:|
| North spoil dump | 3,046,000 | 252,000 | 20.7 | 35.7 |
| South stockpile | 1,269,000 | 110,000 | 8.8 | 33.5 |
| NE ridge fill | 1,153,000 | 129,000 | 10.0 | 24.6 |
| West stockpile | 898,000 | 83,000 | 6.7 | 33.8 |
| **Total** | **6,367,000** | **315,000** | 46.3 | |

* **Grid independence:** coarsening the DTM from 1 m to 2 m changes each
  volume by 0.7–1.6 %.
* **Toe placement matters more than anything else.** Moving the toe across
  plausible positions changes volumes by up to ±20 %; see the toe sweep in
  `results/SUMMARY.md` and the discussion in `docs/method.md`.
* **Pit lake:** the main pit is flooded, a 10.4 ha lake at 230.52 m NAVD88.
  Lidar does not penetrate water, so its submerged volume is not measurable
  from these data.

> **Status:** the 2023 OSMRE survey (41 pts/m², the primary dataset) and the
> 2021→2023 change analysis are wired up and tested on synthetic data. They
> run as soon as the 2023 point cloud is downloaded (`minevol fetch
> osmre2023`), and these tables will then be regenerated from it.

## Quick start

```bash
conda env create -f environment.yml     # PDAL + Python stack from conda-forge
conda activate minevol
make test                               # analytic volume tests, ~5 s
make all                                # fetch → process → volumes → change → report
```

`make all` downloads roughly 1.5 GB of point cloud and takes 30–60 minutes on
a laptop. Outputs go to `results/`, which is committed so you can read the
results without running anything; `results/SUMMARY.md` is the generated
report.

Stages can be run on their own:

```bash
minevol fetch usgs2021        # USGS 3DEP 2021, streamed from AWS (Entwine Point Tiles)
minevol fetch osmre2023       # OSMRE 2023 survey from OpenTopography's bulk bucket
minevol process osmre2023     # clean, classify, TIN DTM, water flattening
minevol toes osmre2023        # propose toe polygons from the search outlines (review, then commit)
minevol volumes osmre2023     # feature volumes + uncertainty ensemble
minevol change                # 2021 → 2023 cut/fill with level of detection
minevol report                # figures + results/SUMMARY.md
make smrf-check               # re-classify ground with SMRF and compare
```

## How it works

```
LAS/LAZ or EPT ─► crop + reproject ─► noise filter ─► ground (vendor | SMRF)
      (PDAL)            EPSG:6339        classes 7/18,      │
                                         statistical        ▼
                                         outlier       Delaunay TIN ─► DTM raster
                                                            │         (faceraster)
                              water returns (class 9) ─► lake extent + level ─► hydro-flatten
                                                            │
   search outline ─► toe detection ─► reviewed toe polygon ─► base surface ─► ∫(z − base) dA
                     (iterative, harmonic)                   (harmonic | TIN)       │
                                                                                    ▼
                               ensemble over base method × toe tolerance + survey error ─► V ± σ
```

The full method, with the reasoning behind each choice, is in
[`docs/method.md`](docs/method.md). The short version:

* **DTM from a TIN, not binned points.** Ground returns are triangulated and
  the TIN is rasterised, so every cell is an exact interpolation between real
  returns, including under canopy.
* **Base surface by harmonic interpolation.** The ground under a pile is
  reconstructed by solving Laplace's equation inside the toe with the
  surrounding terrain as boundary. It reproduces any sloping plane exactly
  and never overshoots the rim. A classic toe-to-toe TIN is computed
  alongside it.
* **Toes proposed automatically, then committed.** A deliberately loose
  outline is shrunk iteratively to the slope break. The resulting polygon
  (`data/features/toes.geojson`) is reviewed against the hillshade and becomes
  the auditable definition of each feature.
* **Uncertainty with two terms.** One is the base-surface ensemble
  (interpolator × ±2 m toe tolerance). The other is survey error propagated
  with spatial correlation (Rolstad et al. 2009), with parameters taken from a
  variogram of the 2021→2023 difference on stable ground.
* **Verified on shapes with known volumes.** Cones and paraboloids on sloping
  ground are recovered to within 0.5 % (`tests/`), and a synthetic change
  test recovers a known datum shift, fill and cut.

## Repository layout

```
config/john_henry.yaml     all site, survey and method parameters
data/features/             search outlines and reviewed toe polygons (GeoJSON, EPSG:6339)
src/minevol/
  pointcloud.py            PDAL pipelines: fetch (EPT, S3), clean, SMRF, TIN DTM
  water.py                 lake extent/level from water returns, hydro-flattening
  reference.py             base surfaces: harmonic, TIN, thin-plate spline
  volume.py                cut/fill integration, toe detection
  uncertainty.py           error propagation, co-registration, variograms
  workflow.py, cli.py      stages and the `minevol` command
  report.py                figures and results/SUMMARY.md
results/                   generated tables, figures and the exact PDAL pipelines used
tests/                     analytic and synthetic end-to-end tests
docs/method.md             method, assumptions and limitations
```

## Data and credits

* **2023:** *Lidar Survey of John Henry Mine, WA 2023*, U.S. DOI Office of
  Surface Mining Reclamation and Enforcement, acquired by Surdex, distributed
  by OpenTopography. <https://doi.org/10.5069/G93F4MVH> (CC BY 4.0).
* **2021:** USGS 3D Elevation Program, *WA_KingCo_1_2021*, via the
  [USGS 3DEP Entwine Point Tiles](https://registry.opendata.aws/usgs-lidar/)
  public dataset on AWS.

Code is MIT-licensed (see `LICENSE`).
