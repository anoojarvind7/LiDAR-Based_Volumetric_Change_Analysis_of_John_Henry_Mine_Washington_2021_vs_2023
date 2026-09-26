# Method

This page explains how every number in `results/` is produced, what it means,
and what it does not mean.

## 1. Data

| Survey | Source | Date | Density | Used for |
|---|---|---|---|---|
| `osmre2023` | OSMRE / Surdex airborne lidar, [OpenTopography OT.112024.6339.2](https://doi.org/10.5069/G93F4MVH), CC BY 4.0 | 2023-10-06 | ~41 pts/m² | headline volumes |
| `usgs2021` | USGS 3DEP `WA_KingCo_1_2021` (QL1), Entwine Point Tiles on AWS | 2021 | ~16 pts/m² (≈2 ground pts/m² under forest) | baseline and change detection |

Everything is processed in **NAD83(2011) / UTM 10N (EPSG:6339)** with
**NAVD88** heights, the native system of the 2023 survey. The 3DEP data is
reprojected from Web Mercator by PDAL on read.

## 2. Point-cloud processing (PDAL)

The exact pipelines are written to `results/pipelines/*.json` on every run and
can be replayed with `pdal pipeline <file>`.

1. **Crop** to the survey extent (`filters.crop`).
2. **Noise removal.** Vendor noise classes 7 and 18 and withheld points are
   dropped (about 20 % of the 2023 points carry the vendor's withheld flag), then a statistical outlier filter (`filters.outlier`, k = 12,
   3 σ) removes isolated spikes the vendor missed.
3. **Ground.** The vendor's ground class (2) is used by default: it has been
   through the vendor's QA and manual editing. As an independent check, the
   whole cloud can be re-classified with the Simple Morphological Filter
   (`filters.smrf`, Pingel et al. 2013) using last returns only
   (`make smrf-check`), and volumes compared.
4. **DTM.** Ground and water returns are triangulated (`filters.delaunay`) and
   the TIN rasterised (`filters.faceraster`) at 1 m (2021) or 0.5 m (2023).
   Every cell is an exact linear interpolation between real returns, so sparse
   under-canopy areas are bridged faithfully; triangles with edges > 50 m are
   left empty rather than invented.
5. **Water.** Water returns (class 9) are rasterised, gaps closed, and each
   lake labelled. Lakes often return no signal away from the shore, so a
   no-data hole that borders exactly one lake (and not the survey edge) is
   assigned to it. Its level is the median of its returns; the DTM is then
   hydro-flattened to that level. Lidar does not see through water, so **all
   pit volumes are "above water"**; lake bathymetry is unknown.

## 3. Feature volumes

For a stockpile or dump the volume is the material above the ground it sits
on. That ground is not observable, so it is interpolated from the ground around
the toe.

**Outlines.** Each feature starts from a deliberately generous, hand-drawn
*search* rectangle (`data/features/search_outlines.geojson`). `minevol toes`
then proposes the toe automatically (`minevol.volume.refine_mask`):

1. interpolate a base surface from the rim of the current mask;
2. keep cells more than 1.0 m above it, the component holding the highest
   point plus any other large ones, and fill holes;
3. pad by 2 m so the next rim lies on ground, and repeat until stable.

The proposal is checked against the hillshade and committed as
`data/features/toes.geojson` (the committed toes were proposed from the 2023
survey; the 2021 survey, processed independently, proposes toes that overlap
them by 97–99 %). A proposed toe that runs along its search outline has been
cut off by it, so the outline is widened until the toe stops on the ground.
Where two piles share a valley (West and South stockpiles), their outlines
meet along the valley floor so each pile's base sits on that floor rather
than on its neighbour's flank. That file, not the algorithm, defines each
feature from then on, the same way a surveyor's digitised toe string does.
This makes the volumes auditable, lets anyone redraw a toe they disagree
with, and means the 2021 and 2023 volumes are measured over the same
footprint.

**Base surfaces** (`minevol.reference`):

* **harmonic** (central): solution of Laplace's equation with the rim as the
  boundary, a "soap film" spanning the toe. It reproduces any plane exactly
  and never overshoots the rim.
* **TIN**: linear interpolation on a Delaunay triangulation of the rim cells,
  the classic surveyor's toe-to-toe base.
* **thin-plate spline** (implemented, not in the default ensemble): where the
  ground rises toward the pile on every side, as it does around these graded
  dumps, a spline bows upward inside the rim and returns 15–35 % less volume.
  That is a property of the interpolator rather than evidence about the
  ground, so it is left out of the uncertainty.

**Integration.** `V = Σ max(z − base, 0) · cell area`. On a TIN-derived
DTM this converges to the exact prism volume; `tests/test_volume.py` checks
cones and paraboloids on a sloping plane against their closed-form volumes
(error < 0.5 % at 0.5–2 m cells), and the results include a resolution study
(1, 2 and 5 × the base cell).

## 4. Uncertainty

Each volume is reported with a 1 σ (`sigma_m3`) and, in the summary, a 95 %
interval. Two independent terms are combined in quadrature:

**Base surface (dominant).** The volume is recomputed with both
interpolators (harmonic, TIN) and with the toe polygon shrunk and grown by
2 m, a typical digitising tolerance. `sigma_reference_m3` is the standard
deviation of those 6 values (saved in `results/<survey>/ensemble_*.csv`).

**Where the toe is drawn is a judgement, and it is shown separately.** These
dumps were graded to blend into the surrounding ground, so their aprons fade
out over tens of metres. `minevol toes` also sweeps the toe threshold
(0.5–1.5 m) and padding (0–6 m) and saves the result
(`results/<survey>/toe_sweep_vendor.csv`, figure `*_toe_sweep.png`). Across
that sweep the volume varies by −49 % to +35 % (−10 % to +27 % for thresholds
of 0.75–1.25 m with 2–4 m padding). It rises as the rim moves
outward, because the ground falls away around each dump (they sit on spurs
and are ringed by drainage ditches), so a rim further out drags the whole base
down. The chosen toe follows the slope break visible in the hillshade. The
sweep is published rather than folded into the ± because it measures a
definition, not a measurement error: two surveyors who agree on the toe will
agree on the volume to within the stated ±.

**Survey error.** Random plus spatially correlated elevation error over the
footprint area *A* (Rolstad et al. 2009):
`σ_V² = σ_r²·a·A + σ_c²·A·πL²/5` for a footprint larger than the
correlation area πL², and `σ_V² = σ_r²·a·A + σ_c²·A²·(1 − q + q³/5)` with
`q = √(A/π)/L` for a smaller one. The parameters come from the variogram
of the 2021→2023 difference on stable ground when available, and from the
config otherwise. A uniform vertical bias of the survey cancels out of a
feature volume, because the base is interpolated from the same survey.

## 5. Change detection (2021 → 2023)

1. Both DTMs are put on the same 1 m grid (the 0.5 m grid is block-averaged).
2. **Co-registration:** the median difference on *stable ground* (outside all
   feature outlines + 25 m, off water, slope < 15°) is removed. It absorbs
   datum and geoid-model differences between the surveys.
3. A spherical variogram fitted to the stable-ground differences gives σ_r,
   σ_c and the correlation range L.
4. Differences smaller than the 95 % **level of detection**
   `1.96·√(σ_r² + σ_c²)` are set to zero; cut and fill are integrated over the
   whole site and inside each feature, with σ from step 3 plus the uncertainty
   of the co-registration shift itself.

`tests/test_change.py` runs this stage on synthetic surveys with a known
datum shift, fill and cut. It recovers the shift to 1 cm and the net change
to within 150 m³. Over a whole site the gross cut and fill totals each carry
a small positive bias, because about 5 % of unchanged cells exceed a 95 %
threshold by chance; per-feature totals and the net are barely affected.

## 6. What these numbers are not

* **Not the mine's total excavation.** There is no pre-mining (pre-1986) lidar,
  so the void of the pits relative to the original topography cannot be
  measured directly. A pit-lake volume would additionally need bathymetry.
* **Not tonnage.** Converting m³ to tonnes needs a bulk density
  (≈1.8–2.1 t/m³ for loose overburden spoil), which is outside the scope of the
  lidar.
* **Vegetated ground.** Reclaimed slopes carry grass and young trees. Where
  ground returns are sparse the TIN is flatter than reality; the ground-density
  raster (`data/processed/*_ground_density.tif`) shows where that applies.
