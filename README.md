# Disaggregation — Fine-grained Property Value Assessment

Code for Archbold et al., **Fine-grained Property Value Assessment Using Probabilistic Disaggregation** (IGARSS 2023).
Paper: <https://arxiv.org/pdf/2306.00246.pdf>

The model takes a 1m-resolution aerial RGB tile of Hennepin County, Minnesota plus
parcel-level boundary masks and learns a per-pixel probabilistic value field whose
parcel-aggregated values match the ground-truth property assessments.

```bibtex
@inproceedings{archbold2023fine,
  author    = {Archbold, Cohen and Brodie, Benjamin and Ogholbake, Aram Ansary and Jacobs, Nathan},
  booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  title     = {Fine-grained Property Value Assessment Using Probabilistic Disaggregation},
  year      = {2023},
}
```

## Repo layout

| Directory | Purpose |
|---|---|
| `aggregation/` | Main pipeline. PyTorch-Lightning modules for the 5 model variants (`ral`, `uniform`, `rsample`, `gauss`, `logsample`), Hennepin dataset loader, train/test/vis entrypoints. |
| `segmentation/` | Optional building-segmentation pre-training that produces the checkpoint `aggregation/` loads when `cfg.train.use_pretrained=True`. |
| `synthetic_testing/` | Self-contained controlled experiment using EuroSAT with synthetic labels. See [`synthetic_testing/README.md`](synthetic_testing/README.md). |
| `dataset_generation/` | Original pipeline that turned the raw Hennepin County parcels shapefile + WMS imagery into the `ver8/`-style per-tile layout. |
| `scripts/` | Setup helpers (`setup_hennepin.sh`). |
| `paths.env` | Single source of truth for filesystem paths. Edit this once; both Python configs and the setup script read it. |
| `paths.py` | Tiny stdlib loader that merges `paths.env` into `os.environ` at import time. |

## Setup

1.  **Install dependencies.** A Python 3.9+ environment is fine; CUDA strongly recommended for training.

    ```bash
    pip install -r requirements.txt
    ```

    The original cluster lockfile is preserved at `aggregation/legacy_py39_cluster.yml` for reference but is not needed.

2.  **Configure paths once in `paths.env`.** Open the file at the repo root and fill in
    the values you care about. The defaults work if you keep the standard layout
    (Box-synced archives + repo-relative `data/` directory). All path-shaped variables
    flow from this file: both Python `config.py` files and `scripts/setup_hennepin.sh`
    read it. Anything you export in your shell takes precedence.

3.  **Get the data.** The Hennepin aggregation dataset ships as two ~455 MB zips on Box:

    ```
    1m_302px.zip                  → 1916 tiles, uncombined parcels
    1m_302px_region_combined.zip  → 1916 tiles, parcels merged with neighbors
    ```

    Both files live in Box at `<Box>/datasets/Hennepin_dataset/`. If your local Box mount
    shows them as 0-byte placeholders, right-click each in Finder → **Download Now** and wait
    for the sync to finish.

4.  **Unpack.** From the repo root:

    ```bash
    bash scripts/setup_hennepin.sh
    ```

    The script reads `BOX_DIR` and `HENNEPIN_DATA_ROOT` from `paths.env` (or from your
    shell) and unpacks each zip to its own subdirectory under `HENNEPIN_DATA_ROOT`:

    ```
    $HENNEPIN_DATA_ROOT/1m_302px/{imgs,masks,vals.pkl}
    $HENNEPIN_DATA_ROOT/1m_302px_region_combined/{imgs,masks,vals.pkl}
    ```

    If you let `HENNEPIN_DATA_ROOT` default, set it after the unpack:

    ```bash
    # add to paths.env, or export in your shell:
    HENNEPIN_DATA_ROOT=/abs/path/to/repo/data/hennepin
    ```

## Train and evaluate the aggregation models

Either edit `aggregation/config.py` directly, or override the most common knobs via env vars at invocation time:

| Env var | Default | What it sets |
|---|---|---|
| `EXPERIMENT_NAME` | `gauss_covariance` | `cfg.experiment_name` (controls `results/<name>/` output dir) |
| `TRAIN_MODEL` | `gauss` | `cfg.train.model` — one of `ral` / `uniform` / `rsample` / `gauss` / `logsample` |
| `USE_PRETRAINED` | `0` | `1` to load `PRETRAINED_CKPT` (see "Optional pre-training") |
| `BATCH_SIZE` | `8` | Training batch size |
| `NUM_EPOCHS` | `300` | |
| `NUM_WORKERS` | `4` | DataLoader workers (use `0` on macOS to avoid mp issues) |
| `PATIENCE` | `100` | EarlyStopping patience on `val_loss` |
| `PL_ACCELERATOR` | `auto` | `auto` / `cpu` / `gpu` / `mps` |
| `PL_DEVICES` | `1` | Number of devices |

Then:

```bash
cd aggregation
python train.py     # writes results/<experiment_name>/best.ckpt
python test.py      # writes results/<experiment_name>/test_results/*.pkl + stats.txt
python vis.py       # writes results/<experiment_name>/visualizations/*.png
```

### Smoke test

The full pipeline has been verified end-to-end on CPU with a 10-tile subset:

```bash
cd aggregation
EXPERIMENT_NAME=smoke_test \
BATCH_SIZE=4 NUM_EPOCHS=1 NUM_WORKERS=0 \
PATIENCE=1 \
PL_ACCELERATOR=cpu PL_DEVICES=1 \
python train.py && python test.py && python vis.py
```

(On macOS with the conda-shipped libomp, prefix with `KMP_DUPLICATE_LIB_OK=TRUE`.)
This finishes in under a minute on CPU with a 10-tile dataset and produces the full
output tree: checkpoint, stats, prediction pickles, scatter/error histograms, and
per-sample visualizations. The model isn't useful at 1 epoch on 10 tiles — this just
confirms the train→test→vis chain runs cleanly.

## Hardware reality

Full training assumes a CUDA GPU and was originally done on a multi-GPU cluster. On a CPU-only
laptop you can import the code and run a 1-step Lightning smoke test, but real training will
not finish in any practical time. Move to a server with a recent NVIDIA GPU (or set
`accelerator='mps'` on Apple Silicon for limited experimentation).

## Optional: pre-train the segmentation backbone

The aggregation models can be initialized from a building-segmentation network. The default
configuration disables this (`cfg.train.use_pretrained = False`); to use it you need to
produce a checkpoint at `segmentation/outputs/buildsegpretrain/building_seg_pretrained.pth`
(or somewhere else and set `PRETRAINED_CKPT`).

The segmentation pipeline expects the **`ver8/`-style raw Hennepin layout**, not the
two zips above. Each tile has its own subdirectory keyed by `lat_mid/lon_mid` and contains:

```
<tile>/building_mask.tif
<tile>/parcel_boundary.tif
<tile>/parcel_mask.tif
<tile>/<lat>_<lon>.tif        # source image
<tile>/parcel_value.tif
<tile>/parcels.shp            # optional
```

plus a top-level `hennepin_bbox.csv` listing every tile bbox. **This data is not in
the Box archives** — those zips contain only the downstream-compiled aggregation
format (per-tile RGB PNG + pickled parcel masks + a single `vals.pkl`), with no
GeoTIFF rasters or per-tile bbox catalog. The `ver8/` layout was the *upstream* input
that `dataset_generation/hennepin_compile.py` consumed to *produce* the zips. It
lives on the original cluster.

Once you have it, set these in `paths.env` (or your shell):

```
HENNEPIN_VER8_ROOT=/path/to/ver8
HENNEPIN_VER8_CSV=/path/to/ver8/hennepin_bbox.csv
SEG_OUT_DIR=/abs/path/to/repo/segmentation/outputs/buildsegpretrain
```

Then:

```bash
cd segmentation
python train.py
```

That writes both `model_dict.pth` (best by val loss) and `building_seg_pretrained.pth`
(canonical filename consumed by `aggregation/`). After training, flip
`cfg.train.use_pretrained = True` and re-train aggregation.

If the `ver8/` data is gone, regenerate it with `dataset_generation/` (next section).

## Optional: regenerate the `ver8/` data and the compiled zips from scratch

`dataset_generation/` is a complete pipeline that produces the per-tile `ver8/` layout
that `segmentation/` consumes, and from there the two compiled zips that `aggregation/`
consumes. You only need this if you want to retrain the segmentation backbone and
the original `ver8/` data is unavailable.

**What you need first:**

| Asset | Where to get it | Notes |
|---|---|---|
| `hennepin_bbox.csv` | Already in repo at `dataset_generation/hennepin_bbox.csv` | 5501-tile catalog, restored from git history. |
| Hennepin County parcels shapefile | Auto-downloaded by `dataset_generation/request_zip.sh` from `gis.hennepin.us` | ~92 MB. |
| `Minnesota_ESPG26915.shp` (Microsoft US Building Footprints, MN, reprojected to EPSG:26915) | Auto-downloaded and reprojected by the same `request_zip.sh` (uses the Microsoft Building Footprints dataset on Azure and `ogr2ogr` to reproject). | Requires `ogr2ogr` (GDAL CLI) on PATH. |
| Aerial imagery | Downloaded on-the-fly by `satellite_gather.py` from the Hennepin County WMS service. | Slow + bandwidth-heavy. |

**Pipeline:**

```bash
cd dataset_generation

# 1. Get both shapefiles in one shot (parcels + buildings, reprojected to EPSG:26915):
chmod u+x request_zip.sh && ./request_zip.sh
# Produces: hennepin_county_parcels/hennepin_county_parcels.shp and Minnesota_ESPG26915.shp

# 2. Download aerial imagery into a per-tile lat/lon directory tree
python satellite_gather.py --path /path/to/raw_hennepin --gsd 1

# 3. Generate per-tile {building_mask, parcel_mask, parcel_boundary, masks/<PID>.tif}
python generate_labels.py --dir /path/to/raw_hennepin/ --gsd 1

# At this point /path/to/raw_hennepin matches the `ver8/` layout. You can now train
# the segmentation backbone (see "Optional: pre-train the segmentation backbone").

# 4. (Aggregation only) Compile the per-tile rasters into the imgs/+masks/+vals.pkl
#    layout the aggregation code reads. HENNEPIN_COMBINE=1 = combined-parcels variant.
HENNEPIN_RAW_ROOT=/path/to/raw_hennepin \
HENNEPIN_PARCEL_SHP=$(pwd)/hennepin_county_parcels/hennepin_county_parcels.shp \
HENNEPIN_COMPILED_OUT=/path/to/hennepin/1m_302px_region_combined \
HENNEPIN_COMBINE=1 \
python hennepin_compile.py
```

The supplied Box zips already contain the output of this entire chain *for the aggregation
pipeline*. Re-run it only if you need the per-tile rasters too — i.e., to retrain the
segmentation backbone.

## Synthetic experiments

A self-contained EuroSAT-based ablation lives under `synthetic_testing/`. See
[`synthetic_testing/README.md`](synthetic_testing/README.md) for full instructions.
