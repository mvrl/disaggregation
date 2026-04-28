# Hennepin Dataset Generation

This directory rebuilds the per-tile `ver8/`-style raw dataset from public sources:
the Hennepin County parcels shapefile, Microsoft's US Building Footprints, and the
county's WMS aerial-imagery service. The output of this pipeline is what
`segmentation/` consumes (building-mask pretraining) and what `hennepin_compile.py`
turns into the compiled `1m_302px*.zip` archives that `aggregation/` consumes.

You only need to run this if you want to retrain the segmentation backbone or
reproduce the compiled aggregation data from scratch. The Box archives skip
straight to the end product for the aggregation pipeline.

## Prerequisites

System tools (not installed by `requirements.txt`):

- `wget`           (`request_zip.sh` uses it; on macOS `brew install wget`)
- `unzip`          (standard)
- `ogr2ogr`        (part of GDAL; `brew install gdal` on macOS, `apt install gdal-bin` on Debian/Ubuntu)
- Python deps: `gdal` (osgeo) bindings, `geopandas`, `pandas`, `Pillow`, `tqdm`, `shapely`. The GDAL Python bindings must match your installed `gdal` library version; using the same conda env that has `geopandas` is the smoothest path.

Files referenced by env vars (see `paths.env` at the repo root):

- `HENNEPIN_BBOX_CSV` (defaults to `dataset_generation/hennepin_bbox.csv`, the 5501-tile catalog tracked in this repo)
- `HENNEPIN_RAW_ROOT`, `HENNEPIN_PARCEL_SHP`, `HENNEPIN_COMPILED_OUT`, `HENNEPIN_COMBINE` for `hennepin_compile.py`

## Pipeline

```bash
cd dataset_generation

# 1) Pull both shapefiles. Produces:
#      hennepin_county_parcels/hennepin_county_parcels.shp     (parcel polygons + TOTAL_MV1)
#      hennepin_county_parcels/Minnesota_ESPG26915.shp         (building footprints, reprojected to EPSG:26915)
chmod u+x request_zip.sh && ./request_zip.sh

# 2) Download 1916 aerial-imagery tiles into a per-tile lat/lon directory tree
#    rooted at $HENNEPIN_RAW_ROOT. Reads dataset_generation/hennepin_bbox.csv
#    (the 5501-row catalog; satellite_gather only fetches the rows it iterates over).
python satellite_gather.py --path /path/to/raw_hennepin --gsd 1

# 3) Rasterize labels for each tile:
#      <tile>/building_mask.tif        from Minnesota_ESPG26915.shp
#      <tile>/parcel_mask.tif          from hennepin_county_parcels.shp (binary)
#      <tile>/parcel_boundary.tif      parcel boundary lines
#      <tile>/masks/<PID>.tif          per-parcel masks
python generate_labels.py --dir /path/to/raw_hennepin/ --gsd 1

# 4) (Aggregation pipeline only) Compile per-tile rasters into the
#    imgs/+masks/+vals.pkl layout that aggregation/datasets/hennepin_rebase.py reads.
HENNEPIN_RAW_ROOT=/path/to/raw_hennepin \
HENNEPIN_PARCEL_SHP=$(pwd)/hennepin_county_parcels/hennepin_county_parcels.shp \
HENNEPIN_COMPILED_OUT=/path/to/hennepin/1m_302px_region_combined \
HENNEPIN_COMBINE=1 \
python hennepin_compile.py
```

After step 3 you have the `ver8/` layout that `segmentation/data_factory.py` reads
(its `cfg.data.csv_path` defaults to `dataset_generation/hennepin_bbox.csv`). After step 4
you have the same data the Box archives provide.

## Files in this directory

| File | Purpose |
|---|---|
| `hennepin_bbox.csv` | 5501-tile catalog (UTM EPSG:26915 bboxes). Restored from git history; tracked in the repo. |
| `request_zip.sh` | Downloads parcels + Microsoft buildings shapefiles and reprojects buildings to EPSG:26915. |
| `satellite_gather.py` | Fetches aerial imagery from `gis.hennepin.us` WMS, one PNG per tile. |
| `generate_labels.py` | Rasterizes parcel/building/boundary masks for each tile. |
| `hennepin_compile.py` | Compiles per-tile rasters into the imgs/+masks/+vals.pkl format. |
| `extended_sets.py` | Standalone style-transfer dataset wrapper between two years of imagery. Not part of the main pipeline. |
| `dataset.py`, `utils.py` | Shared helpers. |
