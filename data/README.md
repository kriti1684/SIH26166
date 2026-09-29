# Planetary Datasets & SPICE Kernels Directory

This directory stores planetary raw rasters, orbital SPICE kernels, and runtime databases used by the **ChandraShakti** registration engine.

---

## 📁 Directory Structure

```
data/
├── spice/                 # NASA NAIF / ISRO Chandrayaan-2 SPICE Kernels
│   ├── *.bsp              # Ephemeris & orbital trajectory kernels
│   ├── *.bc               # Spacecraft attitude / pointing kernels
│   ├── *.ti               # Instrument geometry definition kernels (TMC-2, IIRS, OHRC)
│   ├── *.tf               # Frame definitions
│   ├── *.tsc              # Spacecraft clock calibration
│   └── naif0012.tls       # Leapseconds kernel
├── storage/               # (Git-ignored) Pipeline run products, uploads & SQLite DB
│   ├── uploads/           # Uploaded input rasters from web dashboard
│   ├── outputs/           # Job-specific registration outputs & verification diagnostics
│   └── lunar_reg.db       # SQLite application database
└── README.md              # Dataset documentation
```

---

## 🛰️ SPICE Kernels

The engine requires NAIF SPICE kernels for pure-Python ray-tracing (100% ISIS-free):
- **Chandrayaan-2 Planetary Data System**: Available on the [ISSDC ChMapBrowse Portal](https://chmapbrowse.issdc.gov.in/).
- **LRO Planetary Data System**: Available on the [NASA PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/lro/).
- **Automated Kernel Download**: Run `python scripts/download_may2021_ck.py` to fetch attitude kernels for specific observation dates.

---

## 📥 Sample Test Datasets

For testing and benchmarking:
1. Place pairs of source (OHRC / TMC-2 / IIRS) and reference (LRO NAC / WAC / SELENE TC) in `Test_Images/` or upload via the web dashboard.
2. The pipeline auto-ingests both raw PDS labels (`.xml`, `.lbl`, `.LBL`) and standard GeoTIFF rasters (`.tif`).
