## Dataset Description

This project uses an hourly meteorological dataset for Agra, India, derived
from reanalysis / satellite-based sources.

The full dataset is not included in this repository due to size and data
distribution constraints. Users must supply their own copy of the dataset
with the schema described below.

---

## Required Time Columns

The dataset must contain the following time components:

- YEAR : Year (integer)
- MO   : Month (1–12)
- DY   : Day of month (1–31)
- HR   : Hour of day (0–23)

These columns are internally combined into a `datetime` index during
preprocessing.

---

## Meteorological and Radiation Variables

Required or optionally used columns include:

- ALLSKY_SFC_SW_DWN  
  Global horizontal irradiance under all-sky conditions (W/m²).  
  **Primary prediction target (solar irradiance).**

- CLRSKY_SFC_SW_DWN  
  Clear-sky global horizontal irradiance (W/m²).  
  Used for feature construction when available.

- T2M  
  Air temperature at 2 meters (°C).

- T2MDEW  
  Dew point temperature at 2 meters (°C).  
  Not directly used in the current feature set, but retained for completeness.

- RH2M  
  Relative humidity at 2 meters (%).

- PS  
  Surface pressure (Pa).

- WS50M  
  Wind speed at 50 meters (m/s).  
  **Primary prediction target (wind speed).**

- WD50M  
  Wind direction at 50 meters (degrees).  
  Not directly used in the current feature set, but retained for completeness.

- SZA  
  Solar zenith angle (degrees).  
  Used for physically informed solar feature construction.

---

## Notes on Feature Engineering

- Lag features and rolling statistics are constructed for the target
  variables (`ALLSKY_SFC_SW_DWN`, `WS50M`).
- Cyclical time features (hour, day-of-year, month) are derived from the
  datetime index.
- A physically motivated clear-sky interaction feature is constructed using
  `CLRSKY_SFC_SW_DWN` and `SZA` when available.
- Rows with insufficient history for lag/rolling features are dropped.

---

## Data Ordering and Quality Requirements

- Data must be sorted in chronological order.
- The time resolution is assumed to be **hourly**.
- Missing values are handled implicitly through feature construction and
  row removal after lag/rolling operations.

---

## Example Usage

Place your dataset in the `data/` directory
and update the following entry in `configs/default.yaml`:

```yaml
data_path: data/your_dataset.csv
