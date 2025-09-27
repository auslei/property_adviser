Perfect idea 👍 — here’s the updated **`MACRO.md`** with **column names documented** for each output file.

You can copy this straight into your repo as `MACRO.md`:

````markdown
# Macro Module (`property_adviser.macro`)

The **macro** module provides Australian macroeconomic data for use in property models.  
It downloads, parses, and annualises official sources, and writes tidy CSVs for downstream use.

---

## Sources

- **Consumer Price Index (CPI)**  
  - Source: RBA *Table G1*  
  - Series: *All groups CPI; Index numbers; Weighted average of eight capital cities*  
  - Data: Quarterly (converted to annual averages and December values)

- **ASX Market Index**  
  - Source: Yahoo Finance (via `yfinance`)  
  - Default ticker: `^AXJO` (ASX 200, ~1992–current)  
  - Optional: set to `^AORD` (All Ordinaries) for longer history (1979–current)  

- **Cash Rate Target**  
  - Source: RBA *Table F1.1*  
  - Series: *Cash Rate Target* (Daily or Monthly average)  
  - Data: Daily, with annual averages and year-end values

---

## Configuration (`config/macro.yml`)

```yaml
start_year: 1990        # Earliest year to include
outdir: data/macro      # Output directory

sources:
  rba_cpi_csv_url: "https://www.rba.gov.au/statistics/tables/csv/g1-data.csv"
  rba_cash_csv_url: "https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv"
  asx200_ticker: "^AXJO"   # or "^AORD" for longer history
````

---

## CLI Usage

```bash
uv run python -m property_adviser.macro.cli --config config/macro.yml --verbose
# or, if installed as a console script:
uv run pa-macro --config config/macro.yml --verbose
```

Logs are written using `core.app_logging`.

---

## Outputs

All CSVs are written to `data/macro/`:

### `cpi_quarterly.csv`

| Column    | Description                                    |
| --------- | ---------------------------------------------- |
| date      | Quarter end date                               |
| cpi_index | CPI index value (All groups, 8 capital cities) |

### `cpi_annual_avg.csv`

| Column        | Description                                       |
| ------------- | ------------------------------------------------- |
| year          | Calendar year                                     |
| cpi_index_avg | Average CPI index across all quarters in the year |
| cpi_yoy_avg   | YoY change in average CPI index                   |

### `cpi_annual_december.csv`

| Column        | Description                      |
| ------------- | -------------------------------- |
| year          | Calendar year                    |
| cpi_index_dec | CPI index at December (Q4)       |
| cpi_yoy_dec   | YoY change in December CPI index |

### `asx200_yearly.csv`

| Column       | Description                 |
| ------------ | --------------------------- |
| year         | Calendar year               |
| asx200_close | ASX index value at year-end |
| asx200_yoy   | YoY change in index value   |

### `rba_cash_daily.csv`

| Column    | Description          |
| --------- | -------------------- |
| date      | Observation date     |
| cash_rate | Cash Rate Target (%) |

### `rba_cash_annual.csv`

| Column               | Description                           |
| -------------------- | ------------------------------------- |
| year                 | Calendar year                         |
| cash_rate_avg        | Average cash rate in the year (%)     |
| cash_rate_eoy        | Cash rate at year-end (%)             |
| cash_rate_change_avg | Change in average cash rate YoY (pp)  |
| cash_rate_change_eoy | Change in year-end cash rate YoY (pp) |

### `macro_au_annual.csv`

Merged dataset with one row per year.
Columns include:

| Column               | Description                           |
| -------------------- | ------------------------------------- |
| year                 | Calendar year                         |
| asx200_close         | ASX200 close at year-end              |
| asx200_yoy           | ASX200 YoY change                     |
| cpi_index_dec        | CPI index (December)                  |
| cpi_yoy_dec          | CPI YoY change (December)             |
| cpi_index_avg        | CPI index (annual average)            |
| cpi_yoy_avg          | CPI YoY change (annual average)       |
| cash_rate_avg        | Average cash rate (%)                 |
| cash_rate_eoy        | Cash rate at year-end (%)             |
| cash_rate_change_avg | Change in average cash rate YoY (pp)  |
| cash_rate_change_eoy | Change in year-end cash rate YoY (pp) |

---

## Integration

To join macro data into your derived property dataset:

```python
from property_adviser.macro.macro_data import add_macro_yearly

df = ...  # your property-level dataset with a 'saleYear' column
df_with_macro = add_macro_yearly(df, macro_path="data/macro/macro_au_annual.csv")
```

This merges on `saleYear` → `year` and attaches CPI, ASX, and Cash Rate features.

```

---

Do you want me to also add a **“current coverage” table** (min year → max year) from your latest run, so you can see at a glance that CPI/ASX/Cash go up to 2025?
```
