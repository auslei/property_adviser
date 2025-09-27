from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import requests

from property_adviser.core.app_logging import log, log_exc

# Optional dependency for market index
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


# ---------------------------------------------------------------------
# Sources / Configuration
# ---------------------------------------------------------------------
@dataclass
class MacroSources:
    """
    Endpoints/tickers for macro series. Override via config if needed.
    """
    # CPI (RBA table G1 – Consumer price index measures)
    rba_cpi_csv_url: str = "https://www.rba.gov.au/statistics/tables/csv/g1-data.csv"

    # Cash rate target (RBA table F1.1)
    rba_cash_csv_url: str = "https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv"

    # Australian market index (ASX 200). Consider ^AORD for longer history.
    asx200_ticker: str = "^AXJO"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_rba_csv_with_header_rows(url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read an RBA 'tables' CSV (Windows-1252, multi-row header):
        Row 0: Title
        Row 1: Description
        Row 2: Frequency
        Row 3: Type
        Row 4: Units
        Rows 5+: data; column 0 is date; other columns are series.

    Returns (header_rows, data_rows).
    """
    log("macro.rba.csv.fetch", url=url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.content.decode("cp1252")), header=None)
    if raw.shape[0] < 6:
        raise RuntimeError(f"Unexpected RBA CSV shape for {url}: {raw.shape}")

    header_rows = raw.iloc[:5].fillna("")
    data = raw.iloc[5:].reset_index(drop=True)
    data.rename(columns={0: "date"}, inplace=True)
    return header_rows, data


def _build_column_metadata(header_rows: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    From the 5 header rows, build a per-column metadata map:
      {col_index: {"title":..., "desc":..., "freq":..., "type":..., "units":...}}
    """
    col_map: Dict[int, Dict[str, Any]] = {}
    for col in range(1, header_rows.shape[1]):
        meta = {
            "title": str(header_rows.iat[0, col]).strip(),
            "desc":  str(header_rows.iat[1, col]).strip(),
            "freq":  str(header_rows.iat[2, col]).strip(),
            "type":  str(header_rows.iat[3, col]).strip(),
            "units": str(header_rows.iat[4, col]).strip(),
        }
        if not any(meta.values()):  # skip empty padding columns
            continue
        col_map[col] = meta
    return col_map


# ---------------------------------------------------------------------
# CPI from RBA G1 (periodic → annual)
# ---------------------------------------------------------------------
def fetch_rba_cpi_table(csv_url: str) -> pd.DataFrame:
    """
    Read RBA G1 CPI table and select the exact "All groups CPI; index numbers;
    Weighted average of eight capital cities" series.

    Returns periodic CPI as:
        ['date','cpi_index']
    """
    header_rows, data = _read_rba_csv_with_header_rows(csv_url)
    col_map = _build_column_metadata(header_rows)

    target_col = None
    for col, meta in col_map.items():
        d = meta["desc"].lower()
        # canonical CPI index series (not % change)
        if ("all groups cpi" in d) and ("index" in d) and ("weighted average of eight capital cities" in d):
            target_col = col
            break

    if target_col is None:
        # Fallback: first column whose description mentions 'index' and units are not percentages
        for col, meta in col_map.items():
            d = meta["desc"].lower()
            if "index" in d and "%" not in meta["units"].lower():
                target_col = col
                break

    if target_col is None:
        raise RuntimeError(f"Unable to locate CPI index column. Columns meta: {col_map}")

    out = pd.DataFrame({
        "date": pd.to_datetime(data["date"], errors="coerce", infer_datetime_format=True),
        "cpi_index": pd.to_numeric(data[target_col], errors="coerce"),
    }).dropna().sort_values("date").reset_index(drop=True)

    log("macro.cpi.periodic_rows", rows=len(out))
    return out


def cpi_to_annual_from_rba(periodic: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert periodic CPI to annual aggregates:
      - cpi_index_avg: calendar-year average
      - cpi_index_dec: last observation of the year (Q4 or December)
      - cpi_yoy_*: YoY based on index values

    Returns (cpi_avg_df, cpi_dec_df).
    """
    d = periodic.copy()
    d["year"] = d["date"].dt.year

    cpi_avg = (d.groupby("year", as_index=False)
                 .agg(cpi_index_avg=("cpi_index", "mean"))
                 .sort_values("year"))
    cpi_avg["cpi_yoy_avg"] = cpi_avg["cpi_index_avg"].pct_change()

    dec = (d.sort_values("date")
             .groupby("year", as_index=False)
             .tail(1)[["year", "cpi_index"]]
             .rename(columns={"cpi_index": "cpi_index_dec"})
             .sort_values("year"))
    dec["cpi_yoy_dec"] = dec["cpi_index_dec"].pct_change()

    return cpi_avg.reset_index(drop=True), dec.reset_index(drop=True)


# ---------------------------------------------------------------------
# RBA Cash Rate (periodic → annual)
# ---------------------------------------------------------------------
def fetch_rba_cash_rate_daily(sources: MacroSources) -> pd.DataFrame:
    """
    Fetch the RBA F1.1 table and select the 'Cash Rate Target' series.
    Robust to header variations like:
      desc: 'Cash Rate Target'
      freq: 'Cash Rate Target; monthly average'
      units: 'Original'
    Prefers daily if present; otherwise monthly.
    Returns ['date','cash_rate'] (percent).
    """
    header_rows, data = _read_rba_csv_with_header_rows(sources.rba_cash_csv_url)
    col_map = _build_column_metadata(header_rows)

    # Gather candidates by looking at Title/Description only (ignore Units=Original)
    candidates = []
    for col, meta in col_map.items():
        title = meta["title"].lower()
        desc  = meta["desc"].lower()
        freq  = meta["freq"].lower()
        text = f"{title} || {desc}"
        if "cash rate" in text and "target" in text:
            # Classify frequency if we can
            if "daily" in freq or "business daily" in freq:
                rank = 0
            elif "monthly" in freq:
                rank = 1
            else:
                rank = 5  # unknown, but still acceptable
            candidates.append((rank, col))

    if not candidates:
        # Helpful debug: surface a few series we saw
        examples = [
            {"title": col_map[c]["title"], "desc": col_map[c]["desc"], "freq": col_map[c]["freq"], "units": col_map[c]["units"]}
            for c in sorted(col_map.keys())
        ][:8]
        raise RuntimeError(f"Could not find 'Cash Rate Target' series in RBA F1.1 table. Examples: {examples}")

    # Prefer daily over monthly
    candidates.sort(key=lambda x: x[0])
    target_col = candidates[0][1]

    out = pd.DataFrame({
        "date": pd.to_datetime(data["date"], errors="coerce"),
        "cash_rate": pd.to_numeric(data[target_col], errors="coerce"),
    }).dropna().sort_values("date").reset_index(drop=True)

    # Normalise units to percent if needed
    if not out.empty and out["cash_rate"].median() < 0.2:
        out["cash_rate"] = out["cash_rate"] * 100.0

    log("macro.rba_cash.periodic_rows", rows=len(out))
    return out



def rba_cash_to_annual(periodic: pd.DataFrame) -> pd.DataFrame:
    """
    Annualise the cash-rate series:
      - cash_rate_avg: calendar-year average (%)
      - cash_rate_eoy: last observation in the year (%)
      - cash_rate_change_avg/eoy: year-on-year change (percentage points)
    """
    d = periodic.copy()
    d["year"] = d["date"].dt.year

    avg = (d.groupby("year", as_index=False)
             .agg(cash_rate_avg=("cash_rate", "mean"))
             .sort_values("year"))
    eoy = (d.sort_values("date")
             .groupby("year", as_index=False)
             .tail(1)[["year", "cash_rate"]]
             .rename(columns={"cash_rate": "cash_rate_eoy"})
             .sort_values("year"))

    annual = avg.merge(eoy, on="year", how="outer").sort_values("year")
    annual["cash_rate_change_avg"] = annual["cash_rate_avg"].diff()
    annual["cash_rate_change_eoy"] = annual["cash_rate_eoy"].diff()
    return annual.reset_index(drop=True)


# ---------------------------------------------------------------------
# ASX 200 (^AXJO) – annual
# ---------------------------------------------------------------------
def fetch_asx200_yearly(start_year: int, sources: MacroSources) -> pd.DataFrame:
    """
    Reliable long-span ASX200 history using yfinance with period='max' and
    year-end resampling. Returns ['year','asx200_close','asx200_yoy'].

    Note: ^AXJO history generally starts around ~2000. For earlier coverage, use ^AORD.
    """
    if yf is None:  # pragma: no cover
        raise ImportError("yfinance is required. Install with `pip install yfinance`.")

    tkr = yf.Ticker(sources.asx200_ticker)
    hist = tkr.history(period="max", interval="1d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No data returned for ticker {sources.asx200_ticker}.")

    # Ensure DatetimeIndex
    if not isinstance(hist.index, pd.DatetimeIndex):
        hist = hist.reset_index().set_index("Date")

    hist = hist[["Close"]].dropna()

    # Year-end close; use 'YE' to avoid FutureWarning about 'A'
    yearly = (
        hist.resample("YE")
            .last()
            .rename(columns={"Close": "asx200_close"})
    )
    yearly.index = yearly.index.year
    yearly.index.name = "year"

    yearly = yearly.loc[yearly.index >= start_year]
    yearly["asx200_yoy"] = yearly["asx200_close"].pct_change()

    out = yearly.reset_index()

    if not out.empty:
        log("macro.asx200.years",
            years=len(out),
            first=int(out.loc[:, "year"].min()),
            last=int(out.loc[:, "year"].max()))
    else:
        log("macro.asx200.years", years=0)

    return out


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def build_macro_tables(start_year: int,
                       outdir: str = "data/macro",
                       sources: Optional[MacroSources] = None) -> Path:
    """
    Fetch CPI (RBA G1), ASX200, and RBA cash rate.
    Write tidy CSVs + merged macro_au_annual.csv.

    Returns the output directory Path.
    """
    sources = sources or MacroSources()
    out = Path(outdir)
    _ensure_outdir(out)

    # ---- CPI (periodic + annual) ----
    try:
        cpi_periodic = fetch_rba_cpi_table(sources.rba_cpi_csv_url)
        cpi_periodic = cpi_periodic[cpi_periodic["date"].dt.year >= start_year]
        cpi_avg, cpi_dec = cpi_to_annual_from_rba(cpi_periodic)

        cpi_periodic.to_csv(out / "cpi_quarterly.csv", index=False)
        cpi_avg.to_csv(out / "cpi_annual_avg.csv", index=False)
        cpi_dec.to_csv(out / "cpi_annual_december.csv", index=False)
    except Exception as e:  # pragma: no cover
        log_exc("macro.cpi.error", e)
        cpi_avg = pd.DataFrame(columns=["year", "cpi_index_avg", "cpi_yoy_avg"])
        cpi_dec = pd.DataFrame(columns=["year", "cpi_index_dec", "cpi_yoy_dec"])

    # ---- ASX200 (annual) ----
    try:
        axjo = fetch_asx200_yearly(start_year, sources)
        axjo.to_csv(out / "asx200_yearly.csv", index=False)
    except Exception as e:  # pragma: no cover
        log_exc("macro.asx200.error", e)
        axjo = pd.DataFrame(columns=["year", "asx200_close", "asx200_yoy"])

    # ---- Cash rate (periodic + annual) ----
    try:
        cash_periodic = fetch_rba_cash_rate_daily(sources)
        cash_periodic = cash_periodic[cash_periodic["date"].dt.year >= start_year]
        cash_annual = rba_cash_to_annual(cash_periodic)

        cash_periodic.to_csv(out / "rba_cash_daily.csv", index=False)
        cash_annual.to_csv(out / "rba_cash_annual.csv", index=False)
    except Exception as e:  # pragma: no cover
        log_exc("macro.rba_cash.error", e)
        cash_annual = pd.DataFrame(columns=[
            "year", "cash_rate_avg", "cash_rate_eoy",
            "cash_rate_change_avg", "cash_rate_change_eoy"
        ])

    # ---- Merge annual tables on year (outer join) ----
    macro = (axjo.merge(cpi_dec, on="year", how="outer")
                 .merge(cpi_avg, on="year", how="outer", suffixes=("", "_avg"))
                 .merge(cash_annual, on="year", how="outer")
                 .sort_values("year"))
    macro.to_csv(out / "macro_au_annual.csv", index=False)

    log("macro.write.complete",
        outdir=str(out),
        files=[
            "cpi_quarterly.csv",
            "cpi_annual_avg.csv",
            "cpi_annual_december.csv",
            "asx200_yearly.csv",
            "rba_cash_daily.csv",
            "rba_cash_annual.csv",
            "macro_au_annual.csv",
        ])
    return out


# ---------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------
def add_macro_yearly(df: pd.DataFrame,
                     macro_path: str = "data/macro/macro_au_annual.csv",
                     sale_year_col: str = "saleYear") -> pd.DataFrame:
    """
    Left-join annual macro features to your derived property data on sale year.
    """
    macro = pd.read_csv(macro_path)
    if sale_year_col not in df.columns:
        raise KeyError(f"{sale_year_col} not in df. Ensure your derivations include year.")
    out = df.merge(macro, left_on=sale_year_col, right_on="year", how="left")
    return out.drop(columns=["year"])
