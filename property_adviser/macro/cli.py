from __future__ import annotations
import argparse
from pathlib import Path

from property_adviser.core.app_logging import log, setup_logging
from property_adviser.core.config import load_config, require

from .macro_data import MacroSources, build_macro_tables


def _parse_args():
    p = argparse.ArgumentParser(
        description="Fetch Australian macro data (CPI via RBA G1, ASX index, RBA cash rate F1.1)."
    )
    p.add_argument("--config", type=str, default="config/macro.yml",
                   help="Path to macro config YAML (default: config/macro.yml).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args()


def main():
    args = _parse_args()
    setup_logging(verbose=args.verbose)

    cfg = load_config(Path(args.config))

    start_year = require(cfg, "start_year")
    outdir = cfg.get("outdir", "data/macro")

    src_cfg = cfg.get("sources", {}) or {}
    sources = MacroSources(
        rba_cpi_csv_url=src_cfg.get("rba_cpi_csv_url", MacroSources.rba_cpi_csv_url),
        rba_cash_csv_url=src_cfg.get("rba_cash_csv_url", MacroSources.rba_cash_csv_url),
        asx200_ticker=src_cfg.get("asx200_ticker", MacroSources.asx200_ticker),
    )

    log("macro.start", start_year=start_year, outdir=outdir,
        rba_cpi_csv_url=sources.rba_cpi_csv_url,
        rba_cash_csv_url=sources.rba_cash_csv_url,
        asx200_ticker=sources.asx200_ticker)

    out = build_macro_tables(start_year=start_year, outdir=outdir, sources=sources)
    log("macro.complete", outdir=str(out))


if __name__ == "__main__":
    main()
