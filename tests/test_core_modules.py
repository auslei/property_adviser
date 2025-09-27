import json
from pathlib import Path

import pandas as pd
import pytest

from property_adviser.core import config as cfg_mod
from property_adviser.core import io as io_mod
from property_adviser.core import runner as runner_mod


def test_load_config_returns_mapping(tmp_path):
    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text("key: value\ninner:\n  nested: 1\n")

    data = cfg_mod.load_config(yaml_path)
    assert data["key"] == "value"
    assert data["inner"]["nested"] == 1


def test_load_config_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError):
        cfg_mod.load_config(missing)


def test_require_fetches_nested_keys(tmp_path):
    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text("parent:\n  child: value\n")
    data = cfg_mod.load_config(yaml_path)

    assert cfg_mod.require(data, "parent", "child") == "value"
    with pytest.raises(KeyError):
        cfg_mod.require(data, "parent", "missing")


def test_save_and_load_parquet_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    csv_path = tmp_path / "data.csv"
    io_mod.save_parquet_or_csv(df, csv_path)
    loaded_csv = io_mod.load_parquet_or_csv(csv_path)
    pd.testing.assert_frame_equal(df, loaded_csv)

    parquet_path = tmp_path / "data.parquet"
    io_mod.save_parquet_or_csv(df, parquet_path)
    loaded_parquet = io_mod.load_parquet_or_csv(parquet_path)
    pd.testing.assert_frame_equal(df, loaded_parquet)


def test_runner_enforces_dataframe_output():
    df = pd.DataFrame({"value": [1, 2]})

    def add_one(input_df: pd.DataFrame) -> pd.DataFrame:
        result = input_df.copy()
        result["value"] += 1
        return result

    result = runner_mod.run_step("add_one", add_one, df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"value": [2, 3]}))

    with pytest.raises(TypeError):
        runner_mod.run_step("bad", lambda d: 1, df)
