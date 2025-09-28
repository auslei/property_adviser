import pytest

from property_adviser.preprocess.preprocess_derive import extract_street


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("112 BRUNSWICK ROAD", "Brunswick Road"),
        ("10/6-10 Creek Road", "Creek Road"),
        ("105/435-439 Whitehorse Road", "Whitehorse Road"),
        ("G01/569 Whitehorse Road", "Whitehorse Road"),
        ("UNIT 4/14-16 Mcghee Avenue", "Mcghee Avenue"),
    ],
)
def test_extract_street_removes_unit_numbers(raw, expected):
    cfg = {"unknown_value": "Unknown"}
    assert extract_street(raw, cfg) == expected
