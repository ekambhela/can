"""Edge cases in _raw_to_records — the layer that decides how an upload is read.

The format sniff used to be `text[:1] in "{["`, which is True for the empty
string (every string contains ""), so a whitespace-only upload was routed to
json.loads("") and came back as a JSONDecodeError about column 1. That path is
now explicit.
"""

import pytest

from model import predict as P


def _records(raw, filename=""):
    return P._raw_to_records(raw, filename)


# --- the empty / whitespace path ----------------------------------------------
@pytest.mark.parametrize("raw", [
    b"",
    b"   ",
    b"\n\n",
    b" \t \r\n  \n ",
    "﻿   ".encode(),          # BOM + whitespace
])
def test_blank_input_is_rejected_explicitly(raw):
    with pytest.raises(ValueError) as exc:
        _records(raw, "s.csv")
    msg = str(exc.value)
    assert "no data" in msg.lower(), f"want a clear message, got: {msg}"
    assert "json" not in msg.lower(), "blank input must not surface as a JSON error"


def test_blank_input_is_rejected_for_json_names_too():
    with pytest.raises(ValueError) as exc:
        _records(b"   ", "s.json")
    assert "no data" in str(exc.value).lower()


# --- format sniffing ----------------------------------------------------------
def test_json_object_by_content():
    assert _records(b'{"tissue":"skin"}') == [{"tissue": "skin"}]


def test_json_array_by_content():
    assert _records(b'[{"tissue":"skin"},{"tissue":"breast"}]') == \
        [{"tissue": "skin"}, {"tissue": "breast"}]


def test_json_by_extension_even_without_a_brace():
    """A .json file holding a bare scalar is still parsed as JSON (and fails)."""
    with pytest.raises(ValueError):
        _records(b"not json at all", "s.json")


def test_leading_whitespace_before_json_is_handled():
    assert _records(b'\n  {"tissue":"skin"}  \n')[0]["tissue"] == "skin"


def test_csv_is_the_default():
    recs = _records(b"tissue,BRAF_mut\nskin,1\n", "s.csv")
    assert recs == [{"tissue": "skin", "BRAF_mut": 1}]


def test_tsv_by_extension():
    recs = _records(b"tissue\tBRAF_mut\nskin\t1\n", "s.tsv")
    assert recs[0]["tissue"] == "skin"


def test_tsv_detected_by_a_tab_in_the_header():
    recs = _records(b"tissue\tBRAF_mut\nskin\t1\n", "mystery")
    assert recs[0]["tissue"] == "skin"


@pytest.mark.parametrize("header", ["feature", "key", "name", "marker"])
def test_two_column_key_value_form(header):
    """Rows become keys of ONE record. Values arrive as strings (the column is
    mixed-dtype); _coerce_binary handles that downstream."""
    raw = f"{header},value\ntissue,skin\nBRAF_mut,1\n".encode()
    recs = _records(raw, "s.csv")

    assert len(recs) == 1
    assert recs[0]["tissue"] == "skin"
    assert str(recs[0]["BRAF_mut"]) == "1"

    sample, _w, _spec = P.parse_sample(raw, "s.csv")
    assert sample["BRAF_mut"] == 1.0


def test_two_column_table_that_is_not_key_value_stays_rowwise():
    """Two columns whose first isn't a key-ish name is a normal 2-row table."""
    recs = _records(b"tissue,BRAF_mut\nskin,1\nbreast,0\n", "s.csv")
    assert len(recs) == 2
    assert recs[0]["tissue"] == "skin"


def test_multi_row_csv_yields_one_record_per_row():
    recs = _records(b"tissue,BRAF_mut\nskin,1\nbreast,0\nliver,1\n", "s.csv")
    assert len(recs) == 3
    assert [r["tissue"] for r in recs] == ["skin", "breast", "liver"]


def test_utf8_bom_is_stripped():
    raw = "﻿tissue,BRAF_mut\nskin,1\n".encode()
    assert _records(raw, "s.csv")[0]["tissue"] == "skin"


def test_undecodable_bytes_do_not_raise_unicode_errors():
    """errors='replace' means a binary upload can't escape as a
    UnicodeDecodeError; it yields no records and is rejected one layer up."""
    assert _records(b"\xff\xfe\x00\x01\x02", "s.csv") == []

    with pytest.raises(ValueError, match="No sample found"):
        P.parse_sample(b"\xff\xfe\x00\x01\x02", "s.csv")


# --- the row cap --------------------------------------------------------------
def test_max_rows_caps_the_read():
    raw = b"tissue,BRAF_mut\n" + b"skin,1\n" * 50
    with pytest.raises(P.CohortTooLarge):
        P._raw_to_records(raw, "c.csv", max_rows=10)


def test_max_rows_allows_exactly_the_cap():
    raw = b"tissue,BRAF_mut\n" + b"skin,1\n" * 10
    assert len(P._raw_to_records(raw, "c.csv", max_rows=10)) == 10


def test_max_rows_applies_to_json_too():
    body = "[" + ",".join('{"tissue":"skin"}' for _ in range(20)) + "]"
    with pytest.raises(P.CohortTooLarge):
        P._raw_to_records(body.encode(), "c.json", max_rows=10)
