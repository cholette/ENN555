"""
Download / load NEM regional settlement prices (5-minute RRP).

Since 5-minute settlement (1 Oct 2021) the settlement price for a region is the
5-minute dispatch RRP from the pricing run (INTERVENTION = 0).

Conventions (both sources below):
  * SETTLEMENTDATE is the END of the 5-minute interval.
  * Timestamps are NEM market time = AEST, fixed UTC+10, no daylight saving.
    They are localised here as 'Australia/Brisbane' (identical offset).
    Do NOT treat them as UTC, and do NOT localise as Australia/Melbourne.

Date ranges
-----------
``start`` and ``end`` are instants. The functions return every 5-minute interval
lying wholly inside [start, end], i.e. interval-ending timestamps t with
start < t <= end. So start="2024-10-01", end="2024-11-01" is exactly October;
end="2024-10-31" would stop at 00:00 on 31 Oct (the 31st itself excluded).
Naive bounds are taken as market time; tz-aware bounds are converted to it.

Sources
-------
1. AEMO aggregated price & demand files (default, one CSV per region-month):
   https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_YYYYMM_<REGION>.csv
   Every month touched by the range is downloaded, then trimmed.
2. NEMweb daily PUBLIC_PRICES_*.zip files already on disk (load_public_prices_zips).
   Each file is a *market day* (04:05 .. 04:00 next day), so make sure the folder
   holds the day before ``start`` too; the result is trimmed to the range.

Usage
-----
    python -m enn555.nem_prices                                   # VIC1, Oct 2024
    python -m enn555.nem_prices --region NSW1 --start 2025-01-15 --end 2025-03-01
    python -m enn555.nem_prices --start "2024-10-05 06:00" --end "2024-10-05 18:00"
    python -m enn555.nem_prices --local data/NEM                  # parse local zips instead
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

AEMO_URL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"
PRICE_COL = "RRP (settlement price, $/MWh)"
DEMAND_COL = "TOTALDEMAND (MW)"
_RENAME = {"RRP": PRICE_COL, "TOTALDEMAND": DEMAND_COL}
MARKET_TZ = "Australia/Brisbane"  # AEST, UTC+10 year-round == NEM market time
INTERVAL = pd.Timedelta("5min")
# AEMO's site rejects requests with no browser-like User-Agent (HTTP 403).
_HEADERS = {"User-Agent": "Mozilla/5.0 (enn555 teaching script)"}


def _to_market_time(t) -> pd.Timestamp:
    """Parse a datetime-like bound; naive -> market time, tz-aware -> converted."""
    t = pd.Timestamp(t)
    return t.tz_localize(MARKET_TZ) if t.tz is None else t.tz_convert(MARKET_TZ)


def _bounds(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = _to_market_time(start), _to_market_time(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    return start, end


def _trim(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Keep intervals wholly inside [start, end] (interval-ending index)."""
    return df.loc[(df.index >= start) & (df.index <= end)]


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """(year, month) of every AEMO monthly file needed for intervals in (start, end].

    AEMO's file for month M holds intervals ending 00:05 on the 1st .. 00:00 on
    the 1st of M+1, i.e. file month = month of (interval end - 5 min).
    """
    first = (start).tz_localize(None).to_period("M")          # first interval begins at >= start
    last = (end - INTERVAL).tz_localize(None).to_period("M")  # last interval begins at end - 5 min
    return [(p.year, p.month) for p in pd.period_range(first, last, freq="M")]


def download_month(region: str, year: int, month: int, timeout: float = 60) -> pd.DataFrame:
    """Download one AEMO monthly PRICE_AND_DEMAND file (raw columns RRP, TOTALDEMAND)."""
    url = AEMO_URL.format(ym=f"{year}{month:02d}", region=region.upper())
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    df = pd.read_csv(io.BytesIO(raw))
    df["timestamp"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S").dt.tz_localize(MARKET_TZ)
    return df.set_index("timestamp")[["RRP", "TOTALDEMAND"]]


def download_price_and_demand(region: str = "VIC1", start="2024-10-01", end="2024-11-01",
                              timeout: float = 60) -> pd.DataFrame:
    """Download settlement prices and demand for one region over [start, end].

    Returns a DataFrame indexed by interval-ending market time with columns
    PRICE_COL (RRP, the regional settlement price before loss factors) and DEMAND_COL.
    """
    start, end = _bounds(start, end)
    frames = [download_month(region, y, m, timeout) for y, m in _months(start, end)]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    return _trim(df, start, end).rename(columns=_RENAME)


def _read_dregion(csv_bytes: bytes) -> pd.DataFrame:
    """Extract the DISPATCH REGIONSUM (DREGION) table from a PUBLIC_PRICES CSV.

    The file is AEMO's multi-table 'CSV' (I = header row, D = data row). It holds
    DREGION in two schema versions (2 and 3) with identical prices, so only the
    highest version is kept -- that is the source of the 'duplicate' rows,
    not multiple dispatch runs.
    """
    header, rows = {}, {}
    for line in csv_bytes.decode("utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 4 or parts[1] != "DREGION":
            continue
        ver = parts[3]
        if parts[0] == "I":
            header[ver] = parts
        elif parts[0] == "D":
            rows.setdefault(ver, []).append(parts)
    ver = max(rows, key=int)
    return pd.DataFrame(rows[ver], columns=header[ver])


def load_public_prices_zips(folder: str | Path, region: str = "VIC1",
                            start="2024-10-01", end="2024-11-01") -> pd.DataFrame:
    """Build the same output from NEMweb daily PUBLIC_PRICES_*.zip files on disk."""
    start, end = _bounds(start, end)
    frames = []
    for f in sorted(Path(folder).glob("PUBLIC_PRICES_*.zip")):
        with zipfile.ZipFile(f) as z:
            for name in z.namelist():
                frames.append(_read_dregion(z.read(name)))
    if not frames:
        raise FileNotFoundError(f"No PUBLIC_PRICES_*.zip files in {folder}")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["REGIONID"] == region.upper()) & (df["INTERVENTION"] == "0")]
    df["timestamp"] = pd.to_datetime(df["SETTLEMENTDATE"].str.strip('"'),
                                     format="%Y/%m/%d %H:%M:%S").dt.tz_localize(MARKET_TZ)
    df["RRP"] = df["RRP"].astype(float)
    df["TOTALDEMAND"] = df["TOTALDEMAND"].astype(float)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return _trim(df[["RRP", "TOTALDEMAND"]], start, end).rename(columns=_RENAME)


def check_complete(df: pd.DataFrame, start, end) -> None:
    """Report any 5-minute interval inside [start, end] that is missing."""
    start, end = _bounds(start, end)
    expected = _trim(pd.DataFrame(index=pd.date_range(start.ceil("5min"), end.floor("5min"),
                                                      freq="5min")), start, end).index
    missing = expected.difference(df.index)
    print(f"{len(df)} intervals, expected {len(expected)}; missing {len(missing)}")
    if len(missing):
        print("  first missing:", list(missing[:5]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default="VIC1")
    p.add_argument("--start", default="2024-10-01", help="start instant, market time (default 2024-10-01)")
    p.add_argument("--end", default="2024-11-01", help="end instant, market time (default 2024-11-01)")
    p.add_argument("--local", type=Path, help="folder of PUBLIC_PRICES_*.zip to parse instead of downloading")
    p.add_argument("--out", type=Path,
                   help="output CSV (default data/NEM/<REGION>_prices_<start>_<end>.csv)")
    a = p.parse_args()

    if a.local:
        df = load_public_prices_zips(a.local, a.region, a.start, a.end)
    else:
        df = download_price_and_demand(a.region, a.start, a.end)
    check_complete(df, a.start, a.end)

    if a.out is None:
        from enn555.paths import data_dir
        s, e = _bounds(a.start, a.end)
        tag = f"{s:%Y%m%d%H%M}_{e:%Y%m%d%H%M}"
        a.out = data_dir() / "NEM" / f"{a.region.upper()}_prices_{tag}.csv"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out)
    print(f"Wrote {a.out}")
    print(df[PRICE_COL].describe().round(2).to_string())


if __name__ == "__main__":
    main()
