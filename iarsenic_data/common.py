"""
Shared paths and helpers.

Every preprocessor in preprocessors/ returns a dataframe in the standard format:

    Division,District,Upazila,Union,Mouza,Depth,Arsenic

All values are strings, exactly as they appear in the file, so that they are
treated the same way the JS loader (iArsenic preprocessing/lib/load-data.js)
treated them. Depth is in metres and arsenic in µg/L.

Preprocessors can also return Latitude and Longitude columns (decimal
degrees) when the data has well coordinates; these are used for name
corrections by GPS (see gps.py).
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = ROOT / 'data' / 'sources'
NAME_CORRECTIONS_DIR = ROOT / 'data' / 'name-corrections'
PREPROCESSORS_DIR = ROOT / 'preprocessors'
OUTPUT_DIR = ROOT / 'output'
MAP_PATH = ROOT / 'data' / 'geodata' / 'mouzas.geojson'

COLUMNS = ['Division', 'District', 'Upazila', 'Union', 'Mouza', 'Depth', 'Arsenic']
LOCATION_COLUMNS = COLUMNS[:5]
GPS_COLUMNS = ['Latitude', 'Longitude']


def read_csv_as_strings(path, **kwargs):
    # keep empty strings and names like "NA" exactly as they are in the file
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding='utf-8',
        **kwargs,
    )


def standardise(df):
    # a missing column behaves like an empty value in the JS loader
    standard = df.reindex(columns=COLUMNS, fill_value='')

    # coordinates are optional
    for column in GPS_COLUMNS:
        if column in df:
            standard[column] = pd.to_numeric(df[column], errors='coerce')

    return standard
