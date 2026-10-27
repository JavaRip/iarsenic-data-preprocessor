"""
ARRP_arsenic_master_all_data.csv

Raw ARRP master export from DPHE, about 6.5M wells. Replaces
iArsenicArrpTools/preprocess_arrp.py, which only renamed the columns.

Depths are in feet (deep tube wells have a median of 750, ring wells 30)
so they are converted to metres to match the other sources.
The well coordinates (Latitude, Longitude) are kept for GPS name corrections.
The file contains personal data (names, phone numbers) and is not in git.
"""

import pandas as pd

from iarsenic_data.common import read_csv_as_strings

FEET_TO_METRES = 0.3048

COLUMNS = {
    'DivisionName': 'Division',
    'DistrictName': 'District',
    'UpazilaName': 'Upazila',
    'UnionName': 'Union',
    'WordName': 'Mouza',
    'Depth': 'Depth',
    'TestResult': 'Arsenic',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
}


def preprocess(path):
    df = read_csv_as_strings(path, usecols=list(COLUMNS))
    df = df.rename(columns=COLUMNS)

    depth = pd.to_numeric(df['Depth'], errors='coerce') * FEET_TO_METRES
    df['Depth'] = depth.round(1).astype(str).where(depth.notna(), df['Depth'])

    return df
