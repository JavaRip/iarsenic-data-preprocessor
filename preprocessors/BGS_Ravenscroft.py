"""
BGS_Ravenscroft.csv

Combines (BGS, 2001) and (Ravenscroft et al. 2018), presumed not to overlap
with the other sources. Arsenic is lab measured so takes continuous values.

Columns in the file:
id,Division,District,Upazila,Union,Mouza,Depth,Arsenic,lon,lat
"""

from iarsenic_data.common import read_csv_as_strings


def preprocess(path):
    df = read_csv_as_strings(path)
    return df.rename(columns={'lat': 'Latitude', 'lon': 'Longitude'})
