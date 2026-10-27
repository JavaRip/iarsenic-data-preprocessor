"""
{filename}

Describe the source of this data here: who produced it, when, and anything
unusual about it (units, naming, overlap with other sources).

Columns in the file:
{columns}

preprocess() must return a dataframe with the columns
Division,District,Upazila,Union,Mouza,Depth,Arsenic as strings,
with depth in metres and arsenic in µg/L. If the data has well coordinates,
also return them as Latitude and Longitude (decimal degrees) so regions can
be identified by GPS. Other columns are dropped.
"""

from iarsenic_data.common import read_csv_as_strings


def preprocess(path):
    df = read_csv_as_strings(path)

    # rename columns to the standard names if needed, e.g.
    # df = df.rename(columns={{'DivisionName': 'Division'}})

    return df
