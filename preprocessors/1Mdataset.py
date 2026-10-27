"""
1Mdataset.csv

The national screening data set, about 1M wells. Already in the standard
format; depths are in metres.
"""

from iarsenic_data.common import read_csv_as_strings


def preprocess(path):
    return read_csv_as_strings(path)
