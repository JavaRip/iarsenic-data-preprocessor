"""
mouza-names.csv

All known region names, taken to be the correct spellings. It has no depth
or arsenic values so it only adds locations, no wells.
"""

from iarsenic_data.common import read_csv_as_strings


def preprocess(path):
    return read_csv_as_strings(path)
