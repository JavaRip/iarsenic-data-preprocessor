"""
Danida-20220514.csv

160k wells from the coastal areas, produced by Danida, apparently not
included in 1Mdataset. Division and district names are upper case
(e.g. BARISAL), fixed by data/name-corrections/danida-corrections.csv.

Columns in the file:
Sl,Division,District,Upazila,Union,Mouza,Depth,Arsenic,col1,col2,col3
"""

from iarsenic_data.common import read_csv_as_strings


def preprocess(path):
    return read_csv_as_strings(path)
