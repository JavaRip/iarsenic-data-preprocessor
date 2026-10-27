"""
Port of iArsenic preprocessing/lib/load-data.js

Reads the sources, corrects their region names with the files in
data/name-corrections and keeps the records that have a location.
Records without a valid depth or arsenic value still add their location
to the region hierarchy, but are not wells.

The JS loader returned a nested structure; here the records are kept in a
dataframe and javascript's object key order is reproduced by nested_order()
so that the outputs (see output.py) are identical to the old ones.
"""

import re
import numpy as np
import pandas as pd

from iarsenic_data.common import COLUMNS, LOCATION_COLUMNS, NAME_CORRECTIONS_DIR
from iarsenic_data.name_corrections import GPS_CORRECTION, NameCorrections, SourceCorrections, list_correction_files
from iarsenic_data.sources import get_sources, list_sources

# whitespace stripped by javascript's Number()
_JS_WHITESPACE = '\t\n\v\f\r                  　﻿'
_JS_DECIMAL = re.compile(r'[+-]?(?:Infinity|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)')
_JS_NON_DECIMAL = re.compile(r'0(?:[xX][0-9a-fA-F]+|[oO][0-7]+|[bB][01]+)')

_MAX_ARRAY_INDEX = 2 ** 32 - 2
_ARRAY_INDEX = re.compile(r'0|[1-9]\d*')


def js_number(s):
    """Same result as javascript's Number(string), NaN when not a number."""
    s = s.strip(_JS_WHITESPACE)
    if s == '':
        return 0.0
    if _JS_DECIMAL.fullmatch(s):
        return float(s.replace('Infinity', 'inf'))
    if _JS_NON_DECIMAL.fullmatch(s):
        return float(int(s, 0))
    return np.nan


def to_js_numbers(series):
    # convert each distinct value once, there are few of them
    uniques = series.unique()
    lookup = dict(zip(uniques, (js_number(v) for v in uniques)))
    return series.map(lookup).astype(float)


def read_sources(source_names=None):
    frames = []
    for source in get_sources(source_names):
        print(f'Loading {source.path.name}')
        df = source.load()
        # where each record came from, for tracing records back to the source files
        df['Source'] = source.name
        df['Row'] = np.arange(len(df))
        print(f'  {len(df)} records')
        frames.append(df)

    if not frames:
        raise SystemExit('No sources found in data/sources')

    return pd.concat(frames, ignore_index=True)


def unique_locations(records):
    """Code per row and the table of distinct locations in first-appearance order."""
    codes = records.groupby(LOCATION_COLUMNS, sort=False).ngroup().to_numpy()
    uniq = records.loc[~records.duplicated(LOCATION_COLUMNS), LOCATION_COLUMNS].reset_index(drop=True)
    return codes, uniq


def correct_names(records, name_corrections):
    codes, uniq = unique_locations(records)

    corrected = []
    keep = np.ones(len(uniq), dtype=bool)
    by_gps = np.zeros(len(uniq), dtype=bool)
    for i, row in enumerate(uniq.itertuples(index=False)):
        result = name_corrections.correct(list(row))
        if result is None:
            keep[i] = False
            corrected.append([''] * 5)
        elif result == GPS_CORRECTION:
            # located below
            by_gps[i] = True
            corrected.append(list(row))
        else:
            # JS destructures into 5 variables: extra parts are ignored,
            # missing parts are undefined (treated as empty)
            corrected.append((result + [''] * 5)[:5])

    corrected = np.array(corrected, dtype=object)
    records = records.copy()
    records[LOCATION_COLUMNS] = corrected[codes]

    gps_rows = by_gps[codes]
    if gps_rows.any():
        from iarsenic_data.gps import assign_locations

        # records that can't be located get empty names and are dropped as unlocated
        records.loc[gps_rows, LOCATION_COLUMNS] = assign_locations(records[gps_rows])

    return records[keep[codes]].reset_index(drop=True)


def own_correction_file(source_name):
    return NAME_CORRECTIONS_DIR / f'{source_name}-corrections.csv'


def apply_corrections(records, correction_files=None):
    """
    Corrects the records' names. Without correction_files, each source's own
    file in data/name-corrections is used first, then the rest of the files
    (see name_corrections.py); correction_files are all used for every source.
    """
    sources = list(records['Source'].unique())
    own_files = {}
    if correction_files is None:
        own_files = {s: own_correction_file(s) for s in sources if own_correction_file(s).exists()}

        # a source's own file never applies to other sources, even when it isn't being loaded
        own_paths = {str(own_correction_file(s.name)) for s in list_sources()}
        correction_files = [f for f in list_correction_files(NAME_CORRECTIONS_DIR) if f not in own_paths]

    shared = NameCorrections()
    corrections_length = shared.load_files(correction_files)

    if not own_files:
        if not corrections_length:
            return records
        return correct_names(records, shared)

    # sources are read one after another, so correcting them separately keeps the order
    frames = []
    for source in sources:
        source_records = records[records['Source'] == source]
        if source in own_files:
            own = NameCorrections()
            own.load_files([own_files[source]])
            print(f'Using {own_files[source].name} for {source}')
            frames.append(correct_names(source_records, SourceCorrections(own, shared)))
        elif corrections_length:
            frames.append(correct_names(source_records, shared))
        else:
            frames.append(source_records)
    return pd.concat(frames, ignore_index=True)


def load_records(source_names=None, corrections=True, correction_files=None):
    """
    Records with a location, in the order they were read, with the
    location columns as strings and depth/arsenic as numbers.
    The is_well column says whether the record has a valid depth and arsenic.
    """
    parsed_records = read_sources(source_names)
    records = apply_corrections(parsed_records, correction_files) if corrections else parsed_records

    # skip because we don't have location
    has_location = (records[LOCATION_COLUMNS] != '').all(axis=1)
    records = records[has_location].reset_index(drop=True)

    # wells need depth and arsenic level
    depth = to_js_numbers(records['Depth'])
    arsenic = to_js_numbers(records['Arsenic'])
    records['is_well'] = (records['Depth'] != '') & depth.notna() & (records['Arsenic'] != '') & arsenic.notna()
    records['Depth'] = depth
    records['Arsenic'] = arsenic

    print(f'Parsed {len(parsed_records)} total records, {len(records)} with location, {records["is_well"].sum()} wells')
    return records


def js_key_order(names, first_appearance):
    """
    Sort keys for javascript object property order: integer-like keys
    come first in ascending order, the rest in insertion order.
    """
    is_index = np.array([
        bool(_ARRAY_INDEX.fullmatch(n)) and int(n) <= _MAX_ARRAY_INDEX
        for n in names
    ], dtype=bool)
    index_value = np.array([int(n) if i else 0 for n, i in zip(names, is_index)], dtype=np.int64)
    return (~is_index).astype(np.int8), np.where(is_index, index_value, first_appearance)


def nested_order(records):
    """
    Row order of the JS loader's nested structure: regions in javascript
    key order at every level, records in read order within a mouza.
    """
    codes, uniq = unique_locations(records)

    sort_keys = []
    for level in range(5):
        prefix = LOCATION_COLUMNS[:level + 1]
        first_appearance = uniq.groupby(prefix, sort=False).ngroup().to_numpy()
        not_index, value = js_key_order(uniq[LOCATION_COLUMNS[level]].tolist(), first_appearance)
        sort_keys.append((not_index[codes], value[codes]))

    # np.lexsort uses the last key as the primary key
    keys = [np.arange(len(records))]
    for not_index, value in reversed(sort_keys):
        keys.extend([value, not_index])
    return np.lexsort(keys)


def region_paths(records):
    """Distinct (div, dis, upa, uni, mou) paths in javascript key order."""
    ordered = records.iloc[nested_order(records)]
    return list(ordered.loc[~ordered.duplicated(LOCATION_COLUMNS), LOCATION_COLUMNS].itertuples(index=False, name=None))


def wells_table(records):
    """
    Flat table of wells, in the order model6's load_training_df.py
    produced from the JS loader's JSON.
    """
    ordered = records.iloc[nested_order(records)]
    wells = ordered.loc[ordered['is_well'], COLUMNS].reset_index(drop=True)

    # the JS output went through JSON.stringify: -0 became 0 and Infinity null
    for column in ['Depth', 'Arsenic']:
        wells[column] = wells[column].where(np.isfinite(wells[column])) + 0.0

    return wells
