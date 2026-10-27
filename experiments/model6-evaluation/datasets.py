"""
Builds the training data sets compared in this experiment, all with the same
20% of wells held out for testing:

    A  pre-ARRP     1Mdataset, BGS_Ravenscroft, Danida, mouza-names (the old JS loader's data)
    B  production   A + ARRP as model6 currently gets it: names uncorrected, depth in feet
                    (the production output/training-data.json in model6)
    C  new          A + ARRP located by GPS, depth in metres (this repo's default build)
    C1              C with one test per ARRP well (its latest)
    C2              C1 with ARRP balanced against the older data: in each mouza and model6
                    depth band that has older wells, at most as many ARRP wells as older wells
                    (a random sample); places with only ARRP keep all of it

The test wells are 20% of the records of every source, picked by a hash of
(source, row in the source file), so the same wells are held out of every
data set. They are located as in C, which is where they really are.

    python datasets.py           # A, B, C and the test wells
    python datasets.py C1 C2     # just these
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import iarsenic_data.sources as sources  # noqa: E402
from iarsenic_data.common import COLUMNS, NAME_CORRECTIONS_DIR  # noqa: E402
from iarsenic_data.load_data import load_records, wells_table  # noqa: E402
from iarsenic_data.name_corrections import list_correction_files  # noqa: E402

DATA_DIR = HERE / 'data'
PRE_ARRP = ['1Mdataset', 'BGS_Ravenscroft', 'Danida-20220514', 'mouza-names']
ARRP = 'ARRP_arsenic_master_all_data'
TEST_FRACTION = 5  # 1 in 5

# how iArsenicArrpTools/preprocess_arrp.py prepared ARRP for production
PRODUCTION_ARRP_PREPROCESSOR = '''
from iarsenic_data.common import read_csv_as_strings

COLUMNS = {
    "DivisionName": "Division",
    "DistrictName": "District",
    "UpazilaName": "Upazila",
    "UnionName": "Union",
    "WordName": "Mouza",
    "Depth": "Depth",
    "TestResult": "Arsenic",
}


def preprocess(path):
    return read_csv_as_strings(path, usecols=list(COLUMNS)).rename(columns=COLUMNS)
'''


def read_wells(path):
    """A training data csv, as model6 gets it from training-data.json."""
    df = pd.read_csv(
        path,
        dtype={c: str for c in COLUMNS[:5]},
        keep_default_na=False,
        na_values={'Depth': [''], 'Arsenic': ['']},
        float_precision='round_trip',
    )
    return df[COLUMNS]


def arrp_well_codes():
    """ARRP's TubeWellCode for each row of the file, so repeat tests of a well stay together."""
    path = sources.SOURCES_DIR / f'{ARRP}.csv'
    return pd.read_csv(path, usecols=['TubeWellCode'], dtype=str, keep_default_na=False)['TubeWellCode'].to_numpy()


ARRP_WELL_CODES = None


def arrp_testing_dates():
    path = sources.SOURCES_DIR / f'{ARRP}.csv'
    dates = pd.read_csv(path, usecols=['TestingDate'], dtype=str, keep_default_na=False)['TestingDate']
    return pd.to_datetime(dates, errors='coerce').to_numpy()


def latest_test_per_well(records):
    """Keeps only the latest test of each ARRP well (the last record when dates tie)."""
    arrp = records[records['Source'] == ARRP]
    rows = arrp['Row'].to_numpy()
    order = pd.DataFrame({
        'well': arrp_well_codes()[rows],
        'date': arrp_testing_dates()[rows],
        'row': rows,
    }, index=arrp.index).sort_values(['well', 'date', 'row'], na_position='first')
    keep = order.index[~order.duplicated('well', keep='last')]
    return pd.concat([records[records['Source'] != ARRP], records.loc[keep]]).sort_index()


def balance_sources(records, seed=0):
    """
    In each mouza and model6 depth band with older wells, keeps at most as many
    ARRP wells as older wells, picked at random.
    """
    wells = records[records['is_well']]
    band = np.digitize(wells['Depth'], [15.3, 45, 65, 90, 150])
    cell = wells[COLUMNS[:5]].agg('#'.join, axis=1) + '#' + band.astype(str)
    is_arrp = (wells['Source'] == ARRP).to_numpy()

    older = cell[~is_arrp].value_counts()
    arrp_cells = cell[is_arrp]
    cap = arrp_cells.map(older)

    # random rank of each ARRP well within its cell
    rank = pd.Series(np.random.default_rng(seed).random(len(arrp_cells)), index=arrp_cells.index)
    rank = rank.groupby(arrp_cells).rank(method='first')

    drop = arrp_cells.index[cap.notna() & (rank > cap.fillna(0))]
    print(f'balancing: dropping {len(drop):,} of {len(arrp_cells):,} ARRP training wells in places with older data')
    return records.drop(index=drop)


def is_test(records):
    """
    Test wells: 1 in 5 records by a hash of (source, row in the source file).
    ARRP wells are often tested more than once, so ARRP is split by well
    (TubeWellCode) instead, keeping all the tests of a well on one side.
    """
    global ARRP_WELL_CODES

    ids = pd.DataFrame({'source': records['Source'].astype(str), 'row': records['Row']})
    test = (pd.util.hash_pandas_object(ids, index=False).to_numpy() % TEST_FRACTION) == 0

    arrp = (records['Source'] == ARRP).to_numpy()
    if arrp.any():
        if ARRP_WELL_CODES is None:
            ARRP_WELL_CODES = arrp_well_codes()
        codes = pd.Series(ARRP_WELL_CODES[records['Row'].to_numpy()[arrp]])
        test[arrp] = (pd.util.hash_pandas_object(codes, index=False).to_numpy() % TEST_FRACTION) == 0

    return test


def production_preprocessors():
    """A copy of preprocessors/ with ARRP prepared as it was for production."""
    target = DATA_DIR / 'production-preprocessors'
    target.mkdir(parents=True, exist_ok=True)
    for path in sources.PREPROCESSORS_DIR.glob('*.py'):
        shutil.copy(path, target / path.name)
    (target / f'{ARRP}.py').write_text(PRODUCTION_ARRP_PREPROCESSOR)
    return target


def write_train(records, name, prune=None):
    train = records[~is_test(records)]
    if prune is not None:
        train = prune(train)
    wells = wells_table(train)
    path = DATA_DIR / f'{name}-train.csv'
    wells.to_csv(path, index=False)
    print(f'{name}: {len(wells):,} training wells -> {path}')


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        records = load_records()
        prunes = {
            'C1': latest_test_per_well,
            'C2': lambda train: balance_sources(latest_test_per_well(train)),
        }
        for name in sys.argv[1:]:
            write_train(records, name, prunes[name])
        return

    # C, and the test wells located as in C
    records = load_records()
    write_train(records, 'C')
    test = records[is_test(records) & records['is_well']]
    test = test[COLUMNS + ['Source', 'Row']]
    test.to_csv(DATA_DIR / 'test.csv', index=False)
    print(f'test: {len(test):,} wells ({(test.Source == ARRP).sum():,} ARRP) -> {DATA_DIR / "test.csv"}')

    # A
    write_train(load_records(PRE_ARRP), 'A')

    # B: production ARRP preprocessing, and the shared corrections applied to every
    # source as the JS loader did (no ARRP corrections file)
    default_preprocessors = sources.PREPROCESSORS_DIR
    sources.PREPROCESSORS_DIR = production_preprocessors()
    shared = [f for f in list_correction_files(NAME_CORRECTIONS_DIR) if Path(f).name != f'{ARRP}-corrections.csv']
    write_train(load_records(correction_files=shared), 'B')
    sources.PREPROCESSORS_DIR = default_preprocessors


if __name__ == '__main__':
    main()
