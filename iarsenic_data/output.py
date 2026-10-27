"""
Writes the training data.

training-data.csv  flat table of wells: Division,District,Upazila,Union,Mouza,Depth,Arsenic
training-data.json the nested format written by the old preprocessing/cli/csv-to-json.js,
                   byte for byte, so model6 can use it unchanged
"""

import json
from decimal import Decimal
from pathlib import Path

import numpy as np

from iarsenic_data.common import LOCATION_COLUMNS
from iarsenic_data.load_data import nested_order, wells_table

# key holding the subregions of each level, the mouza level has none
CHILD_KEYS = ['districts', 'upazilas', 'unions', 'mouzas', None]


def write_csv(records, path):
    wells = wells_table(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wells.to_csv(path, index=False)
    print(f'Wrote {len(wells)} wells to {path}')


def js_number_to_string(x):
    """Same text as JSON.stringify gives for a javascript number."""
    if not np.isfinite(x):
        return 'null'
    if x == 0:
        return '0'

    sign = '-' if x < 0 else ''
    # repr gives the shortest digits that round trip, as javascript does
    digits, exponent = Decimal(repr(abs(float(x)))).normalize().as_tuple()[1:]
    digits = ''.join(map(str, digits))
    k = len(digits)
    n = k + exponent  # position of the decimal point

    if k <= n <= 21:
        return sign + digits + '0' * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + '.' + digits[n:]
    if -6 < n <= 0:
        return sign + '0.' + '0' * -n + digits

    e = n - 1
    mantissa = digits[0] + ('.' + digits[1:] if k > 1 else '')
    return f'{sign}{mantissa}e{"+" if e >= 0 else "-"}{abs(e)}'


def js_string(s):
    return json.dumps(s, ensure_ascii=False)


def write_legacy_json(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    well_index = np.flatnonzero(records['is_well'].to_numpy())
    arsenic = records['Arsenic'].to_numpy()[well_index]
    depth = records['Depth'].to_numpy()[well_index]

    number_text = {v: js_number_to_string(v) for v in set(arsenic) | set(depth)}
    well_text = np.array(
        [f'{{"arsenic":{number_text[a]},"depth":{number_text[d]}}}' for a, d in zip(arsenic, depth)],
        dtype=object,
    )

    # for each level: region code of every record, and the wells of every
    # region in read order (as fillArsenicData pushed them)
    codes = []
    region_wells = []
    for level in range(5):
        level_codes = records.groupby(LOCATION_COLUMNS[:level + 1], sort=False).ngroup().to_numpy()
        well_codes = level_codes[well_index]
        order = np.argsort(well_codes, kind='stable')
        bounds = np.searchsorted(well_codes[order], np.arange(level_codes.max() + 2))
        codes.append(level_codes)
        region_wells.append((well_text[order], bounds))

    def wells_json(level, code):
        text, bounds = region_wells[level]
        return '[' + ','.join(text[bounds[code]:bounds[code + 1]]) + ']'

    # walk the distinct locations in javascript key order
    order = nested_order(records)
    location_codes = codes[4][order]
    first_rows = order[np.concatenate([[True], location_codes[1:] != location_codes[:-1]])]

    names = records[LOCATION_COLUMNS].to_numpy()

    with open(path, 'w', encoding='utf-8') as f:
        f.write('{')
        previous = None

        for row in first_rows:
            path_codes = [codes[level][row] for level in range(5)]

            # first level where this location differs from the previous one
            start = 0
            if previous is not None:
                while path_codes[start] == previous[0][start]:
                    start += 1

                # close the regions of the previous location below that level
                for level in range(4, start - 1, -1):
                    if CHILD_KEYS[level] is not None:
                        f.write('}')
                    f.write(',"name":' + js_string(previous[1][level]) + '}')
                f.write(',')

            for level in range(start, 5):
                f.write(js_string(names[row][level]) + ':{"wells":' + wells_json(level, path_codes[level]))
                if CHILD_KEYS[level] is not None:
                    f.write(',"' + CHILD_KEYS[level] + '":{')

            previous = (path_codes, names[row])

        if previous is not None:
            for level in range(4, -1, -1):
                if CHILD_KEYS[level] is not None:
                    f.write('}')
                f.write(',"name":' + js_string(previous[1][level]) + '}')

        f.write('}')

    print(f'Wrote {len(well_index)} wells to {path}')
