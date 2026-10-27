"""
Port of iArsenic preprocessing/lib/name-corrections.js

Name corrections for division/district/upazila/union/mouza spelling
variations, loaded from CSV files in data/name-corrections that look like:

    path,correct
    divName#disName#upaName,correctedDivName#correctedDisName#correctedUpaName

A correction to ~none~ means the region is skipped (its records are dropped).
A correction to ~gps~ means the region's records are located by their
coordinates on the mouza map instead (see gps.py).

Corrections in a file named <source>-corrections.csv only apply to that
source, and are used before the others. Every other file applies to every
source, as all corrections did in the JS loader.
"""

import csv

SKIPPED_CORRECTION = '~none~'
GPS_CORRECTION = '~gps~'


def combine_path(arr):
    return '#'.join(arr)


def parse_path(s):
    return s.split('#')


def read_corrections_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def list_correction_files(dir_path):
    # node's fs.readdirSync returns names sorted
    return sorted(str(p) for p in dir_path.iterdir() if p.name.endswith('.csv'))


class NameCorrections:
    def __init__(self):
        self.corrections = {}

    def load_corrections(self, corrections_csv_arr):
        for correction in corrections_csv_arr:
            # first check if that correction is already there
            existing = self.corrections.get(correction['path'])
            if existing is not None:
                existing_combined = combine_path(existing)
                if existing_combined != correction['correct']:
                    print(f"SKIPPING CORRECTION {correction['correct']} for path {correction['path']} which was previously corrected to {existing_combined}")

                # skip the existing correction
                continue
            self.corrections[correction['path']] = parse_path(correction['correct'])

    def load_files(self, file_paths):
        """Loads the files, returns the number of corrections read."""
        rows = []
        for file_path in file_paths:
            rows.extend(read_corrections_csv(file_path))
        self.load_corrections(rows)
        return len(rows)

    def matches(self, arr):
        """Whether any part of the path has a correction."""
        return any(combine_path(arr[:i]) in self.corrections for i in range(len(arr), 0, -1))

    def correct(self, arr):
        """
        Start from the complete path, then if we don't find a correction,
        try to find partial corrections, e.g. just division name.
        Returns None if the region should be skipped, GPS_CORRECTION if
        it should be located by GPS.
        """
        arr = list(arr)
        for i in range(len(arr), 0, -1):
            correction = self.corrections.get(combine_path(arr[:i]))

            if correction is None:
                continue

            if correction[0] == SKIPPED_CORRECTION:
                return None

            if correction[0] == GPS_CORRECTION:
                return GPS_CORRECTION

            return correction + arr[i:]

        # no corrections available, assume arr is correct already
        return arr


class SourceCorrections:
    """
    A source's own corrections (data/name-corrections/<source>-corrections.csv),
    used before the shared ones: if any part of a path has an own correction,
    the shared corrections aren't used for it.
    """

    def __init__(self, own, shared):
        self.own = own
        self.shared = shared

    def correct(self, arr):
        if self.own.matches(arr):
            return self.own.correct(arr)
        return self.shared.correct(arr)
