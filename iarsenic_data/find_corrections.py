"""
Port of iArsenic preprocessing/data-import/find-name-corrections.js

Walks the regions of a source from divisions down to mouzas. Every region
whose name (under its corrected parent) is not in the reference data is
shown with the correctly spelled regions it could be, and the choice is
appended to a corrections file as soon as it is made:

    path,correct
    oldDiv#oldDis,correctDiv#correctDis

Pressing enter without a choice records ~none~, which skips the region and
everything in it.

Unlike the JS tool, a name that matches exactly one correct name under the
same parent apart from case (e.g. JESSORE and Jessore) is corrected without
asking, and saved the same way as if it had been chosen. Subregion names are
also compared ignoring case, and every option shows how many subregions it
shares with the misspelt region, the best one marked as suggested.
--exact-case turns these off, giving the JS tool's behaviour.

When the data has well coordinates, the options also show the share of the
region's wells that are inside each of them on the mouza map, and option g
records ~gps~: the region (and everything in it) is located by GPS when the
training data is built (see gps.py). A case correction above mouza level
is only made without asking if GPS doesn't put most of the wells elsewhere.

Corrections already in the output file are reused, so the tool can be
stopped with ctrl-c and restarted.
"""

import csv
import os
import re
import sys
import unicodedata
from collections import deque

from iarsenic_data.common import LOCATION_COLUMNS
import numpy as np

from iarsenic_data.gps import NO_COORDINATES, OUTSIDE_MAP, load_map, locate_records, map_names, valid_coordinates
from iarsenic_data.load_data import js_number, load_records, region_paths
from iarsenic_data.name_corrections import GPS_CORRECTION, SKIPPED_CORRECTION, combine_path, parse_path

REGION_LABELS = ['division', 'district', 'upazila', 'union', 'mouza']


# terminal colours, as the colors package did
def _style(code, end):
    return lambda s: f'\x1b[{code}m{s}\x1b[{end}m'


bold = _style(1, 22)
italic = _style(3, 23)
underline = _style(4, 24)
red = _style(31, 39)
yellow = _style(33, 39)
magenta = _style(35, 39)
bright_blue = _style(94, 39)
bright_green = _style(92, 39)


# javascript's localeCompare (ICU root collation): spaces and punctuation
# sort before digits, digits before letters, letters ignore case and accents
# first, then accents, then lower case before upper case
_PUNCTUATION_ORDER = ' _-,;:!?.\'"()[]{}@*/\\&#%`^+<=>|~$'


def _primary(c):
    if c.isspace():
        return (0, 0, '')
    if c in _PUNCTUATION_ORDER:
        return (1, _PUNCTUATION_ORDER.index(c), '')
    if c.isdigit():
        return (3, int(c), '')
    base = unicodedata.normalize('NFD', c)[0].casefold()
    if base.isalpha():
        return (4, 0, base)
    return (2, ord(c), '')


def locale_key(s):
    chars = [c for c in s if not unicodedata.combining(c)]
    primary = tuple(_primary(c) for c in chars)
    secondary = tuple(unicodedata.normalize('NFD', c)[1:] for c in s)
    tertiary = tuple(0 if c == c.lower() else 1 for c in chars)
    return primary, secondary, tertiary


class Region:
    """A correctly spelled region in the reference data."""

    def __init__(self, name, parent, has_subregions):
        self.name = name
        self.parent = parent
        self.sub_regions = {} if has_subregions else None
        self.sub_regions_arr = [] if has_subregions else None


class UncheckedRegion:
    """A region of the data being corrected."""

    def __init__(self, old_name, old_parent_path):
        self.old_name = old_name
        self.old_parent_path = old_parent_path
        self.correct_parent_path = [] if not old_parent_path else None
        self.correct_name = None
        self.sub_regions_arr = []
        # number of regions in this one's subtree, itself included
        self.size = 1


def build_reference(records):
    divisions = {}
    for path in region_paths(records):
        regions, parent = divisions, None
        for level, name in enumerate(path):
            if name not in regions:
                region = Region(name, parent, level < 4)
                regions[name] = region
                if parent is not None:
                    parent.sub_regions_arr.append(region)
            parent = regions[name]
            regions = parent.sub_regions
    return divisions


def build_unchecked_tree(records):
    divisions = {}
    for path in region_paths(records):
        regions, parent_arr = divisions, None
        for level, name in enumerate(path):
            key = path[:level + 1]
            if key not in regions:
                region = UncheckedRegion(name, list(path[:level]))
                regions[key] = region
                if parent_arr is not None:
                    parent_arr.append(region)
            parent_arr = regions[key].sub_regions_arr

    # count subtree sizes, deepest regions first
    for region in sorted(divisions.values(), key=lambda r: -len(r.old_parent_path)):
        region.size += sum(r.size for r in region.sub_regions_arr)

    return [r for r in divisions.values() if not r.old_parent_path]


def find_region(reference, names):
    regions, region = reference, None
    for name in names:
        if regions is None or name not in regions:
            raise SystemExit(f'Region {" -> ".join(names)} is not in the reference data')
        region = regions[name]
        regions = region.sub_regions
    return region


def correct_data_contains(reference, names):
    regions = reference
    for name in names:
        if regions is None or name not in regions:
            return False
        regions = regions[name].sub_regions
    return True


def region_name_path(region):
    names = []
    while region is not None:
        names.insert(0, region.name)
        region = region.parent
    return names


def fix_nl(s):
    # fix newlines for display, turn them into spaces
    return re.sub(r'\s+', ' ', s)


def exact(name):
    return name


def name_key(exact_case):
    """How sub-area names are compared: exactly (as the JS tool did) or ignoring case."""
    return exact if exact_case else str.casefold


def highlight_common_subregions(regions_to_list, regions_to_highlight, common_subregions=None, key=exact):
    highlight_keys = {key(n) for n in regions_to_highlight}
    listed = []
    for name in regions_to_list:
        if key(name) in highlight_keys:
            if common_subregions is not None:
                common_subregions.append(name)
            name = bright_blue(fix_nl(name))
        listed.append(name)
    return '' if not listed else f'({italic(", ".join(listed))})'


def subregion_names(region):
    # lowest level region
    if region.sub_regions_arr is None:
        return []
    return sorted(r.name for r in region.sub_regions_arr)


def sibling_regions(reference, region):
    if not region.old_parent_path:
        # a division: siblings are the divisions
        return sorted(reference.values(), key=lambda r: locale_key(r.name))

    parent = find_region(reference, region.correct_parent_path)
    # sorted in place, as the JS did; this changes the order cousins are listed in later
    parent.sub_regions_arr.sort(key=lambda r: locale_key(r.name))
    return parent.sub_regions_arr


def cousin_regions(reference, region):
    # a division doesn't have any cousins
    if not region.correct_parent_path:
        return []

    parent = find_region(reference, region.correct_parent_path)
    grand_parent = parent.parent

    # we may not have grand_parent if region is a district
    parent_level_regions = grand_parent.sub_regions_arr if grand_parent is not None else list(reference.values())

    return [r for uncle in parent_level_regions if uncle is not parent for r in uncle.sub_regions_arr]


def selectable_regions(reference, misspelt_region, misspelt_subregion_names, key=exact):
    selectable = [('sibling', r) for r in sibling_regions(reference, misspelt_region)]

    # only cousins with subregions in common with the misspelt region
    misspelt_keys = {key(n) for n in misspelt_subregion_names}
    for cousin in cousin_regions(reference, misspelt_region):
        if misspelt_keys & {key(n) for n in subregion_names(cousin)}:
            selectable.append(('cousin', cousin))

    return selectable


def shared_subregions(misspelt_subregion_names, region, key):
    """How many of the misspelt region's subregions the option also has."""
    misspelt_keys = {key(n) for n in misspelt_subregion_names}
    return len(misspelt_keys & {key(n) for n in subregion_names(region)})


def option_annotations(selectable, misspelt_subregion_names, key):
    """
    The number of shared subregions for every option, with the option sharing
    the most marked as the suggestion (if exactly one shares the most).
    """
    total = len({key(n) for n in misspelt_subregion_names})
    if total == 0:
        return [''] * len(selectable)

    shared = [shared_subregions(misspelt_subregion_names, region, key) for _, region in selectable]
    best = max(shared, default=0)
    suggested = shared.index(best) if best > 0 and shared.count(best) == 1 else None

    annotations = []
    for i, count in enumerate(shared):
        text = f' {count}/{total} shared'
        annotations.append(bold(bright_green(text + '  <- suggested')) if i == suggested else text)
    return annotations


def options_string(starting_index, regions, misspelt_subregion_names, common_subregions, key, annotations):
    text = ''
    for i, (_, region) in enumerate(regions):
        option_subregions = subregion_names(region) if region.sub_regions is not None else []
        highlighted = highlight_common_subregions(option_subregions, misspelt_subregion_names, common_subregions, key)
        parent_name = '' if starting_index == 0 else region.parent.name + ' -> '
        text += f'\n{yellow(starting_index + i + 1)} {underline(parent_name)}{underline(region.name)}{annotations[i]} {highlighted}\n'
    return text


def options_table(selectable, misspelt_subregion_names, common_subregions, key, annotations):
    siblings = [r for r in selectable if r[0] == 'sibling']
    cousins = [r for r in selectable if r[0] == 'cousin']

    text = options_string(0, siblings, misspelt_subregion_names, common_subregions, key, annotations[:len(siblings)])
    if not cousins:
        return text

    text += f'\n\n{magenta("We also found potential corrections to cousin regions below")}\n'
    return text + options_string(len(siblings), cousins, misspelt_subregion_names, common_subregions, key, annotations[len(siblings):])


def valid_choice(user_input, count):
    value = js_number(user_input)
    return value == value and 0 < value <= count and value == int(value)


def progress_bar(done, total, width=30):
    filled = width * done // total if total else width
    percent = 100 * done / total if total else 100
    return f'[{"#" * filled}{"-" * (width - filled)}] {done:,} / {total:,} regions checked ({percent:.1f}%)'


class GpsContext:
    """Where the wells of each region of the data being corrected are on the map."""

    def __init__(self, records):
        print('Locating wells on the map')
        self.map_rows = locate_records(records)
        names = map_names(np.arange(len(load_map())))
        self.map_paths = [[tuple(n[:level + 1]) for n in names] for level in range(5)]
        self.indices = [
            records.groupby(LOCATION_COLUMNS[:level + 1], sort=False).indices
            for level in range(5)
        ]

        located = (self.map_rows >= 0).sum()
        print(f'  {located:,} of {len(records):,} records are inside the map')

    def shares(self, region):
        """
        (records, records with coordinates, {map path: share of those}) for
        an unchecked region; the None path is the share outside the map.
        """
        level = len(region.old_parent_path)
        path = (*region.old_parent_path, region.old_name)
        rows = self.indices[level].get(path)
        if rows is None and level == 0:
            # pandas versions differ in whether one-column group keys are tuples
            rows = self.indices[level].get(path[0])

        map_rows = self.map_rows[rows]
        map_rows = map_rows[map_rows != NO_COORDINATES]
        if len(map_rows) == 0:
            return len(rows), 0, {}

        shares = {}
        unique, counts = np.unique(map_rows[map_rows >= 0], return_counts=True)
        for map_row, count in zip(unique, counts):
            map_path = self.map_paths[level][map_row]
            shares[map_path] = shares.get(map_path, 0) + count
        shares[None] = int((map_rows == OUTSIDE_MAP).sum())

        return len(rows), len(map_rows), {p: n / len(map_rows) for p, n in shares.items() if n}

    def disagrees(self, region, match):
        """Whether most of the region's wells are in a different map region than match."""
        _, count, shares = self.shares(region)
        if count == 0:
            return False
        match_share = shares.get(tuple(region_name_path(match)), 0)
        best_other = max((share for p, share in shares.items() if p is not None and p != tuple(region_name_path(match))), default=0)
        return match_share < 0.5 and best_other > match_share


def gps_option(record_count, gps_count, shares):
    places = [(path, share) for path, share in shares.items() if path is not None]
    places.sort(key=lambda p: -p[1])

    text = ', '.join(f'{" -> ".join(path[-2:])} {share:.0%}' for path, share in places[:3])
    if len(places) > 3:
        text += ', ...'
    if shares.get(None, 0) >= 0.005:
        text += f', outside the map {shares[None]:.0%}'

    return (
        f'\n{yellow("g")} {underline("Identify by GPS")} '
        f'({gps_count:,} of {record_count:,} records have coordinates, on the map they are in: {text})\n'
    )


def choose_correction(misspelt_region, reference, exact_case=False, progress='', gps=None):
    """Returns ('sibling' | 'cousin', region), ('gps', None) or ('none', None)."""
    key = name_key(exact_case)
    misspelt_subregion_names = sorted(r.old_name for r in misspelt_region.sub_regions_arr)
    corrected_path = misspelt_region.correct_parent_path
    region_label = REGION_LABELS[len(corrected_path)]
    misspelt_name = bold(fix_nl(misspelt_region.old_name))
    common_subregions = []

    selectable = selectable_regions(reference, misspelt_region, misspelt_subregion_names, key)
    annotations = [''] * len(selectable) if exact_case else option_annotations(selectable, misspelt_subregion_names, key)

    gps_count = 0
    if gps is not None:
        record_count, gps_count, shares = gps.shares(misspelt_region)
        for i, (_, region) in enumerate(selectable):
            share = shares.get(tuple(region_name_path(region)), 0)
            if share >= 0.005:
                annotations[i] += f' GPS {share:.0%}'

    options = options_table(selectable, misspelt_subregion_names, common_subregions, key, annotations)
    if gps_count:
        options += gps_option(record_count, gps_count, shares)
    misspelt_subregions = highlight_common_subregions(misspelt_subregion_names, common_subregions, key=key)

    prompt = options + '\n\n'
    if corrected_path:
        prompt += 'In ' + ' -> '.join(corrected_path) + '\n'
    prompt += f'Incorrect {region_label}: {misspelt_name} {misspelt_subregions}'

    while True:
        print('\n\n\n\n\n')
        if progress:
            print(progress)
        print(prompt)
        try:
            user_input = input('Input ID of correct spelling: ')
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        # if user presses enter skip this correction
        if user_input == '':
            print(bold(bright_green(f'SKIPPING {region_label} {misspelt_name}')))
            return 'none', None

        if gps_count and user_input.strip().lower() == 'g':
            print(bold(bright_green(f'Identifying {region_label} {misspelt_name} by GPS')))
            return 'gps', None

        if valid_choice(user_input, len(selectable)):
            selected = selectable[int(js_number(user_input)) - 1]
            print(bold(bright_green(f'Selected: {selected[1].name}')))
            return selected

        or_gps = ' or g' if gps_count else ''
        print(bold(red(f'\nINVALID INPUT, please enter a number 1-{len(selectable)}{or_gps}')))


def load_existing_corrections(file):
    if not os.path.exists(file):
        return {}
    with open(file, newline='', encoding='utf-8') as f:
        return {row['path']: row['correct'] for row in csv.DictReader(f)}


def append_correction(path, correct, file):
    # create the file with headers if it doesn't exist yet
    is_new = not os.path.exists(file)
    with open(file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        if is_new:
            writer.writerow(['path', 'correct'])
        writer.writerow([combine_path(path), combine_path(correct)])


def apply_known_correction(region, known_corrections):
    correction = known_corrections.get(combine_path([*region.old_parent_path, region.old_name]))
    if correction is None:
        return False

    parent_path = parse_path(correction)
    region.correct_name = parent_path.pop()
    region.correct_parent_path = parent_path
    return True


def case_insensitive_match(reference, region):
    """
    The one region under the corrected parent whose name only differs in
    case from the region's name, e.g. JESSORE and Jessore. None if there is
    no such region, or more than one (e.g. Cox's Bazar and Cox'S Bazar).
    """
    if region.correct_parent_path:
        siblings = find_region(reference, region.correct_parent_path).sub_regions_arr
    else:
        siblings = reference.values()

    name = region.old_name.casefold()
    matches = [r for r in siblings if r.name.casefold() == name]
    return matches[0] if len(matches) == 1 else None


def correct(reference, region, correction_file, known_corrections, exact_case=False, progress='', gps=None):
    """Returns whether the region's subregions should be checked."""
    path = combine_path([*region.old_parent_path, region.old_name])
    if known_corrections.get(path) == GPS_CORRECTION:
        # located by GPS when building, along with everything in it
        region.correct_name = GPS_CORRECTION
        return False

    if apply_known_correction(region, known_corrections):
        # treat as corrected
        return True

    if correct_data_contains(reference, [*region.correct_parent_path, region.old_name]):
        # it's already correct
        region.correct_name = region.old_name
        return True

    if region.correct_parent_path and region.correct_parent_path[0] == SKIPPED_CORRECTION:
        region.correct_name = SKIPPED_CORRECTION
        region.correct_parent_path = []
        return False

    match = None if exact_case else case_insensitive_match(reference, region)

    # GPS is too coarse to second guess mouza names
    if match is not None and gps is not None and len(region.correct_parent_path) < 4 and gps.disagrees(region, match):
        print(f'Not case correcting {region.old_name} -> {match.name}: GPS puts most of its wells elsewhere')
        match = None

    if match is not None:
        print(f'Case corrected {REGION_LABELS[len(region.correct_parent_path)]}: {region.old_name} -> {match.name}')
        kind, chosen = 'sibling', match
    else:
        kind, chosen = choose_correction(region, reference, exact_case, progress, gps)

    if kind == 'none':
        region.correct_name = SKIPPED_CORRECTION
        region.correct_parent_path = []
    elif kind == 'gps':
        region.correct_name = GPS_CORRECTION
        region.correct_parent_path = []
    else:
        region.correct_name = chosen.name
        if kind == 'cousin':
            region.correct_parent_path = region_name_path(chosen.parent)

    append_correction(
        [*region.old_parent_path, region.old_name],
        [*region.correct_parent_path, region.correct_name],
        correction_file,
    )
    return kind not in ('none', 'gps')


def find_corrections(source_name, reference_names, correction_file, exact_case=False):
    print(f'Reference data: {", ".join(reference_names)}')
    reference = build_reference(load_records(reference_names, corrections=False))

    print(f'Data to correct: {source_name}')
    records = load_records([source_name], corrections=False)
    queue = deque(build_unchecked_tree(records))

    gps = None
    if valid_coordinates(records).any():
        print(f'{source_name} has well coordinates, regions can be identified by GPS (option g)')
        load_map()
        gps = GpsContext(records)

    known_corrections = load_existing_corrections(correction_file)
    print(f'Saving corrections to {correction_file}, {len(known_corrections)} already there')

    total = sum(r.size for r in queue)
    done = 0

    while queue:
        region = queue.popleft()

        if correct(reference, region, correction_file, known_corrections, exact_case, progress_bar(done, total), gps):
            for sub_region in region.sub_regions_arr:
                # give each subregion a (currently known) correct parent path
                sub_region.correct_parent_path = [*region.correct_parent_path, region.correct_name]
                queue.append(sub_region)
            done += 1
        else:
            # skipped or identified by GPS, so is everything in it
            done += region.size

    print(progress_bar(done, total))
    print(bold(bright_green('All done.')))
