"""
A faster copy of iArsenic model6's utils/generate_prediction_data.py, for
running the model on many training data sets.

The original filters the whole training table for every mouza and stratum
(td[(td['region_key'] == key) & ...]), which takes days with the ARRP data.
Here the rows of each region are looked up from an index first and the same
pandas operations are applied to just those rows, so the selected wells,
the de-duplication and the statistics are the same. find_regions_within_radius
(sjoin_nearest with max_distance) is replaced by the equivalent spatial
index query, cached per mouza.

check_fast_model6.py compares the output with the original code.
"""

import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

MODEL6_DIR = '/home/kane/Default Folder/code/iArsenic2024/preprocessing/models/model6'
PATCH_LTE_90_PATH = os.path.join(MODEL6_DIR, 'model_patch/arsenic_statistics_lte_90m_v2.csv')
PATCH_GT_90_PATH = os.path.join(MODEL6_DIR, 'model_patch/arsenic_statistics_gt_90m_v2.csv')
MOUZA_TOPOJSON = os.path.join(MODEL6_DIR, 'output/geodata/intersected_mouzas.topojson')
DROPDOWN_DATA = os.path.join(MODEL6_DIR, 'output/dropdown-data.json')

MAX_WORKERS = 20

# set before the worker processes are forked, so they share them
CONTEXT = None


class Context:
    def __init__(self, td, gd):
        self.td = td
        self.gd = gd
        self.key_rows = td.groupby('region_key', sort=False).indices
        self.gd_keys = gd['region_key'].to_numpy()
        self.patch_lte_90 = pd.read_csv(PATCH_LTE_90_PATH)
        self.patch_gt_90 = pd.read_csv(PATCH_GT_90_PATH)
        self.depth = td['Depth'].to_numpy()
        self.nearby_cache = {}

    def own(self, region_key):
        """td rows of the region, as td[td['region_key'] == region_key] would give."""
        rows = self.key_rows.get(region_key)
        return self.td.iloc[rows if rows is not None else []]

    def nearby(self, distance, region_key, centroid, depth_filter):
        """
        td rows of the regions within distance that pass depth_filter, as
        td[td['region_key'].isin(keys) & depth_filter(td['Depth'])] would give.
        The depth filter is applied before any rows are copied, because the
        regions within 100km can have millions of wells.
        """
        cache_key = (region_key, distance)
        if cache_key not in self.nearby_cache:
            positions = self.gd.sindex.query(centroid, predicate='dwithin', distance=distance)
            keys = self.gd_keys[positions]
            rows = [self.key_rows[k] for k in keys if k in self.key_rows]
            rows = np.sort(np.concatenate(rows)) if rows else np.array([], dtype=np.int64)
            self.nearby_cache[cache_key] = rows

        rows = self.nearby_cache[cache_key]
        return self.td.iloc[rows[depth_filter(self.depth[rows])]]


def produce_message(med, max_val):
    if med is None:
        return 0

    chem_test_status = 0 if max_val <= 100 else 1

    if med <= 10:
        return 1 + chem_test_status
    elif med <= 50:
        return 3 + chem_test_status
    elif med <= 150:
        return 5 + chem_test_status
    else:
        return 7 + chem_test_status


def patch(table, mou):
    div, dis, upa = mou['div'].lower(), mou['dis'].lower(), mou['upa'].lower()
    match = table[
        (table['Division'] == div) &
        (table['District'] == dis) &
        (table['Upazila'] == upa)
    ]
    if not match.empty:
        return {"m": int(match['Risk'].values[0])}
    return None


def stats(wells, mou, patch_table):
    arsenic = wells['Arsenic'].dropna()

    if len(arsenic) >= 7:
        p10, p50, p90 = np.percentile(arsenic, [10, 50, 90])
        return {
            "m": produce_message(p50, arsenic.max()),
            "l": round(p10, 1),
            "u": round(p90, 1),
            "median": p50,
        }

    patched = patch(patch_table, mou)
    if patched is not None:
        return patched
    return {"m": 0}


def s15_flood_stats(c, region_key, centroid):
    own = c.own(region_key)
    wells_15 = own[own['Depth'] < 15]

    if len(wells_15) < 7:
        nearby_wells = c.nearby(5000, region_key, centroid, lambda d: d < 45)
        wells_15 = pd.concat([wells_15, nearby_wells]).drop_duplicates()

    arsenic_15 = wells_15['Arsenic'].dropna()

    if len(arsenic_15) >= 7:
        p25, p50, p75, p95 = np.percentile(arsenic_15, [25, 50, 75, 95])
        max_val = arsenic_15.max()

        return {
            "m2": produce_message(p25, max_val),
            "m7": produce_message(p75, max_val),
            "m9": produce_message(p95, max_val),
            "median": p50,
        }


def s15_stats(c, mou, region_key, centroid):
    own = c.own(region_key)
    wells_15 = own[own['Depth'] < 15]

    if len(wells_15) < 7:
        extra = own[own['Depth'] < 45]
        wells_15 = pd.concat([wells_15, extra]).drop_duplicates()

    if len(wells_15) < 7:
        nearby_wells = c.nearby(10000, region_key, centroid, lambda d: d < 45)
        wells_15 = pd.concat([wells_15, nearby_wells]).drop_duplicates()

    return stats(wells_15, mou, c.patch_lte_90)


def s45_stats(c, mou, region_key, centroid):
    own = c.own(region_key)
    wells_45 = own[(own['Depth'] >= 15) & (own['Depth'] < 45)]

    if len(wells_45) < 7:
        extra = own[(own['Depth'] >= 15) & (own['Depth'] < 65)]
        wells_45 = pd.concat([wells_45, extra]).drop_duplicates()

    if len(wells_45) < 7:
        nearby_wells_10k = c.nearby(10000, region_key, centroid, lambda d: (d >= 15) & (d < 45))
        wells_45 = pd.concat([wells_45, nearby_wells_10k]).drop_duplicates()

    if len(wells_45) < 7:
        nearby_wells_20k = c.nearby(20000, region_key, centroid, lambda d: (d >= 15) & (d < 65))
        wells_45 = pd.concat([wells_45, nearby_wells_20k]).drop_duplicates()

    return stats(wells_45, mou, c.patch_lte_90)


def s65_stats(c, mou, region_key, centroid):
    own = c.own(region_key)
    wells_65 = own[(own['Depth'] >= 45) & (own['Depth'] < 65)]

    if len(wells_65) < 7:
        extra = own[(own['Depth'] >= 45) & (own['Depth'] < 90)]
        wells_65 = pd.concat([wells_65, extra]).drop_duplicates()

    if len(wells_65) < 7:
        nearby_wells_10k = c.nearby(10000, region_key, centroid, lambda d: (d >= 45) & (d < 65))
        wells_65 = pd.concat([wells_65, nearby_wells_10k]).drop_duplicates()

    if len(wells_65) < 7:
        nearby_wells_20k = c.nearby(20000, region_key, centroid, lambda d: (d >= 45) & (d < 90))
        wells_65 = pd.concat([wells_65, nearby_wells_20k]).drop_duplicates()

    return stats(wells_65, mou, c.patch_lte_90)


def s90_stats(c, mou, region_key, centroid):
    own = c.own(region_key)
    wells_90 = own[(own['Depth'] >= 65) & (own['Depth'] < 90)]

    if len(wells_90) < 7:
        extra = own[(own['Depth'] >= 65) & (own['Depth'] < 150)]
        wells_90 = pd.concat([wells_90, extra]).drop_duplicates()

    if len(wells_90) < 7:
        nearby_wells_90 = c.nearby(20000, region_key, centroid, lambda d: (d >= 65) & (d < 90))
        wells_90 = pd.concat([wells_90, nearby_wells_90]).drop_duplicates()

    if len(wells_90) < 7:
        # the original reuses the 20km keys from the step above
        nearby_wells_150 = c.nearby(20000, region_key, centroid, lambda d: (d >= 65) & (d < 150))
        wells_90 = pd.concat([wells_90, nearby_wells_150]).drop_duplicates()

    return stats(wells_90, mou, c.patch_lte_90)


def s150_stats(c, mou, region_key, centroid):
    own = c.own(region_key)
    wells_150 = own[(own['Depth'] >= 90) & (own['Depth'] < 150)]

    if len(wells_150) < 7:
        extra = own[own['Depth'] >= 90]
        wells_150 = pd.concat([wells_150, extra]).drop_duplicates()

    if len(wells_150) < 7:
        nearby_wells_100k = c.nearby(100000, region_key, centroid, lambda d: d >= 90)
        wells_150 = pd.concat([wells_150, nearby_wells_100k]).drop_duplicates()

    return stats(wells_150, mou, c.patch_gt_90)


def sD_stats(c, mou, region_key, centroid):
    own = c.own(region_key)
    wells_d = own[own['Depth'] >= 150]

    if len(wells_d) < 7:
        nearby_wells_d = c.nearby(100000, region_key, centroid, lambda d: d >= 150)
        wells_d = pd.concat([wells_d, nearby_wells_d]).drop_duplicates()

    return stats(wells_d, mou, c.patch_gt_90)


def mouza_predictions(c, mou, region_key):
    centroid = mou["geometry"].centroid

    # the nearby regions are only reused within a mouza
    c.nearby_cache = {}

    s15_main = s15_stats(c, mou, region_key, centroid)
    s15_flood = s15_flood_stats(c, region_key, centroid)

    if isinstance(s15_main, dict) and isinstance(s15_flood, dict):
        s15_combined = {**s15_main, **s15_flood}
    else:
        s15_combined = s15_main if isinstance(s15_main, dict) else s15_flood

    return {
        "s15": s15_combined,
        "s45": s45_stats(c, mou, region_key, centroid),
        "s65": s65_stats(c, mou, region_key, centroid),
        "s90": s90_stats(c, mou, region_key, centroid),
        "s150": s150_stats(c, mou, region_key, centroid),
        "sD": sD_stats(c, mou, region_key, centroid),
    }


def process_upa(args):
    filename, mouza_batch, output_dir = args
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        return

    c = CONTEXT
    file_data = {}

    for mou, region_key in mouza_batch:
        file_data.setdefault(mou["uni"], {})[mou["mou"]] = mouza_predictions(c, mou, region_key)

    # write then rename, so an interrupted run doesn't leave half written files;
    # the temporary name is per process, so runs working in parallel can't share one
    temporary = f'{filepath}.{os.getpid()}.tmp'
    with open(temporary, "w") as f:
        json.dump(file_data, f, separators=(",", ":"))
    os.replace(temporary, filepath)


def load_geodata():
    mou_topo = gpd.read_file(MOUZA_TOPOJSON)
    if mou_topo.crs is None:
        mou_topo.set_crs(epsg=4326, inplace=True)

    geodata = mou_topo.to_crs(epsg=32645)
    geodata['region_key'] = geodata['div'] + '-' + geodata['dis'] + '-' + geodata['upa'] + '-' + geodata['uni'] + '-' + geodata['mou']
    return geodata


def prediction_jobs(geodata, upazila_files=None):
    with open(DROPDOWN_DATA) as f:
        region_tree = json.load(f)

    geodata_index = geodata.set_index('region_key')
    grouped_jobs = defaultdict(list)

    for div_obj in region_tree:
        div = div_obj["division"]
        for dis_obj in div_obj["districts"]:
            dis = dis_obj["district"]
            for upa_obj in dis_obj["upazilas"]:
                upa = upa_obj["upazila"]
                filename = f"{div}-{dis}-{upa}.json"
                if upazila_files is not None and filename not in upazila_files:
                    continue

                for uni_obj in upa_obj["unions"]:
                    uni = uni_obj["union"]
                    for mou in uni_obj["mouzas"]:
                        region_key = f"{div}-{dis}-{upa}-{uni}-{mou}"
                        grouped_jobs[filename].append((geodata_index.loc[region_key], region_key))

    return list(grouped_jobs.items())


def generate_prediction_data(training_data, output_dir, upazila_files=None, workers=MAX_WORKERS, order='largest-first'):
    """training_data: dataframe with Division,District,Upazila,Union,Mouza,Depth,Arsenic."""
    global CONTEXT

    os.makedirs(output_dir, exist_ok=True)
    geodata = load_geodata()

    td = training_data.copy()
    td['region_key'] = td['Division'] + '-' + td['District'] + '-' + td['Upazila'] + '-' + td['Union'] + '-' + td['Mouza']

    # filter geo data with no corresponding training data
    gd_in_td = geodata[geodata['region_key'].isin(td['region_key'])]

    jobs = prediction_jobs(geodata, upazila_files)
    CONTEXT = Context(td, gd_in_td)

    # biggest upazilas first, so the run doesn't end waiting on one; more runs can
    # work through the list in another order, skipping files that already exist
    if order == 'shuffled':
        np.random.default_rng().shuffle(jobs)
    else:
        jobs.sort(key=lambda job: len(job[1]) if order == 'smallest-first' else -len(job[1]))
    args = [(filename, batch, output_dir) for filename, batch in jobs]

    print(f'Generating {len(args)} upazila files into {output_dir}')
    with mp.get_context('fork').Pool(processes=workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_upa, args), total=len(args)):
            pass
