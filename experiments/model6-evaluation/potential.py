"""
A quick look at how much there is to gain beyond model6 with the new data:
gradient boosting on hand-made spatial features, the cheap version of a CNN
that looks at the wells around each place at several scales.

For each well the features are computed from the training wells only:

    knn     the arsenic of the nearest training wells in the same model6 depth
            band (median, share > 10 and > 50 ppb, distance) for 4, 16, 64 and
            256 neighbours, and of the 16 nearest at any depth
    patch   model6's expert patch: the upazila's risk for wells <= 90m and > 90m
    env     elevation, flood index, distance to river and geology (250m rasters
            from the 3D CNN/PointNet project), and the coordinates

Wells are located by GPS where the data has it (ARRP, BGS) and at their map
mouza's centre otherwise (1Mdataset, Danida). The test wells are the same
held-out wells as in datasets.py (a sample of them), and all tests of a well
are excluded from its neighbours.

Two targets: polluted above 10 ppb, scored as in the dissertation (model6 is
polluted at message 0 or 3 and up), and unsafe above 50 ppb (model6 is unsafe
at risk band 2 and up, as in experiments/BGDEstimates).

    python potential.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import datasets  # noqa: E402
from evaluate import Model, model_estimate_to_risk_assessment  # noqa: E402
from iarsenic_data.common import LOCATION_COLUMNS, MAP_PATH  # noqa: E402
from iarsenic_data.load_data import load_records  # noqa: E402

RASTERS = Path('/home/kane/cnn-iarsenic/Arsenic-Risk-w-3D-CNN-PointNet-/data/rasters')
PATCH_DIR = Path('/home/kane/Default Folder/code/iArsenic2024/preprocessing/models/model6/model_patch')
CRS = 'EPSG:32646'  # the rasters' projection, in metres

DEPTH_BANDS = [15.3, 45, 65, 90, 150]  # model6's strata
KS = [4, 16, 64, 256]
TRAIN_SAMPLE = 400_000
TEST_SAMPLE = {'ARRP': 150_000, 'other': 60_000}
SEED = 0
TARGETS = {'polluted > 10 ppb': 10, 'unsafe > 50 ppb': 50}

FEATURE_SETS = {
    'patch': ['depth', 'patch_lte90', 'patch_gt90', 'patch_for_depth'],
    'env': ['depth', 'x', 'y', 'elevation', 'flood_index', 'river_distance', 'geology'],
    'knn': ['depth'],  # + the knn features, added below
    'knn + patch': ['depth', 'patch_lte90', 'patch_gt90', 'patch_for_depth'],
    'knn + patch + env': ['depth', 'patch_lte90', 'patch_gt90', 'patch_for_depth',
                          'x', 'y', 'elevation', 'flood_index', 'river_distance', 'geology'],
}


def load_wells():
    records = load_records()
    wells = records[records['is_well']].reset_index(drop=True)
    wells['test'] = datasets.is_test(wells)

    # all tests of an ARRP well share its TubeWellCode; other records are their own well
    arrp = (wells['Source'] == datasets.ARRP).to_numpy()
    codes = wells['Source'].astype(str) + ':' + wells['Row'].astype(str)
    codes[arrp] = 'ARRP:' + pd.Series(datasets.arrp_well_codes()[wells['Row'].to_numpy()[arrp]])
    wells['well'] = pd.factorize(codes)[0]

    # GPS where there is one, otherwise the centre of the well's mouza on the map
    mouzas = gpd.read_file(MAP_PATH).to_crs(CRS)
    centres = mouzas.geometry.representative_point()
    key = mouzas[['div', 'dis', 'upa', 'uni', 'mou']].agg('#'.join, axis=1)
    centre_x = pd.Series(centres.x.to_numpy(), index=key)
    centre_y = pd.Series(centres.y.to_numpy(), index=key)
    well_key = wells[LOCATION_COLUMNS].agg('#'.join, axis=1)

    has_gps = wells['Latitude'].notna().to_numpy()
    gps = gpd.GeoSeries(gpd.points_from_xy(wells['Longitude'][has_gps], wells['Latitude'][has_gps]), crs='EPSG:4326').to_crs(CRS)
    wells['x'] = well_key.map(centre_x).to_numpy()
    wells['y'] = well_key.map(centre_y).to_numpy()
    wells.loc[has_gps, 'x'] = gps.x.to_numpy()
    wells.loc[has_gps, 'y'] = gps.y.to_numpy()
    wells['gps'] = has_gps

    wells = wells.dropna(subset=['x', 'y']).reset_index(drop=True)
    wells['depth'] = wells['Depth']
    wells['band'] = np.digitize(wells['Depth'], DEPTH_BANDS)
    for target, ppb in TARGETS.items():
        wells[target] = wells['Arsenic'] > ppb
    return wells


def knn_features(pool, queries):
    """Statistics of each query's nearest pool wells, leaving out its own well's tests."""
    margin = 16
    features = pd.DataFrame(index=queries.index)

    def neighbours(tree_pool, query, k):
        tree = cKDTree(tree_pool[['x', 'y']].to_numpy())
        distance, index = tree.query(query[['x', 'y']].to_numpy(), k=min(k + margin, len(tree_pool)))
        same = tree_pool['well'].to_numpy()[index] == query['well'].to_numpy()[:, None]
        # push the query's own well to the end, keep the k nearest others
        order = np.argsort(same, axis=1, kind='stable')
        index = np.take_along_axis(index, order, axis=1)[:, :k]
        distance = np.take_along_axis(distance, order, axis=1)[:, :k]
        return tree_pool['Arsenic'].to_numpy()[index], distance

    for band in range(len(DEPTH_BANDS) + 1):
        in_band = queries['band'] == band
        if not in_band.any():
            continue
        band_pool = pool[pool['band'] == band]
        arsenic, distance = neighbours(band_pool, queries[in_band], max(KS))
        for k in KS:
            features.loc[in_band, f'band_k{k}_median'] = np.median(arsenic[:, :k], axis=1)
            features.loc[in_band, f'band_k{k}_gt50'] = (arsenic[:, :k] > 50).mean(axis=1)
            features.loc[in_band, f'band_k{k}_gt10'] = (arsenic[:, :k] > 10).mean(axis=1)
            features.loc[in_band, f'band_k{k}_distance'] = np.log1p(distance[:, k - 1])

    arsenic, distance = neighbours(pool, queries, 16)
    features['any_k16_median'] = np.median(arsenic, axis=1)
    features['any_k16_gt50'] = (arsenic > 50).mean(axis=1)
    features['any_k16_distance'] = np.log1p(distance[:, -1])
    return features


def patch_features(wells):
    def lookup(name):
        patch = pd.read_csv(PATCH_DIR / name)
        return patch.set_index(['Division', 'District', 'Upazila'])['Risk'].to_dict()

    lte90, gt90 = lookup('arsenic_statistics_lte_90m_v2.csv'), lookup('arsenic_statistics_gt_90m_v2.csv')
    key = list(zip(wells['Division'].str.lower(), wells['District'].str.lower(), wells['Upazila'].str.lower()))
    features = pd.DataFrame(index=wells.index)
    features['patch_lte90'] = [lte90.get(k, np.nan) for k in key]
    features['patch_gt90'] = [gt90.get(k, np.nan) for k in key]
    features['patch_for_depth'] = np.where(wells['depth'] <= 90, features['patch_lte90'], features['patch_gt90'])
    return features


def env_features(wells):
    features = pd.DataFrame(index=wells.index)
    for column, name in [('elevation', 'elevation_250m.tif'), ('flood_index', 'flood_index_250m.tif'),
                         ('river_distance', 'RiverDistance_250m.tif'), ('geology', 'geology_specific_250m.tif')]:
        with rasterio.open(RASTERS / name) as raster:
            values = np.array([v[0] for v in raster.sample(zip(wells['x'], wells['y']))], dtype=float)
            values[values == raster.nodata] = np.nan
        features[column] = values
    return features


def binary_scores(truth, guess):
    tp, tn = (truth & guess).sum(), (~truth & ~guess).sum()
    fp, fn = (~truth & guess).sum(), (truth & ~guess).sum()
    precision = tp / (tp + fp) if tp + fp else np.nan
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    return {
        'accuracy': (tp + tn) / len(truth), 'precision': precision, 'sensitivity': sensitivity,
        'specificity': tn / (tn + fp) if tn + fp else np.nan,
        'f1': 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else np.nan,
    }


def best_threshold(truth, probability):
    thresholds = np.linspace(0.05, 0.95, 91)
    f1 = [binary_scores(truth, probability >= t)['f1'] for t in thresholds]
    return thresholds[int(np.nanargmax(f1))]


def model6_messages(run_dir, wells):
    model = Model(HERE / 'runs' / run_dir)
    messages = []
    for row in wells.itertuples(index=False):
        result = model.estimate(row.Division, row.District, row.Upazila, row.Union, row.Mouza, row.depth, 'red')
        messages.append(np.nan if result is None else result[0])
    return np.array(messages, dtype=float)


def model6_positive(messages, target):
    if target == 'polluted > 10 ppb':
        return (messages == 0) | (messages >= 3)
    return np.array([model_estimate_to_risk_assessment(int(m)) >= 2 for m in messages])


def main():
    rng = np.random.default_rng(SEED)
    wells = load_wells()
    print(f'{len(wells):,} wells, {wells.gps.mean():.0%} with GPS')

    pool = wells[~wells['test']]
    train = pool.iloc[rng.choice(len(pool), TRAIN_SAMPLE, replace=False)]
    test_wells = wells[wells['test']]
    is_arrp = test_wells['Source'] == datasets.ARRP
    test = pd.concat([
        test_wells[is_arrp].sample(TEST_SAMPLE['ARRP'], random_state=SEED),
        test_wells[~is_arrp].sample(min(TEST_SAMPLE['other'], (~is_arrp).sum()), random_state=SEED),
    ])

    samples = pd.concat([train.assign(split='train'), test.assign(split='test')])
    print('computing features')
    knn = knn_features(pool, samples)
    features = pd.concat([samples, knn, patch_features(samples), env_features(samples)], axis=1)
    features['geology'] = features['geology'].astype('category')
    knn_columns = list(knn.columns)

    train_features = features[features['split'] == 'train']
    test_features = features[features['split'] == 'test']

    # a validation part of the training sample picks each model's threshold
    validation = rng.random(len(train_features)) < 0.2
    results, importances = [], {}

    def record(target, name, test_set, truth, guess, probability=None):
        row = {'target': target, 'model': name, 'test set': test_set, 'wells': len(truth), **binary_scores(truth, guess)}
        row['auc'] = roc_auc_score(truth, probability) if probability is not None else np.nan
        results.append(row)

    test_sets = {
        'holdout: ARRP': (test_features['Source'] == datasets.ARRP).to_numpy(),
        'holdout: non-ARRP sources': (test_features['Source'] != datasets.ARRP).to_numpy(),
    }

    model6_runs = {}
    for run in ['A', 'C']:
        if len(list((HERE / 'runs' / run).glob('*.json'))) < 538:
            print(f'model6 {run} has not finished, leaving it out')
            continue
        model6_runs[run] = model6_messages(run, test_features)

    for target, ppb in TARGETS.items():
        for name, columns in FEATURE_SETS.items():
            columns = columns + (knn_columns if 'knn' in name else [])
            model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
            model.fit(train_features.loc[~validation, columns], train_features.loc[~validation, target])
            threshold = best_threshold(train_features.loc[validation, target].to_numpy(),
                                       model.predict_proba(train_features.loc[validation, columns])[:, 1])
            probability = model.predict_proba(test_features[columns])[:, 1]
            for test_set, mask in test_sets.items():
                truth = test_features[target].to_numpy()[mask]
                record(target, f'GBM: {name}', test_set, truth, probability[mask] >= threshold, probability[mask])
            importances[f'{target}: {name}'] = pd.Series(model.booster_.feature_importance('gain'), index=columns)
            print(f'GBM {target} {name}: done')

        # plain nearest neighbours, no learning: positive if the median of the 16 nearest in the depth band is
        for test_set, mask in test_sets.items():
            truth = test_features[target].to_numpy()[mask]
            median = test_features['band_k16_median'].to_numpy()[mask]
            record(target, '16 nearest wells in the depth band (no model)', test_set, truth, median > ppb, median)

        # model6, on the same wells
        for run, messages in model6_runs.items():
            for test_set, mask in test_sets.items():
                has = mask & ~np.isnan(messages)
                record(target, f'model6 {run}', test_set, test_features[target].to_numpy()[has],
                       model6_positive(messages[has], target))

    results = pd.DataFrame(results)
    results.to_csv(HERE / 'runs' / 'potential-results.csv', index=False)
    pd.DataFrame(importances).to_csv(HERE / 'runs' / 'potential-importance.csv')

    pd.set_option('display.width', 200)
    print(results.round(3).to_string(index=False))
    print(pd.DataFrame(importances)['polluted > 10 ppb: knn + patch + env'].sort_values(ascending=False).head(15).round(0))


if __name__ == '__main__':
    main()
