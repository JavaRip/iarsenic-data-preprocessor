"""
Scores model6 runs on the held-out wells and on the independent test sets,
the same way the iArsenic experiments do (experiments/BGDEstimates):
produceEstimate.py's prediction rules, model_estimate_to_risk_assessment and
expected_risk, binary unsafe at risk >= 3 (> 100 ppb) as in binary_cm.py, and
also at the Bangladesh standard (> 50 ppb, risk >= 2).

The first score is the dissertation's (UoP-SoftEng-Dissertation, which reported
model5 at f1 0.82): a well is polluted above 10 ppb (gen_labels), and a
prediction is polluted when model5/6's severity is polluted or highlyPolluted
(message 3 and up) or there isn't enough data (message 0).

Held-out wells have no staining information, so they are predicted as red
stained wells (the prediction that uses the training data); a black stained
well is always predicted safe whatever the data.

    python evaluate.py runs/A runs/B runs/C
"""

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from iarsenic_data.gps import locate, map_names  # noqa: E402

IARSENIC = Path('/home/kane/Default Folder/code/iArsenic2024')
BGD_TRAVERSE = IARSENIC / 'experiments/BGDEstimates/output/BGD_Traverse_with_regions.csv'
VGQD = IARSENIC / 'data/vgqd/vgqd-all-data.csv'
UNIONS = IARSENIC / 'preprocessing/models/model6/output/geodata/intersected_unions.geojson'
TEST_WELLS = HERE / 'data' / 'test.csv'

VALID_STAINS = {"Red": "red", "Red?": "red", "Black": "black", "Black?": "black"}


# from experiments/BGDEstimates/script.py
def model_estimate_to_risk_assessment(model_estimate):
    if model_estimate == 0:
        return 2
    elif model_estimate == 1:
        return 0
    elif model_estimate in (2, 3):
        return 1
    elif model_estimate in (4, 5):
        return 2
    elif model_estimate == 6:
        return 3
    elif model_estimate in (7, 8):
        return 4
    else:
        return 2


def expected_risk(as_ppb):
    if as_ppb < 10:
        return 0
    elif as_ppb <= 50:
        return 1
    elif as_ppb <= 100:
        return 2
    elif as_ppb <= 150:
        return 3
    else:
        return 4


class Model:
    """A model6 run directory, with produceEstimate.py's prediction rules."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.files = {}

    def file(self, div, dis, upa):
        filename = f"{div}-{dis}-{upa}.json"
        if filename not in self.files:
            path = self.directory / filename
            self.files[filename] = json.loads(path.read_text()) if path.exists() else None
        return self.files[filename]

    def estimate(self, div, dis, upa, uni, mou, depth, staining, flooding=False):
        """(estimate, used patch) or None when the model has no data for the place."""
        data = self.file(div, dis, upa)
        model_data = data.get(uni, {}).get(mou) if data is not None else None
        if model_data is None:
            return None

        if depth < 15.3:
            key = "s15"
        elif depth < 45:
            key = "s45"
        elif depth < 65:
            key = "s65"
        elif depth < 90:
            key = "s90"
        elif depth < 150:
            key = "s150"
        else:
            key = "sD"

        stratum = model_data.get(key, {})
        patched = list(stratum.keys()) == ['m']

        if key == "s15" and "m2" in stratum:
            if staining == "black":
                return stratum["m2"], patched
            elif flooding and "m9" in stratum:
                return stratum["m9"], patched
            elif "m7" in stratum:
                return stratum["m7"], patched
            return None
        if staining == "black":
            return 1, patched
        if staining == "red":
            return (stratum["m"], patched) if "m" in stratum else None
        return 0, patched


def predict(model, wells):
    """Model message (NaN where the model has no prediction) and patch use."""
    messages, patched = [], []
    for row in wells.itertuples(index=False):
        result = model.estimate(row.div, row.dis, row.upa, row.uni, row.mou, row.depth, row.staining)
        if result is None:
            messages.append(np.nan)
            patched.append(np.nan)
        else:
            messages.append(result[0])
            patched.append(result[1])
    return np.array(messages, dtype=float), np.array(patched, dtype=float)


def binary_scores(expected, predicted, threshold):
    truth = np.asarray(expected) >= threshold
    guess = np.asarray(predicted) >= threshold
    tp = int((truth & guess).sum())
    tn = int((~truth & ~guess).sum())
    fp = int((~truth & guess).sum())
    fn = int((truth & ~guess).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    return {
        'accuracy': (tp + tn) / len(truth) if len(truth) else np.nan,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': tn / (tn + fp) if tn + fp else np.nan,
        'f1': 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else np.nan,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def scores(wells, messages, patched):
    has_prediction = ~np.isnan(messages)
    expected = wells['expected'].to_numpy()[has_prediction]
    messages = messages[has_prediction]
    predicted = np.array([model_estimate_to_risk_assessment(int(m)) for m in messages], dtype=float)

    result = {
        'wells': len(wells),
        'coverage': has_prediction.mean(),
        'patch_used': np.nanmean(patched) if has_prediction.any() else np.nan,
        'risk_band_accuracy': (expected == predicted).mean() if has_prediction.any() else np.nan,
    }

    # the dissertation's evaluation
    polluted = wells['arsenic'].to_numpy()[has_prediction] > 10
    predicted_polluted = (messages == 0) | (messages >= 3)
    for metric, value in binary_scores(polluted, predicted_polluted, True).items():
        result[f'gt10_{metric}'] = value

    for name, threshold in [('gt100', 3), ('gt50', 2)]:
        for metric, value in binary_scores(expected, predicted, threshold).items():
            result[f'{name}_{metric}'] = value
    return result


def holdout_wells():
    test = pd.read_csv(
        TEST_WELLS,
        dtype={c: str for c in ['Division', 'District', 'Upazila', 'Union', 'Mouza']},
        keep_default_na=False,
        na_values={'Depth': [''], 'Arsenic': ['']},
    )
    test = test.dropna(subset=['Depth', 'Arsenic'])
    wells = pd.DataFrame({
        'div': test.Division, 'dis': test.District, 'upa': test.Upazila, 'uni': test.Union, 'mou': test.Mouza,
        'depth': test.Depth, 'staining': 'red', 'arsenic': test.Arsenic, 'source': test.Source,
    })
    wells['expected'] = wells['arsenic'].map(expected_risk)
    return wells


def bgd_traverse_wells():
    # locations from the iArsenic region-from-point API, as script.py used them
    df = pd.read_csv(BGD_TRAVERSE)
    df['staining'] = df['Stain'].astype(str).str.strip().map(VALID_STAINS)
    df = df.dropna(subset=['staining', 'Depth_m', 'As_ppb'])
    wells = pd.DataFrame({
        'div': df['coord-div'].fillna(df['N_DIVISION']), 'dis': df['coord-dis'].fillna(df['N_DISTRICT']),
        'upa': df['coord-upa'].fillna(df['N_UPAZILA']), 'uni': df['coord-uni'].fillna(df['N_UNION']),
        'mou': df['coord-mou'].fillna(df['N_MAUZA']),
        'depth': df['Depth_m'].astype(float), 'staining': df['staining'], 'arsenic': df['As_ppb'],
    })
    wells['expected'] = wells['arsenic'].map(expected_risk)
    return wells


def vgqd_wells():
    # vgqdEstimates/script.py locates each well at its union's centroid
    df = pd.read_csv(VGQD).dropna(subset=['depth', 'colour', 'as'])
    unions = gpd.read_file(UNIONS)
    for column in ['div', 'dis', 'upa', 'uni']:
        df[column] = df[column].astype(str).str.strip()
        unions[column] = unions[column].astype(str).str.strip()

    unions = unions.drop_duplicates(['div', 'dis', 'upa', 'uni'])
    merged = df.merge(unions[['div', 'dis', 'upa', 'uni', 'geometry']], on=['div', 'dis', 'upa', 'uni'], how='left')
    centroids = gpd.GeoSeries(merged['geometry'], crs=unions.crs).to_crs('EPSG:4326').centroid
    found = merged['geometry'].notna().to_numpy()

    rows = np.full(len(merged), -1)
    rows[found] = locate(centroids[found].y.to_numpy(), centroids[found].x.to_numpy())
    names = np.full((len(merged), 5), None, dtype=object)
    names[rows >= 0] = map_names(rows[rows >= 0])

    wells = pd.DataFrame({
        'div': names[:, 0], 'dis': names[:, 1], 'upa': names[:, 2], 'uni': names[:, 3], 'mou': names[:, 4],
        'depth': merged['depth'].astype(float), 'staining': merged['colour'].str.lower(), 'arsenic': merged['as'],
    })
    wells['expected'] = wells['arsenic'].map(expected_risk)
    return wells


def evaluate(run_dirs):
    test_sets = {}
    holdout = holdout_wells()
    test_sets['holdout: all sources'] = holdout
    test_sets['holdout: non-ARRP sources'] = holdout[holdout.source != 'ARRP_arsenic_master_all_data']
    test_sets['holdout: ARRP'] = holdout[holdout.source == 'ARRP_arsenic_master_all_data']
    bgd = bgd_traverse_wells()
    test_sets['BGD Traverse'] = bgd
    test_sets['BGD Traverse: red stained'] = bgd[bgd.staining == 'red']
    vgqd = vgqd_wells()
    test_sets['VGQD'] = vgqd
    test_sets['VGQD: red stained'] = vgqd[vgqd.staining == 'red']

    results = []
    predictions = {}
    for run_dir in run_dirs:
        model = Model(run_dir)
        name = Path(run_dir).name
        for test_name, wells in test_sets.items():
            messages, patched = predict(model, wells)
            predictions[(name, test_name)] = messages
            results.append({'model': name, 'test set': test_name, **scores(wells, messages, patched)})
            print(f'{name} {test_name}: done')

    return pd.DataFrame(results), test_sets, predictions


def main():
    results, test_sets, predictions = evaluate(sys.argv[1:])
    output = HERE / 'runs' / 'results.csv'
    results.to_csv(output, index=False)

    pd.set_option('display.width', 250)
    columns = ['model', 'test set', 'wells', 'coverage', 'gt10_accuracy', 'gt10_precision', 'gt10_sensitivity',
               'gt10_specificity', 'gt10_f1', 'gt50_f1', 'gt100_f1', 'risk_band_accuracy']
    print(results[columns].round(3).to_string(index=False))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
