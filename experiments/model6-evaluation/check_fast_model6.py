"""
Checks that fast_model6.py gives byte-identical model files to the original
model6 code (utils/generate_prediction_data.py in the iArsenic repo), for a
sample of upazilas.

    python check_fast_model6.py <training-data.csv> <work dir> <upazila file>...
"""

import filecmp
import multiprocessing as mp
import os
import sys
import time

import fast_model6
from datasets import read_wells

sys.path.insert(0, fast_model6.MODEL6_DIR)
from utils import generate_prediction_data as original  # noqa: E402


def run_original(job):
    filename, batch = job
    start = time.time()
    original.process_upa((filename, batch))
    return filename, len(batch), time.time() - start


def main():
    training_csv, work_dir, *upazila_files = sys.argv[1:]
    work_dir = os.path.abspath(work_dir)

    td = read_wells(training_csv)
    td['region_key'] = td['Division'] + '-' + td['District'] + '-' + td['Upazila'] + '-' + td['Union'] + '-' + td['Mouza']

    geodata = fast_model6.load_geodata()
    gd_in_td = geodata[geodata['region_key'].isin(td['region_key'])]
    jobs = fast_model6.prediction_jobs(geodata, set(upazila_files))

    # the original works relative to the model6 directory: model_patch/, logs/, m150_10/
    original_dir = os.path.join(work_dir, 'original')
    os.makedirs(os.path.join(original_dir, 'm150_10'), exist_ok=True)
    os.makedirs(os.path.join(original_dir, 'logs', 'generate_prediction_data'), exist_ok=True)
    if not os.path.exists(os.path.join(original_dir, 'model_patch')):
        os.symlink(os.path.join(fast_model6.MODEL6_DIR, 'model_patch'), os.path.join(original_dir, 'model_patch'))
    os.chdir(original_dir)

    print(f'Original model6 code on {len(jobs)} upazilas')
    original.init_worker(gd_in_td, td)
    with mp.get_context('fork').Pool(processes=len(jobs)) as pool:
        for filename, mouzas, seconds in pool.imap_unordered(run_original, jobs):
            print(f'  {filename}: {mouzas} mouzas in {seconds:.0f}s')

    print('Fast copy')
    start = time.time()
    fast_dir = os.path.join(work_dir, 'fast')
    fast_model6.generate_prediction_data(td.drop(columns='region_key'), fast_dir, set(upazila_files), workers=len(jobs))
    print(f'  {time.time() - start:.0f}s')

    identical = True
    for filename, _ in jobs:
        same = filecmp.cmp(os.path.join(original_dir, 'm150_10', filename), os.path.join(fast_dir, filename), shallow=False)
        identical &= same
        print(f'{"IDENTICAL" if same else "DIFFERENT"}: {filename}')
    print('All identical' if identical else 'NOT IDENTICAL')


if __name__ == '__main__':
    main()
