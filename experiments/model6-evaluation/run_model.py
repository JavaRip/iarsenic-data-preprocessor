"""
Runs model6 (fast_model6.py) on a training data csv.

    python run_model.py data/C-train.csv runs/C [workers] [largest-first|smallest-first|shuffled]
"""

import sys
import time

import fast_model6
from datasets import read_wells


def main():
    training_csv, output_dir = sys.argv[1:3]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else fast_model6.MAX_WORKERS
    order = sys.argv[4] if len(sys.argv) > 4 else 'largest-first'

    start = time.time()
    fast_model6.generate_prediction_data(read_wells(training_csv), output_dir, workers=workers, order=order)
    print(f'Done in {(time.time() - start) / 60:.1f} minutes')


if __name__ == '__main__':
    main()
