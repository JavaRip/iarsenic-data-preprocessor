"""
Finds the input CSV files and their preprocessors.

Each file data/sources/<name>.csv must have a preprocessor
preprocessors/<name>.py with a preprocess(path) function returning the
standard format (see common.py).

Files are read in the order listed in data/sources/order.txt (one file name
per line), then any others in sorted name order, as data/*.csv was in the
JS loader. The order only changes the order of regions and wells in the output.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from iarsenic_data.common import PREPROCESSORS_DIR, SOURCES_DIR, standardise

ORDER_FILE = SOURCES_DIR / 'order.txt'


@dataclass
class Source:
    name: str
    path: Path

    @property
    def preprocessor_path(self):
        return PREPROCESSORS_DIR / f'{self.name}.py'

    def load(self):
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f'No preprocessor for {self.path.name}, create one with:\n'
                f'  python prep.py new {self.name}'
            )

        spec = importlib.util.spec_from_file_location(f'preprocessors.{self.name}', self.preprocessor_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return standardise(module.preprocess(self.path))


def list_sources():
    # sorted by code point, like the shell glob data/*.csv was
    paths = sorted(SOURCES_DIR.glob('*.csv'))

    if ORDER_FILE.exists():
        order = [line.strip() for line in ORDER_FILE.read_text().splitlines() if line.strip()]
        missing = [name for name in order if not (SOURCES_DIR / name).exists()]
        if missing:
            raise SystemExit(f'{ORDER_FILE} lists files that are not in data/sources: {missing}')
        paths.sort(key=lambda p: order.index(p.name) if p.name in order else len(order))

    return [Source(p.stem, p) for p in paths]


def get_sources(names=None):
    sources = list_sources()
    if names is None:
        return sources

    by_name = {s.name: s for s in sources}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit(f'Unknown sources {unknown}, available: {list(by_name)}')
    return [by_name[n] for n in names]
