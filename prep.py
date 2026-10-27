"""
Prepares the iArsenic training data.

    python prep.py new <file>             create a preprocessor for data/sources/<file>.csv
    python prep.py corrections <file>     find name corrections for a source, interactively
    python prep.py build                  write output/training-data.json and .csv

See README.md for the full workflow.
"""

import argparse
import sys
from pathlib import Path

from iarsenic_data.common import NAME_CORRECTIONS_DIR, OUTPUT_DIR, PREPROCESSORS_DIR, SOURCES_DIR
from iarsenic_data.sources import get_sources

# the list of known correct region names, from the mouza map
DEFAULT_REFERENCE = 'mouza-names'


def source_name(value):
    # accept a name, a file name or a path
    return Path(value).stem


def cmd_new(args):
    name = source_name(args.source)
    csv_path = SOURCES_DIR / f'{name}.csv'
    target = PREPROCESSORS_DIR / f'{name}.py'

    if not csv_path.exists():
        sys.exit(f'{csv_path} does not exist, copy the file into data/sources first')
    if target.exists():
        sys.exit(f'{target} already exists')

    with open(csv_path, encoding='utf-8') as f:
        columns = f.readline().strip()

    template = (PREPROCESSORS_DIR / '_template.py').read_text()
    target.write_text(template.format(filename=csv_path.name, columns=columns))
    print(f'Created {target}')


def cmd_corrections(args):
    from iarsenic_data.find_corrections import find_corrections

    name = source_name(args.source)
    get_sources([name])

    if args.reference:
        reference = [source_name(r) for r in args.reference]
        get_sources(reference)
    else:
        reference = [DEFAULT_REFERENCE]
        print(f'Using {DEFAULT_REFERENCE}.csv as the reference for correct names (use -r to change)')

    if name in reference:
        sys.exit(f'{name} cannot be its own reference')

    output = args.output or NAME_CORRECTIONS_DIR / f'{name}-corrections.csv'
    find_corrections(name, reference, output, exact_case=args.exact_case)


def cmd_build(args):
    from iarsenic_data.load_data import load_records
    from iarsenic_data.output import write_csv, write_legacy_json

    records = load_records(
        source_names=args.sources and [source_name(s) for s in args.sources],
        corrections=not args.no_corrections,
        correction_files=args.corrections,
    )

    if 'json' in args.format:
        write_legacy_json(records, args.output_dir / 'training-data.json')
    if 'csv' in args.format:
        write_csv(records, args.output_dir / 'training-data.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    new = commands.add_parser('new', help='create a preprocessor for a source file')
    new.add_argument('source', help='file in data/sources, e.g. ARRP_arsenic_master_all_data.csv')
    new.set_defaults(run=cmd_new)

    corrections = commands.add_parser('corrections', help='find name corrections for a source')
    corrections.add_argument('source', help='file in data/sources to correct')
    corrections.add_argument('-r', '--reference', nargs='+', help='sources with the correct names (default: mouza-names)')
    corrections.add_argument('-o', '--output', type=Path, help='corrections file (default: data/name-corrections/<source>-corrections.csv)')
    corrections.add_argument('--exact-case', action='store_true', help='compare names exactly, as the JS tool did (no automatic case corrections or shared counts)')
    corrections.set_defaults(run=cmd_corrections)

    build = commands.add_parser('build', help='build the training data')
    build.add_argument('-s', '--sources', nargs='+', help='sources to use (default: all in data/sources)')
    build.add_argument('-c', '--corrections', nargs='+', help='correction files (default: data/name-corrections/*.csv)')
    build.add_argument('--no-corrections', action='store_true', help='do not apply name corrections')
    build.add_argument('-f', '--format', nargs='+', choices=['json', 'csv'], default=['json', 'csv'])
    build.add_argument('-o', '--output-dir', type=Path, default=OUTPUT_DIR)
    build.set_defaults(run=cmd_build)

    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
