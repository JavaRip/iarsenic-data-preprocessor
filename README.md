# iArsenic training data

Builds the iArsenic training data (wells with their region, depth and arsenic
level) from the source CSV files. This replaces the Node loader in the iArsenic
repo (`preprocessing/lib/load-data.js`, `preprocessing/cli/csv-to-json.js` and
`preprocessing/data-import/find-name-corrections.js`), which could not handle
the size of the ARRP data.

The output is identical to the old loader: `output/training-data.json` is byte for
byte what `csv-to-json.js` wrote, so model6 can use it without changes.

## Setup

Requires python 3.10+ and poetry.

```shell
poetry install
```

## Adding a data set

1. Copy the CSV into `data/sources/`.

2. Create its preprocessor, which turns the file into the standard format:

   ```shell
   poetry run python prep.py new my-new-data.csv
   ```

   This creates `preprocessors/my-new-data.py`. Edit `preprocess()` so it returns the
   columns `Division,District,Upazila,Union,Mouza,Depth,Arsenic`, with depth in
   metres and arsenic in µg/L (rename columns, convert units, etc.). Document the
   source of the data at the top of the file.

3. Find name corrections, so that the region names match the existing data:

   ```shell
   poetry run python prep.py corrections my-new-data.csv
   ```

   For every region whose name isn't in the other sources, you choose the correct
   spelling from a list, or press enter to skip the region (its wells are dropped).
   Choices are saved to `data/name-corrections/my-new-data-corrections.csv` as you go,
   so you can stop with ctrl-c and carry on later. A progress bar shows how many
   of the source's regions have been checked.

   The correct names are those in `mouza-names.csv` (the names in the mouza map),
   which is the reference we have. Use `-r` to choose other reference sources, but
   don't use data sets that have their own name corrections (like `1Mdataset`):
   corrections aren't chained, so choosing one of their misspellings leaves the
   wells unmatched.

   To make the choices quicker:
   * names that match exactly one correct name in the same parent region apart
     from case (e.g. `JESSORE` and `Jessore`) are corrected without asking and
     saved the same way. If more than one name matches (e.g. `Cox's Bazar` and
     `Cox'S Bazar`) you are asked
   * every option shows how many of the incorrect region's subregions it also has
     (e.g. `Kotwali 14/15 shared`), and the option with the most is marked as
     suggested; subregion names are compared ignoring case

   `--exact-case` turns these off, as the old JS tool behaved.

4. Optionally add the file to `data/sources/order.txt`, which sets the order the
   sources are read in. Files not listed are read after, in name order. The order
   only changes the order of regions and wells in the output.

5. Build the training data:

   ```shell
   poetry run python prep.py build
   ```

   This writes `output/training-data.json` (the format model6 reads) and
   `output/training-data.csv` (the same wells as a flat table, in the order model6's
   `load_training_df.py` produces). Use `-s` to build from some sources only.

## How the data is processed

Same as the old Node loader:

* every source is read through its preprocessor
* every name correction in `data/name-corrections/` is applied to every source;
  corrections are matched on the longest part of the path
  `Division#District#Upazila#Union#Mouza` that has one, and a correction to `~none~`
  drops the records. If two files correct the same path differently, the first file
  (by name) wins and a warning is printed
* records without all five region names are dropped
* records without a numeric depth and arsenic level add their region but no well
  (this is how `mouza-names.csv` adds the known region names)

## Data

`data/sources/`

* `1Mdataset.csv`: the national screening data set, about 1M wells
* `BGS_Ravenscroft.csv`: (BGS, 2001) and (Ravenscroft et al. 2018)
* `Danida-20220514.csv`: 160k wells from the coastal areas, produced by Danida
* `mouza-names.csv`: all known region names, no wells
* `ARRP_arsenic_master_all_data.csv`: the raw ARRP export from DPHE, about 6.5M wells.
  Not in git: it is too large and contains personal data (names, phone numbers).
  Depths are converted from feet to metres in its preprocessor.

`data/name-corrections/`: name corrections, as `path,correct` CSV files.
ARRP has none yet, so most of its regions don't match the other sources.

## Layout

```
prep.py                  command line: new, corrections, build
preprocessors/           one preprocessor per source file, same name as the file
iarsenic_data/
  common.py              paths, standard columns, CSV reading
  sources.py             finds the source files and their preprocessors
  name_corrections.py    port of lib/name-corrections.js
  load_data.py           port of lib/load-data.js
  find_corrections.py    port of data-import/find-name-corrections.js
  output.py              writes training-data.json and training-data.csv
```
