"""
Locating wells on the mouza map by their coordinates.

The map is data/geodata/mouzas.geojson, iArsenic model6's
output/geodata/intersected_mouzas.geojson, so the names given to located
wells are exactly the names model6 joins the training data on.

A region corrected to ~gps~ gets its division, district, upazila and union
from the mouza polygon each well is in. The mouza is the well's own mouza
name if that union on the map has a mouza of that name (ignoring case),
otherwise the mouza polygon the well is in. Well coordinates and mouza
borders are both approximate, so the name is trusted when it fits.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from iarsenic_data.common import GPS_COLUMNS, LOCATION_COLUMNS, MAP_PATH

MAP_COLUMNS = ['div', 'dis', 'upa', 'uni', 'mou']

# a generous box around Bangladesh, to throw away (0, 0) and swapped coordinates
LATITUDE_RANGE = (20.0, 27.0)
LONGITUDE_RANGE = (88.0, 93.0)

CHUNK_SIZE = 500_000


@lru_cache(maxsize=None)
def load_map():
    import geopandas as gpd

    if not MAP_PATH.exists():
        raise SystemExit(f'{MAP_PATH} does not exist, copy model6 output/geodata/intersected_mouzas.geojson there')

    print(f'Loading mouza map {MAP_PATH.name}')
    mouzas = gpd.read_file(MAP_PATH)[[*MAP_COLUMNS, 'geometry']].to_crs('EPSG:4326')
    mouzas.sindex  # build the spatial index once
    return mouzas


def has_coordinates(df):
    return all(column in df for column in GPS_COLUMNS)


def valid_coordinates(df):
    if not has_coordinates(df):
        return np.zeros(len(df), dtype=bool)

    latitude = df['Latitude'].to_numpy(dtype=float)
    longitude = df['Longitude'].to_numpy(dtype=float)
    return (
        (latitude >= LATITUDE_RANGE[0]) & (latitude <= LATITUDE_RANGE[1])
        & (longitude >= LONGITUDE_RANGE[0]) & (longitude <= LONGITUDE_RANGE[1])
    )


def locate(latitude, longitude):
    """Row of the map's mouza polygon each point is in, -1 if none."""
    import shapely

    mouzas = load_map()
    coordinates = pd.DataFrame({'lat': latitude, 'lon': longitude})
    unique = coordinates.drop_duplicates().reset_index(drop=True)

    found = np.full(len(unique), -1, dtype=np.int64)
    for start in range(0, len(unique), CHUNK_SIZE):
        chunk = unique.iloc[start:start + CHUNK_SIZE]
        points = shapely.points(chunk['lon'].to_numpy(), chunk['lat'].to_numpy())
        point_index, mouza_index = mouzas.sindex.query(points, predicate='within')

        # a point on a shared border is in two polygons, keep the first
        first = pd.Series(mouza_index).groupby(point_index).first()
        found[start + first.index.to_numpy()] = first.to_numpy()

    lookup = unique.assign(row=found)
    return coordinates.merge(lookup, on=['lat', 'lon'], how='left')['row'].to_numpy()


def map_names(rows):
    """The map's names for the given rows, as an array of [div, dis, upa, uni, mou]."""
    return load_map()[MAP_COLUMNS].to_numpy()[rows]


def assign_locations(records):
    """
    Location names for records corrected to ~gps~, as a dataframe with the
    location columns, '' where a record has no usable coordinates or is
    outside the map.
    """
    located = pd.DataFrame('', index=records.index, columns=LOCATION_COLUMNS)

    valid = valid_coordinates(records)
    rows = np.full(len(records), -1, dtype=np.int64)
    if valid.any():
        rows[valid] = locate(records['Latitude'].to_numpy()[valid], records['Longitude'].to_numpy()[valid])

    inside = rows >= 0
    names = map_names(rows[inside])
    located.loc[inside, LOCATION_COLUMNS] = names

    # use the well's own mouza name when its union on the map has that mouza
    mouzas = load_map()
    union_key = mouzas[MAP_COLUMNS[:4]].agg('#'.join, axis=1)
    by_name = pd.Series(mouzas['mou'].to_numpy(), index=union_key + '#' + mouzas['mou'].str.casefold())
    by_name = by_name[~by_name.index.duplicated()]

    union_of_record = pd.Series(names[:, :4].tolist(), dtype=object).str.join('#')
    own_mouza = (union_of_record + '#' + pd.Series(records['Mouza'].to_numpy()[inside]).str.casefold()).map(by_name)
    located.loc[inside, 'Mouza'] = np.where(own_mouza.notna(), own_mouza, names[:, 4])

    print(
        f'GPS located {inside.sum()} records '
        f'({(own_mouza.notna()).sum()} by their mouza name in the union they are in), '
        f'dropped {(~valid).sum()} without usable coordinates and {(valid & ~inside).sum()} outside the map'
    )
    return located


NO_COORDINATES = -2
OUTSIDE_MAP = -1


def locate_records(records):
    """
    Row of the map's mouza polygon each record is in, OUTSIDE_MAP or
    NO_COORDINATES when it can't be located.
    """
    rows = np.full(len(records), NO_COORDINATES, dtype=np.int64)
    valid = valid_coordinates(records)
    if valid.any():
        rows[valid] = locate(records['Latitude'].to_numpy()[valid], records['Longitude'].to_numpy()[valid])
    return rows
