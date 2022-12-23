import os
from typing import Union, List

import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.utils.python_utils import ensure_list

from .. import config


class PvUS(PandasDataset):
    r"""Simulated solar power production from more than 5,000 photovoltaic
    plants in the US.

    Data are provided by `National Renewable Energy Laboratory (NREL)
    <https://www.nrel.gov/>`_'s `Solar Power Data for Integration Studies
    <https://www.nrel.gov/grid/solar-power-data.html>`_. Original raw data
    consist of 1 year (2006) of 5-minute solar power (in MW) for approximately
    5,000 synthetic PV plants in the United States.

    Preprocessed data are resampled in 10-minutes intervals taking the average.
    The entire dataset contains 5016 plants, divided in two macro zones (east
    and west). The "east" zone contains 4084 plants, the "west" zone has 1082
    plants. Some states appear in both zones, with plants at same geographical
    position. When loading the entire datasets, duplicated plants in "east" zone
    are dropped.
    """

    available_zones = ['east', 'west']
    similarity_options = {'distance', 'correntropy'}

    def __init__(self, zones: Union[str, List] = None, mask_zeros: bool = False,
                 root: str = None, freq: str = None):
        # set root path
        if zones is None:
            zones = self.available_zones
        else:
            zones = ensure_list(zones)
            if not set(zones).issubset(self.available_zones):
                invalid_zones = set(zones).difference(self.available_zones)
                raise ValueError(f"Invalid zones {invalid_zones}. "
                                 f"Allowed zones are {self.available_zones}.")
        self.zones = zones
        self.mask_zeros = mask_zeros
        self.root = config['data_dir'] + '/pv_us/' if root is None else root
        # set name
        name = "PvUS" if len(zones) == 2 else f"PvUS-{zones[0]}"
        # load dataset
        actual, mask, metadata = self.load(mask_zeros)
        super().__init__(target=actual, mask=mask, freq=freq,
                         similarity_score="distance",
                         spatial_aggregation="sum",
                         temporal_aggregation="mean",
                         name=name)
        self.add_covariate('metadata', metadata, pattern='n f')

    @property
    def raw_file_names(self):
        return [f'{zone}.h5' for zone in self.zones]

    @property
    def required_file_names(self):
        return self.raw_file_names

    def load_raw(self):
        actual, metadata = [], []
        for zone in self.zones:
            # load zone data
            zone_path = os.path.join(self.root_dir, f'{zone}.h5')
            actual.append(pd.read_hdf(zone_path, key='actual'))
            metadata.append(pd.read_hdf(zone_path, key='metadata'))
        # concat zone and sort by plant id
        actual = pd.concat(actual, axis=1).sort_index(axis=1, level=0)
        metadata = pd.concat(metadata, axis=0).sort_index()
        # drop duplicated farms when loading whole dataset
        if len(self.zones) == 2:
            duplicated_farms = metadata.index[[s_id.endswith('-east')
                                               for s_id in metadata.state_id]]
            metadata = metadata.drop(duplicated_farms, axis=0)
            actual = actual.drop(duplicated_farms, axis=1, level=0)
        return actual, metadata

    def load(self, mask_zeros):
        actual, metadata = self.load_raw()
        mask = (actual > 0) if mask_zeros else None
        return actual, mask, metadata

    def compute_similarity(self, method: str, theta: float = 150, **kwargs):
        if method == "distance":
            from tsl.ops.similarities import (gaussian_kernel,
                                              geographical_distance)
            # compute distances from latitude and longitude degrees
            loc_coord = self.metadata.loc[:, ['lat', 'lon']]
            dist = geographical_distance(loc_coord, to_rad=True).values
            return gaussian_kernel(dist, theta=theta)
