from typing import Callable

import numpy as np
import pandas as pd

from tsl.typing import Index, FrameArray


def framearray_to_numpy(x: FrameArray) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        if x.columns.nlevels == 1:
            return x.to_numpy()
        cols = [x.columns.unique(i) for i in range(x.columns.nlevels)]
        cols = pd.MultiIndex.from_product(cols)
        x = x.reindex(columns=cols)
        return x.values.reshape((-1, *cols.levshape))
    return np.asarray(x)


def framearray_shape(x: FrameArray) -> tuple:
    if not isinstance(x, pd.DataFrame):
        return np.asarray(x).shape
    elif x.columns.nlevels > 1:
        return (len(x),) + x.columns.levshape
    return x.shape


def aggregate(x: FrameArray, index: Index, aggr_fn: Callable = np.sum,
              axis: int = 1, level: int = 0) -> FrameArray:
    """Aggregate rows/columns in (MultiIndexed) DataFrame according to a new
    index.

    Args:
        x (pd.DataFrame): :class:`~pandas.DataFrame` to be aggregated.
        index (Index): A sequence of :obj:`cluster_id` with length equal to
            the index over which aggregation is performed. The :obj:`i`-th
            element of index at :obj:`axis` and :obj:`level` will be mapped to
            :obj:`index[i]`-th position in new index.
        aggr_fn (Callable): Function to be used for aggregation.
        axis (int): Axis over which performing aggregation, :obj:`0` for index,
            :obj:`1` for columns.
            (default :obj:`1`)
        level (int): Level over which performing aggregation if :obj:`axis` is
            a :class:`~pandas.MultiIndex`.
            (default :obj:`0`)
    """
    to_numpy = False
    if not isinstance(x, pd.DataFrame):
        cols = pd.MultiIndex.from_product([np.arange(s) for s in x.shape[1:]])
        x = x.reshape(x.shape[0], -1)
        x = pd.DataFrame(x, columns=cols)
        if axis > 1:
            axis -= 1
            level += 1
        to_numpy = True
    if axis == 0:
        x = x.groupby(index, axis=0).aggregate(aggr_fn)
    elif axis == 1:
        cols = [x.columns.unique(i).values for i in
                range(x.columns.nlevels)]
        cols[level] = index
        grouper = pd.MultiIndex.from_product(cols, names=x.columns.names)
        x = x.groupby(grouper, axis=1).aggregate(aggr_fn)
        x.columns = pd.MultiIndex.from_tuples(x.columns,
                                              names=grouper.names)
    if to_numpy:
        x = framearray_to_numpy(x)
    return x


def reduce(x: FrameArray, index: Index,
           axis: int = 0, level: int = 0) -> FrameArray:
    if index is None:
        return x
    elif not isinstance(index, (pd.Index, slice)):
        index: np.ndarray = np.asarray(index)

    if isinstance(x, pd.DataFrame):
        if axis == 0:
            return x.loc[index]

        n_levels = x.columns.nlevels
        if n_levels > 1:
            if index.dtype == np.bool:
                index = x.columns.unique(level)[index]
            index = tuple([index if i == level else slice(None)
                           for i in range(n_levels)])
        return x.loc[:, index]
    else:
        axis = axis + level
        index = tuple([index if i == axis else slice(None)
                       for i in range(x.ndim)])
        return x[index]


def temporal_mean(x: FrameArray, index: pd.DatetimeIndex = None) \
        -> FrameArray:
    """Compute the mean values for each row.

    The mean is first computed hourly over the week of the year. Further
    :obj:`NaN` values are imputed using hourly mean over the same month through
    the years. If other :obj:`NaN` are present, they are replaced with the mean
    of the sole hours. Remaining missing values are filled with :obj:`ffill` and
    :obj:`bfill`.

    Args:
        x (np.array | pd.Dataframe): Array-like with missing values.
        index (pd.DatetimeIndex | pd.PeriodIndex | pd.TimedeltaIndex, optional):
            Temporal index if x is not a :obj:'~pandas.Dataframe' with a
            temporal index. Must have same length as :obj:`x`.
            (default :obj:`None`)
    """
    if index is not None:
        if not isinstance(index, pd.DatetimeIndex):
            # try casting
            index = pd.to_datetime(index)
        assert len(index) == len(x)
        if isinstance(x, pd.DataFrame):
            # override index of x
            df_mean = x.copy().set_index(index)
        else:
            # try casting to np.ndarray
            x = np.asarray(x)
            shape = x.shape
            # x can be N-dimensional, we flatten all but the first dimensions
            x = x.reshape((shape[0], -1))
            df_mean = pd.DataFrame(x, index=index)
    elif isinstance(x, pd.DataFrame):
        df_mean = x.copy()
    else:
        raise TypeError("`x` must be a pd.Dataframe or a np.ndarray.")
    cond0 = [df_mean.index.year, df_mean.index.isocalendar().week,
             df_mean.index.hour]
    cond1 = [df_mean.index.year, df_mean.index.month, df_mean.index.hour]
    conditions = [cond0, cond1, cond1[1:], cond1[2:]]
    while df_mean.isna().values.sum() and len(conditions):
        nan_mean = df_mean.groupby(conditions[0]).transform(np.nanmean)
        df_mean = df_mean.fillna(nan_mean)
        conditions = conditions[1:]
    if df_mean.isna().values.sum():
        df_mean = df_mean.fillna(method='ffill')
        df_mean = df_mean.fillna(method='bfill')
    if isinstance(x, np.ndarray):
        df_mean = df_mean.values.reshape(shape)
    return df_mean
