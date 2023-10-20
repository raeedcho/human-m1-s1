from . import events

import numpy as np
import pandas as pd

def get_step_grasp_release_data(data):
    '''
    Get data for step trials, arranged into grasp and release portions
    '''
    return get_epoch_data(
        (
            data
            .groupby('trial type',observed=True)
            .get_group('step')
        ),
        {
            'grasp1': slice('-1500 ms','3 sec'),
            'release1': slice('-3 sec','4 sec'),
        }
    )

def get_epoch_data(data,epochs):
    '''
    Get data arranged by epoch and relative time

    Parameters:
    -----------
    data : pandas.DataFrame
        The data to be arranged by epoch and relative time.
    epochs : dict
        The list of epoch names to be used for grouping the data.
        Keys correspond to state name and values are datetime slices
        around the event onset.

    Returns:
    --------
    pandas.DataFrame
        The data arranged by epoch and relative time.
    '''
    epoch_data_list = [
        (
            data
            .pipe(events.reindex_from_event,event)
            .loc[(slice(None),event_slice),:]
        )
        for event,event_slice in epochs.items()
    ]
    return (
        pd.concat(epoch_data_list)
        .reset_index(level='relative time')
        .assign(**{
            'relative time': lambda df: df['relative time'] / np.timedelta64(1,'s'),
        })
        .set_index('relative time',append=True)
        .swaplevel('relative time',1)
    )

def group_average(td,keys=[]):
    return (
        td
        .stack()
        .groupby(keys+['channel'],observed=True)
        .agg('mean')
        .unstack()
        .dropna(axis=1,how='all')
    )

def hierarchical_assign(df,assign_dict):
    '''
    Extends pandas.DataFrame.assign to work with hierarchical columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to assign to
    assign_dict : dict of pandas.DataFrame or callable
        dictionary of dataframes to assign to df
    '''
    return (
        df
        .join(
            pd.concat(
                [val(df) if callable(val) else val for val in assign_dict.values()],
                axis=1,
                keys=assign_dict.keys(),
                names=['signal','channel'],
            )
        )
    )