from . import events

import numpy as np
import pandas as pd

def extract_unit_activity(data,which_area,which_unit):
    return data[which_area].apply(lambda arr: arr[which_unit])

def add_unit_activity(data,which_area,which_unit):
    return (
        data
        .assign(**{
            f'{which_area} ch{which_unit} activity': lambda df:
                extract_unit_activity(df,which_area,which_unit)
        })
    )

def get_step_grasp_release_data(data):
    '''
    Get data for step trials, arranged into grasp and release portions
    '''
    step_data = (
        data
        .groupby('trial type',observed=True)
        .get_group('step')
    )
    step_grasp_data = (
        step_data
        .pipe(events.reindex_from_event,'grasp1')
        .loc[(slice(None),slice(None),slice('-1500 ms','3000 ms'))]
        .reset_index(level=2)
        .assign(**{
            'relative time': lambda df: df['time from grasp1'] / np.timedelta64(1,'s'),
            'phase': 'grasp',
        })
        .drop(columns=[('time from grasp1','')])
        .set_index(['phase','relative time'],append=True)
    )
    step_release_data = (
        step_data
        .pipe(events.reindex_from_event,'release1')
        .loc[(slice(None),slice(None),slice('-3 sec','4 sec'))]
        .reset_index(level=2)
        .assign(**{
            'relative time': lambda df: df['time from release1'] / np.timedelta64(1,'s'),
            'phase': 'release',
        })
        .drop(columns=[('time from release1','')])
        .set_index(['phase','relative time'],append=True)
    )

    return pd.concat([step_grasp_data,step_release_data])

def mean_step_grasp_release_data(td_shifted):
    step_data = (
        td_shifted
        .groupby('trial type')
        .get_group('step')
    )
    step_grasp_data = (
        step_data
        .pipe(events.reindex_from_event,'grasp1')
        .loc[(slice(None),slice(None),slice('-1500 ms','3000 ms'))]
        .groupby(['force level','time from grasp1'],observed=True)
        .agg({
            'force': np.nanmean,
            'motor': lambda s: np.nanmean(np.row_stack(s),axis=0),
            'sensory': lambda s: np.nanmean(np.row_stack(s),axis=0),
        })
    )
    step_release_data = (
        step_data
        .pipe(events.reindex_from_event,'release1')
        .loc[(slice(None),slice(None),slice('-3 sec','4 sec'))]
        .groupby(['force level','time from release1'],observed=True)
        .agg({
            'force': np.nanmean,
            'motor': lambda s: np.nanmean(np.row_stack(s),axis=0),
            'sensory': lambda s: np.nanmean(np.row_stack(s),axis=0),
        })
    )

    return step_grasp_data,step_release_data

