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

def get_step_grasp_release_data(td_shifted):
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

