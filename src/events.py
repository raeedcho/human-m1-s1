import pandas as pd

def get_state_transition_times(state_list):
    timestep = (
        state_list
        .reset_index(level=2)
        ['trial_time']
        .diff()
        .mode()
        .values[0]
    )
    prev_state = (
        state_list
        .rename(lambda t: t+timestep,level=2)
        .reindex(state_list.index)
    )
    state_transition_times = (
        pd.concat(
            [prev_state,state_list],
            keys=['previous_state','new_state'],
            axis=1,
        )
        # .dropna()
        .loc[
            lambda df: df['previous_state']!=df['new_state']
        ]
        .reset_index(level=2)
        .set_index('new_state',append=True)
        ['trial_time']
    )
    
    return state_transition_times

def reindex_from_event(data,event):
    event_times = (
        data['state']
        .pipe(get_state_transition_times)
        .loc[(slice(None),slice(None),event)]
    )

    new_data = (
        data
        .reset_index(level=2)
        .assign(**{
            f'time from {event}':
                lambda df: df['trial_time']-event_times,
        })
        .set_index(f'time from {event}',append=True)
    )

    return new_data