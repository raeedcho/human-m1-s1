import pandas as pd

def get_state_transition_times(state_list):
    timestep = (
        state_list
        .reset_index(level='trial_time')
        ['trial_time']
        .diff()
        .mode()
        .values[0]
    )
    prev_state = (
        state_list
        .rename(lambda t: t+timestep,level='trial_time')
        .reindex(state_list.index)
    )
    state_transition_times = (
        pd.concat(
            [prev_state,state_list],
            keys=['previous_state','new_state'],
            axis=1,
        )
        .loc[
            lambda df: df['previous_state']!=df['new_state']
        ]
        .reset_index(level='trial_time')
        .set_index('new_state',append=True)
        ['trial_time']
    )
    
    return state_transition_times

def reindex_from_event(data,event):
    event_times = (
        data
        .reset_index(level='state')
        ['state']
        .pipe(get_state_transition_times)
        .xs(event,level='new_state')
    )

    new_data = (
        data
        .reset_index(level='trial_time')
        .assign(**{
            'relative time': lambda df: df['trial_time']-event_times,
            'phase': event,
        })
        .drop(columns='trial_time',level='signal')
        .set_index(['phase','relative time'],append=True)
        .swaplevel('relative time',1)
    )

    return new_data