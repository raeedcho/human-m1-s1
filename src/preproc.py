from . import events

import scipy
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def load_data(filename='CRS07_Grasp_03242023.mat'):
    mat = scipy.io.loadmat(
        f'../data/{filename}',
        simplify_cells=True,
    )
    
    bin_size = 20e-3
    cols_to_expose = ['Kin']
    cols_to_transpose = [
        'Motor',
        'Sensory',
        'set_idx',
        'target_trace',
        'kin_pos',
        'kin_vel',
        'kin_force',
        'kin_analog',
    ]
    cols_to_keep = [
        'set',
        'trial',
        'TrialType',
        'Force_levels',
        'trial_time',
        'state',
        'kin_commforce',
        'Motor',
        'Sensory',
    ]
    cols_to_explode = [
        'trial_time',
        'state',
        'kin_commforce',
        'Motor',
        'Sensory',
    ]
    signal_types = {
        'trial type': 'category',
        'force level': 'category',
        'state': 'category',
        'force': 'float32',
    }

    TS = pd.DataFrame(mat['TS'])
    return (
        TS
        .assign(**pd.concat(
            [expose_single_col(TS[col]) for col in cols_to_expose],
            axis=1)
        )
        .drop(columns=cols_to_expose)
        .assign(**{
            col: lambda df,which_col=col: df[which_col].map(np.transpose)
            for col in cols_to_transpose
        })
        .assign(
            state_labels = lambda df: df['state_labels'].map(condition_state_labels),
        )
        .apply(name_states,axis=1)
        .assign(trial_time = get_trial_times)
        [cols_to_keep]
        .explode(cols_to_explode)
        .assign(**{
            'set_trial': lambda df: 100*df['set']+df['trial'],
            'trial type': lambda df: df['TrialType'],
            'force level': lambda df: df['Force_levels'].map(condition_force_levels),
            'force': lambda df: df['kin_commforce'],
            'motor': lambda df: df['Motor'].map(lambda arr: arr/bin_size),
            'sensory': lambda df: df['Sensory'].map(lambda arr: arr/bin_size),
        })
        .drop(columns=['set','trial','TrialType','Force_levels','kin_commforce','Motor','Sensory'])
        .astype(signal_types)
        .set_index(['set_trial','trial_time','trial type','force level','state'])
        .filter(items=['force','motor','sensory'])
        .pipe(crystallize_td)
    )

def expose_single_col(s: pd.Series):
    return (
        pd.DataFrame(list(s),s.index)
        .rename(columns=lambda subcol: f'{s.name}_{subcol}'.lower())
    )

def condition_state_labels(label_list):
    def state_map(label):
        if (
            type(label) is not str
            or label=='FSafe1'
            or label=='FailSafe1'
        ):
            return 'pretrial'
        elif label=='InterTrial':
            return 'posttrial'
        elif label=='Presentation':
            return 'grasp_prep'
        elif label=='Presentation2':
            return 'release_prep'
        elif label=='Grasp':
            return 'grasp1'
        elif label=='Release':
            return 'release1'
        else:
            return label.lower()
        
    return np.array(list(map(state_map,label_list)))

def condition_force_levels(level):
    if type(level) is np.ndarray:
        return 'double_ramp'
    elif level==4.7:
        return 'low'
    elif level==9.4:
        return 'medium'
    elif level==14:
        return 'high'
    else:
        return 'unknown'

def name_states(row):
    row_copy = row.copy()
    row_copy['state'] = row['state_labels'][row['state']-1]
    return row_copy

def get_trial_times(temp_df,bin_size='20ms'):
    return temp_df['Motor'].map(
        lambda x: pd.timedelta_range(
            start=0,
            periods=x.shape[0],
            freq=bin_size,
        )
    )

def crystallize_td(
        td,
        single_cols= [
            'force',
        ],
        array_cols=[
            'motor',
            'sensory',
        ],
    ):
    '''
    Expose dataframe columns with numpy arrays as hierarchical columns
    '''
    return (
        pd.concat(
            [td[col].rename(0) for col in single_cols] +
            [
                pd.DataFrame.from_records(
                    td[array_name].values,
                    td[array_name].index,
                ) for array_name in array_cols
            ],
            axis=1,
            keys=single_cols+array_cols,
        )
        #.assign(**{(col,0): td[col] for col in single_cols})
        [single_cols+array_cols]
        .rename_axis(columns=['signal','channel'])
    )

def norm_and_shift_rates(td,arrays=['motor','sensory'],norm_method='softnorm'):
    if norm_method.lower()=='zscore':
        method = zscore_array
    elif norm_method.lower()=='softnorm':
        method = softnorm_array
    else:
        Warning('Skipping array norm...')
        method = lambda x: x

    td_scored = td.assign(**{
        array: lambda df,arr=array: method(df[arr])
        for array in arrays
    })

    td_baseline = (
        td_scored
        .groupby('state',observed=True)
        .get_group('pretrial')
        [arrays]
        .groupby(['set_trial'])
        .agg(lambda s: np.nanmean(s,axis=0))
    )
    td_shifted = td_scored.assign(**{
        array: lambda df,arr=array: df[arr]-td_baseline[arr]
        for array in arrays
    })

    return td_shifted

def zscore_array(arr):
    return StandardScaler().fit_transform(arr)

def softnorm_array(arr,norm_const=5):
    def get_range(arr,axis=None):
        return np.nanmax(arr,axis=axis)-np.nanmin(arr,axis=axis)
    activity_range = get_range(arr,axis=0)
    return arr/(activity_range+norm_const)
