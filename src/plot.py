from . import preproc,events,munge

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def plot_step_psth(data,channel_index,single_trial=True):
    step_data = (
        data
        .groupby('trial type')
        .get_group('step')
    )
    step_grasp_data = (
        step_data
        .pipe(events.reindex_from_event,'grasp1')
        .loc[(slice(None),slice(None),slice('-1500 ms','3000 ms'))]
        .reset_index()
        .assign(set_trial=lambda df: 100*df['set']+df['trial'])
    )
    step_release_data = (
        step_data
        .pipe(events.reindex_from_event,'release1')
        .loc[(slice(None),slice(None),slice('-3 sec','4 sec'))]
        .reset_index()
        .assign(set_trial=lambda df: 100*df['set']+df['trial'])
    )

    if single_trial:
        extra_kwargs = {
            'units' : 'set_trial',
            'estimator' : None,
            'errorbar' : None,
            'lw' : 0.5,
        }
    else:
        extra_kwargs = {}

    fig,axs = plt.subplots(1,2,figsize=(6,4),sharey=True)
    sns.lineplot(
        step_grasp_data,
        x='time from grasp1',
        y=channel_index,
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[0],
        legend=False,
        **extra_kwargs,
    )
    sns.lineplot(
        step_release_data,
        x='time from release1',
        y=channel_index,
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[1],
        **extra_kwargs,
    )
    sns.despine(fig=fig,trim=True)

    return fig,axs

def plot_pcs(td_shifted,which_area,which_comp,axs=None,add_legend=True):
    step_grasp_data,step_release_data = preproc.get_step_grasp_release_data(td_shifted)
    
    model = Pipeline([
        # ('scale',StandardScaler()),
        ('pca',PCA(n_components=15,whiten=False)),
    ])
    model.fit(np.row_stack([
        np.row_stack(step_grasp_data[which_area]),
        np.row_stack(step_release_data[which_area]),
    ]))

    step_grasp_data['pca'] = [model.transform(arr[None,:]).squeeze() for arr in step_grasp_data[which_area]]
    step_release_data['pca'] = [model.transform(arr[None,:]).squeeze() for arr in step_release_data[which_area]]

    if axs is None:
        fig,axs = plt.subplots(1,2,figsize=(6,4),sharey=True)

    sns.lineplot(
        (
            step_grasp_data
            .pipe(munge.add_unit_activity,'pca',which_comp)
            .reset_index()
        ),
        x='time from grasp1',
        y=f'pca ch{which_comp} activity',
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[0],
        legend=False,
    )
    sns.despine(ax=axs[0],trim=True)
    sns.lineplot(
        (
            step_release_data
            .pipe(munge.add_unit_activity,'pca',which_comp)
            .reset_index()
        ),
        x='time from release1',
        y=f'pca ch{which_comp} activity',
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[1],
        legend=add_legend,
    )
    sns.despine(ax=axs[1],trim=True)