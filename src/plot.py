from . import preproc,events,munge,models

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

def plot_step_psth(data,channel_index,single_trial=True):
    step_grasp_data,step_release_data = preproc.get_step_grasp_release_data(data)

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

def plot_pcs(td_shifted,which_area,which_comp,axs=None,add_legend=True,single_trial=True):
    step_grasp_data,step_release_data = preproc.get_step_grasp_release_data(td_shifted)
    step_grasp_data = step_grasp_data.dropna()
    step_release_data = step_release_data.dropna()
    
    model = Pipeline([
        ('pca',PCA(n_components=15,whiten=False)),
    ])
    model.fit(pd.concat([
        step_grasp_data[which_area],
        step_release_data[which_area],
    ]))

    step_grasp_data = (
        step_grasp_data
        .join(
            pd.DataFrame(
                model.transform(step_grasp_data[which_area]),
                index=step_grasp_data.index,
                columns=pd.MultiIndex.from_product([['pca'],range(15)]),
            )
        )
    )
    step_release_data = (
        step_release_data
        .join(
            pd.DataFrame(
                model.transform(step_release_data[which_area]),
                index=step_release_data.index,
                columns=pd.MultiIndex.from_product([['pca'],range(15)]),
            )
        )
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

    if axs is None:
        fig,axs = plt.subplots(1,2,figsize=(6,4),sharey=True)

    sns.lineplot(
        step_grasp_data.reset_index(),
        x='time from grasp1',
        y=('pca',which_comp),
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[0],
        legend=False,
        **extra_kwargs,
    )
    sns.despine(ax=axs[0],trim=True)
    sns.lineplot(
        step_release_data.reset_index(),
        x='time from release1',
        y=('pca',which_comp),
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[1],
        legend=False,
        **extra_kwargs,
    )
    sns.despine(ax=axs[1],trim=True)

def plot_rrr_comps(td_shifted,which_comp,axs=None,add_legend=True,single_trial=True):
    step_grasp_data,step_release_data = preproc.get_step_grasp_release_data(td_shifted)
    step_grasp_data = step_grasp_data.dropna()
    step_release_data = step_release_data.dropna()
    
    model = models.ReducedRankRegression(rank=5)
    model.fit(
        pd.concat([
            step_grasp_data['motor'],
            step_release_data['motor'],
        ]),
        pd.concat([
            step_grasp_data['sensory'],
            step_release_data['sensory'],
        ]),
    )
    step_grasp_data = (
        step_grasp_data
        .join(
            pd.concat(
                [model.transform(step_grasp_data['motor'])],
                axis=1,
                keys=['rrr'],
            )
        )
    )
    step_release_data = (
        step_release_data
        .join(
            pd.concat(
                [model.transform(step_release_data['motor'])],
                axis=1,
                keys=['rrr'],
            )
        )
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

    if axs is None:
        fig,axs = plt.subplots(1,2,figsize=(6,4),sharey=True)

    sns.lineplot(
        step_grasp_data.reset_index(),
        x='time from grasp1',
        y=('rrr',which_comp),
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[0],
        legend=False,
        **extra_kwargs,
    )
    sns.despine(ax=axs[0],trim=True)
    sns.lineplot(
        step_release_data.reset_index(),
        x='time from release1',
        y=('rrr',which_comp),
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[1],
        legend=False,
        **extra_kwargs,
    )
    sns.despine(ax=axs[1],trim=True)