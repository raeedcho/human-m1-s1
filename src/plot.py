from . import preproc,events,munge,models

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

def plot_step_signal(step_grasp_data,step_release_data,signal,axs=None,single_trial=True,add_legend=True):
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
        step_grasp_data,
        x='time from grasp1',
        y=signal,
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[0],
        legend=False,
        **extra_kwargs,
    )
    sns.despine(ax=axs[0],trim=True)
    sns.lineplot(
        step_release_data,
        x='time from release1',
        y=signal,
        hue='force level',
        hue_order=['low','medium','high'],
        ax=axs[1],
        legend=add_legend,
        **extra_kwargs,
    )
    sns.despine(ax=axs[1],trim=True)

    return axs

def run_step_pca(td_shifted,which_area,n_components=15):
    step_grasp_data,step_release_data = preproc.get_step_grasp_release_data(td_shifted)
    step_grasp_data = step_grasp_data.dropna()
    step_release_data = step_release_data.dropna()
    
    model = Pipeline([
        ('pca',PCA(n_components=n_components,whiten=False)),
    ])
    model.fit(pd.concat([
        step_grasp_data[which_area],
        step_release_data[which_area],
    ]))

    return (
        td_shifted
        .join(
            pd.DataFrame(
                model.transform(td_shifted[which_area].dropna()),
                index=td_shifted[which_area].dropna().index,
                columns=pd.MultiIndex.from_product([[f'{which_area} pca'],range(15)]),
            )
        )
    )

def run_step_rrr(td_shifted,rank=15):
    step_grasp_data,step_release_data = preproc.get_step_grasp_release_data(td_shifted)
    step_grasp_data = step_grasp_data.dropna()
    step_release_data = step_release_data.dropna()
    
    model = models.ReducedRankRegression(rank=rank)
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

    return (
        td_shifted
        .join(
            pd.concat(
                [model.transform(td_shifted['motor'])],
                axis=1,
                keys=['motor rrr'],
            )
        )
    )