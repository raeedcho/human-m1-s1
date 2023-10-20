from . import preproc,events,munge,models

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plot_step_signal(
    data,
    signal,
    channels,
    how_to_plot=None,
    height=1.5,
    aspect=2.5,
):
    '''
    Plot out (signal,channels) of step trials, split into grasp and release phases
    in a single FacetGrid seaborn figure.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing signals and channels
    signal : str or list of str
        Signal(s) to plot. If list, signals will be lined up by channel and plotted
        on the same plot with different line styles
    channels : index of which channels to plot
        (default is '', meaning the unnamed channel for single column signals)
    how_to_plot : str, optional
        How to plot the signal. Options are:
        - None: line plot of trial average, with calculated confidence intervals
        - 'average_only': line plot of trial average without confidence intervals
        - 'single_trial': line plot of single trials with out average or confidence intervals

    Returns
    -------
    FacetGrid object
    '''
    
    if 'phase' not in data.index.names:
        data = munge.get_step_grasp_release_data(data)

    if type(signal) is not list:
        signal=[signal]
    if type(channels) is not list:
        channels=[channels]

    if how_to_plot=='average_only':
        extra_kwargs = {
            'errorbar' : None,
        }
    elif how_to_plot=='single_trial':
        extra_kwargs = {
            'units' : 'set_trial',
            'estimator' : None,
            'errorbar' : None,
            'lw' : 0.5,
        }
    else:
        extra_kwargs = {}

    grasp_release_data = (
        data
        [signal]
        .stack(level='signal')
        .rename_axis(index={'signal': 'signal type'})
        [channels]
        .stack()
        .to_frame(signal[0])
    )

    g = sns.relplot(
        data=grasp_release_data,
        x='relative time',
        y=signal[0],
        hue='force level',
        hue_order=['low','medium','high'],
        style='signal type',
        style_order=signal,
        kind='line',
        col='phase',
        row='channel',
        aspect=aspect,
        height=height,
        facet_kws={
            'sharex': 'col',
            'sharey': True,
            'margin_titles': True,
        },
        **extra_kwargs,
    )
    g.refline(x=0)
    g.despine(trim=True)
    g.tight_layout()

    g.set_axis_labels('Relative time (s)',signal[0])
    g.set_titles(col_template='{col_name} phase',row_template='channel {row_name}')

    return g

def plot_trial_signal(
    data,
    signal,
    channels,
    hue='force level',
    hue_order=['low','medium','high'],
    how_to_plot=None,
    height=1.5,
    aspect=2.5,
):
    '''
    Plot out (signal,channels) of step trials, split into grasp and release phases
    in a single FacetGrid seaborn figure.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing signals and channels
    signal : str or list of str
        Signal(s) to plot. If list, signals will be lined up by channel and plotted
        on the same plot with different line styles
    channels : index of which channels to plot
        (default is '', meaning the unnamed channel for single column signals)
    hue: str, optional
        The name of the column in data to use for coloring trials
    hue_order: list, optional
        The order of the hue levels
    how_to_plot : str, optional
        How to plot the signal. Options are:
        - None: line plot of trial average, with calculated confidence intervals
        - 'average_only': line plot of trial average without confidence intervals
        - 'single_trial': line plot of single trials with out average or confidence intervals

    Returns
    -------
    FacetGrid object
    '''
    
    if type(signal) is not list:
        signal=[signal]
    if type(channels) is not list:
        channels=[channels]

    if 'phase' not in data.index.names:
        raise ValueError('Data must be indexed by phase')

    if how_to_plot=='average_only':
        extra_kwargs = {
            'errorbar' : None,
        }
    elif how_to_plot=='single_trial':
        extra_kwargs = {
            'units' : 'set_trial',
            'estimator' : None,
            'errorbar' : None,
            'lw' : 0.5,
        }
    else:
        extra_kwargs = {}

    grasp_release_data = (
        data
        [signal]
        .stack(level='signal')
        .rename_axis(index={'signal': 'signal type'})
        [channels]
        .stack()
        .to_frame(signal[0])
    )

    g = sns.relplot(
        data=grasp_release_data,
        x='relative time',
        y=signal[0],
        hue=hue,
        hue_order=hue_order,
        style='signal type',
        style_order=signal,
        kind='line',
        col='phase',
        row='channel',
        aspect=aspect,
        height=height,
        facet_kws={
            'sharex': 'col',
            'sharey': True,
            'margin_titles': True,
        },
        **extra_kwargs,
    )
    g.refline(x=0)
    g.despine(trim=True)
    g.tight_layout()

    g.set_axis_labels('Relative time (s)',signal[0])
    g.set_titles(col_template='{col_name} phase',row_template='channel {row_name}')

    return g