# %% load data
import src
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

td = src.preproc.load_data()

# %% look at PSTHs
src.plot.plot_step_psth(td,('motor',91))

# %% look at average step trial PCs
# z-scoring and softnorming seem to give very different results on PC space
td_shifted = src.preproc.norm_and_shift_rates(td,norm_method='zscore')

num_comps=5
fig,axs = plt.subplots(num_comps,2,sharex='col',sharey='row')
for compnum in range(num_comps):
    src.plot.plot_pcs(
        td_shifted,
        'motor',
        compnum,
        axs=axs[compnum,:],
        add_legend=False,
    )

# %%
step_ramp_data = (
    pd.concat([
        td_shifted.groupby('trial type').get_group(g)
        for g in ['step']
    ])
)
event_times = src.events.get_state_transition_times(step_ramp_data['state'])
grasp_times = event_times.loc[(slice(None),slice(None),'grasp1')]
release_times = event_times.loc[(slice(None),slice(None),'release1')]
hold_lens = release_times-grasp_times
sns.histplot(hold_lens.dt.total_seconds())
# %%
