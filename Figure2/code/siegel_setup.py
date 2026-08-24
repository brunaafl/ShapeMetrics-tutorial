import numpy as np

areas = ['PFC','FEF','LIP','Parietal','IT','MT','V4']
wells = ['frontal', 'parietal', 'temporal']

time_bins = (-2.5, 3.5)
bin_size = 0.025
time = np.arange(-2.5,3.5,bin_size)

cue_idx = (time[:-1]>-1) & (time[:-1]<0)
stim_idx = (time[:-1]>0) & (time[:-1]<0.20)
response_idx = (time[:-1]>0.2) & (time[:-1]<0.40)
fixation_idx = (time[:-1]<-1)