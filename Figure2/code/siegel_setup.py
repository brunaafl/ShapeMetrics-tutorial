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

def folds():
    """The 10 cross-validated folds, as `folds[fold][half][unit] -> array`.

    Three panel notebooks used to open this as a literal relative path:

        pickle.load(open("data/Siegel_10_folds_avg_stim_netrep.pkl", "rb"))

    which resolved only when the kernel happened to start in siegel_analyses/,
    and named a 77 MB file that is not in git -- so on a clone the notebooks
    failed on their first data cell whatever the working directory.

    The same arrays are now tracked as Data/derived/siegel2015/siegel_folds.npz
    (float64, nothing dropped; see extract/siegel_folds.py). This rebuilds the
    nested structure from its flat `fold|half|unit` keys, so callers see exactly
    what pickle.load gave them.
    """
    import numpy as np
    from shapemetrics import paths

    z = np.load(paths.derived("siegel2015", "siegel_folds.npz"))
    n_folds = max(int(k.split("|")[0]) for k in z.files) + 1
    n_halves = max(int(k.split("|")[1]) for k in z.files) + 1
    out = [[{} for _ in range(n_halves)] for _ in range(n_folds)]
    for k in z.files:
        i, h, unit = k.split("|", 2)
        out[int(i)][int(h)][unit] = z[k]
    return out
