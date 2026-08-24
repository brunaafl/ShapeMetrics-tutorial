import matplotlib.pyplot as plt
import numpy as np
from netrep.metrics import LinearMetric
import scipy.stats as sts

from sklearn.decomposition import PCA
from statsmodels.multivariate.cancorr import CanCorr
from sklearn.preprocessing import StandardScaler
z_scaler = StandardScaler()

# todo sqrt spike counts

def prep_area(averaged_train,averaged_test,conds,idx,MM=20,M=3):

    # average activity during stimulus
    X_train =  np.mean(averaged_train,-1)
    X_test =  np.mean(averaged_test,-1)

    # reduce to MM dimensions
    X_train, X_test = get_low_d(X_train,X_test,m=MM,zscore=True)

    # pick the M dimensions with most variance for this conditon
    X_train = X_train[:,idx[:M]]
    X_test = X_test[:,idx[:M]]

    # compute means and covariances for distance metfic
    cov_train,cov_test = condition_cov(X_train,X_test,conds)
    mean_train = marginalize_cond(X_train,conds)
    mean_test = marginalize_cond(X_test,conds)

    # put in stocastic format
    X_train = (mean_train,cov_train)
    X_test = (mean_test,cov_test)

    return X_train,X_test

def get_low_d(X_train,X_test,m=50,cv=False,zscore=False):

    if zscore:
        # X_train = z_scaler.fit_transform(X_train)
        # X_test = z_scaler.fit_transform(X_test)

        X_train -= np.mean(X_train,0)
        X_test -= np.mean(X_test,0)

        X_train_std =  np.std(X_train,0)
        X_test_std = np.std(X_test,0)

        idx = (X_train_std > 0) & (X_test_std > 0)

        X_train = X_train[:,idx]/X_train_std[idx]
        X_test = X_test[:,idx]/X_test_std[idx]

    else:
        X_train -= np.mean(X_train,0)
        X_test -= np.mean(X_test,0)

    # fit PCA only on train set
    pca = PCA(n_components=m,svd_solver="full").fit(X_train)
    proj_train = pca.components_[:m]

    if cv: 
        proj_test = proj_train
    else:
        pca = PCA(n_components=m,svd_solver="full").fit(X_test)
        proj_test = pca.components_[:m]

    X1_low_train, X1_low_test = X_train @ proj_train.T, X_test @ proj_test.T

    return  X1_low_train, X1_low_test

def cv_CCA(X1_train,X1_test,X2_train,X2_test,m=50,pca=True):


    if pca:
        X1_low_train, X1_low_test = get_low_d(X1_train,X1_test,m=m)
        X2_low_train, X2_low_test = get_low_d(X2_train,X2_test,m=m)
    else:
        X1_low_train, X1_low_test = X1_train, X1_test 
        X2_low_train, X2_low_test = X2_train, X2_test

    # Find canonical dimensions on train data
    cc = CanCorr(X1_low_train, X2_low_train)
    cdims1 = cc.y_cancoef
    cdims2 = cc.x_cancoef

    # (test) projections on the canonical dimensions
    X1_proj = X1_low_test @ cdims1
    X2_proj = X2_low_test @ cdims2

    # compute canonical correlations
    ccs =  [sts.pearsonr(X1_proj[:,c],X2_proj[:,c])[0] for c in range(m)]
    return ccs

def compute_metrics(A_train,A_test,B_train,B_test,m):
    metric = LinearMetric(alpha=0,score_method="angular")
    metric.fit(A_train,B_train)
    dist_CCA = metric.score(A_test,B_test)

    metric = LinearMetric(alpha=1,score_method="angular")
    metric.fit(A_train,B_train)
    dist_reg = metric.score(A_test,B_test)
    
    #dist_cvCCA = cv_CCA(A_train,A_test,B_train,B_test,m=m,pca=False)[0]

    return dist_CCA,dist_reg,0 #dist_cvCCA

def cross_temporal_rep(averaged_train,averaged_test,key_timepoints,m=10):

    all_cross_rep =  np.zeros([3,len(key_timepoints),len(key_timepoints)])

    for t1 in range(len(key_timepoints)):
    
        t1_train=np.mean(averaged_train[:,:,key_timepoints[t1]],-1)
        t1_test=np.mean(averaged_test[:,:,key_timepoints[t1]],-1)
        t1_low_D_train, t1_low_D_test = get_low_d(t1_train,t1_test,m=M,cv=True)

        for t2 in range(t1,len(key_timepoints)):

            t2_train=np.mean(averaged_train[:,:,key_timepoints[t2]],-1)
            t2_test=np.mean(averaged_test[:,:,key_timepoints[t2]],-1)
            t2_low_D_train, t2_low_D_test = get_low_d(t2_train,t2_test,m=M,cv=True)

            dist_CCA,dist_reg,dist_cvCCA = compute_metrics(
                                            t1_low_D_train,t1_low_D_test,
                                            t2_low_D_train,t2_low_D_test,m=M)

            all_cross_rep[0,t1,t2] = all_cross_rep[0,t2,t1] = dist_CCA
            all_cross_rep[1,t1,t2] = all_cross_rep[1,t2,t1] = dist_reg
            all_cross_rep[2,t1,t2] = all_cross_rep[2,t2,t1] = dist_cvCCA

    return all_cross_rep

def marginalize_cond(averaged_data,cond):
    marginalized = []
    for cond_i in cond.unique():
        avg_cond = np.mean(averaged_data[cond==cond_i],0)
        marginalized.append(avg_cond)

    marginalized = np.array(marginalized)

    return marginalized

def condition_cov(X_train,X_test,cond):

    conds = cond.unique()
    covs_train = np.stack([np.cov(X_train[cond==c].T) for c in conds], 0)
    covs_test = np.stack([np.cov(X_test[cond==c].T) for c in conds], 0)

    return covs_train,covs_test

def get_exp_var(proj,all_conds):
    conds = np.unique(all_conds)
    all_fs = []

    for dim in range(proj.shape[1]):
        if len(conds) < 3:
            f = sts.f_oneway(proj[all_conds==conds[0],dim],proj[all_conds==conds[1],dim])[0]
        else:
            f = sts.f_oneway(proj[all_conds==conds[0],dim],proj[all_conds==conds[1],dim],
                        proj[all_conds==conds[2],dim],proj[all_conds==conds[3],dim])[0]
        
        all_fs.append(f)

    return np.array(all_fs)

def combine_monkeys(monkey_A, monkey_B,A="rex",B="paula"):
    monkey_both = {}
    for area in monkey_A.keys():
        monkey_both[area+"_"+A] = monkey_A[area]
        monkey_both[area+"_"+B] = monkey_B[area]
        # monkey_both[area] = np.concatenate([monkey_A[area],monkey_B[area]],axis=1)

    # monkey_both["all_"+A] = np.concatenate(list(monkey_A.values()),axis=1)
    # monkey_both["all_"+B] = np.concatenate(list(monkey_B.values()),axis=1)
    # monkey_both["all"] = np.concatenate([monkey_both["all_"+A],monkey_both["all_"+B]],axis=1)

    return monkey_both

def plot_cov(means,covs,colors):
    for ci,cov in enumerate(covs):
        s,V = np.linalg.eig(cov)
        theta = np.linspace(0, 2*np.pi, 1000);
        ellipsis = (2*np.sqrt(s[None,:]) * V) @ [np.sin(theta), np.cos(theta)]
        plt.plot(ellipsis[0,:]+means[ci,0], ellipsis[1,:]+means[ci,1],color=colors[ci])