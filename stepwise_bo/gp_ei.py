import os
import numpy as np
from scipy import stats

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from stepwise_bo.record_bo import read_selection_record_file
from stepwise_bo.eggs import distribute_eggs

from utils.utils import find_differences_numpy




def select_next_training_data(selection_record_path, current_berri_selected_rep, berri_datasum_rep, selected_sum, pruned_berri_idxs=[], used_berri_rep=[]):
    """Bayesian optimization with Gaussian process and expected improvement.
    Inputs
    ----------
    X : array-like, shape = (n_samples, n_features)
        Training data.

    y : array-like, shape = (n_samples,)
        Target values.

    X_test : array-like, shape = (n_samples, n_features)
        Test data.

    kernel : sklearn.gaussian_process.kernels.Kernel
        Kernel.

    n_restarts_optimizer : int, optional (default=10)
        Number of restarts of the optimizer for finding kernel's parameters.

    random_state : int, optional (default=0)
        Random state.

    Outputs
    -------
    X_test : array-like, shape = (n_features,)
        The text data with maximum utility.
    """
    
    
    X, y = read_selection_record_file(selection_record_path, len(current_berri_selected_rep))
        
    num_baskets = len(current_berri_selected_rep)
    num_eggs = selected_sum
    egg_rep = np.array(distribute_eggs(num_eggs, num_baskets))+np.array(current_berri_selected_rep)
    selected_egg_distribution = egg_rep
    if len(used_berri_rep) > 0:
        selected_egg_distribution = np.array(distribute_eggs(num_eggs, num_baskets))+np.array(used_berri_rep)

    if_distributions_fair = berri_datasum_rep-selected_egg_distribution
    X_test = egg_rep[np.all(if_distributions_fair>=0, axis=1)]

    X_test = np.array([row for row in X_test if not any(np.array_equal(row, x) for x in X[num_baskets:])])
    if int(os.getenv('LOCAL_RANK', -1)) in [-1,0]:
        print("2-3")
        print(X_test)
    
    if len(pruned_berri_idxs) > 0:
        X_test = np.array([row for row in X_test if not any(find_differences_numpy(row, current_berri_selected_rep)[0]==x for x in pruned_berri_idxs)])
    if int(os.getenv('LOCAL_RANK', -1)) in [-1,0]:
        print("2-4")
        print(X_test)

    if X_test.size == 0:
        return None
    
    rbf_kernel_func = RBF(length_scale=0.5) 
    gp = GaussianProcessRegressor(kernel=rbf_kernel_func, n_restarts_optimizer=10, random_state=0) 
    gp.fit(X, y)
    y_test_pred, sigma = gp.predict(X_test, return_std=True)
        
    def cal_EI(mu, std, best_y, xi=0.001): 
        mu = mu.flatten() 
        Z_ = mu-best_y-xi
        EI = np.where(std>0, Z_*(1-stats.norm.cdf(Z_/std)) + std*stats.norm.pdf(Z_/std) , 0)
        return EI
    utility = cal_EI(y_test_pred, sigma, max(y))
    next_data_selection_rep = X_test[np.argmax(utility)]
    
    if int(os.getenv('LOCAL_RANK', -1)) in [0,-1]:
        print("the max utility:", np.max(utility))
        print("the max utilite number:", np.sum(utility==np.max(utility)))
    
    return next_data_selection_rep