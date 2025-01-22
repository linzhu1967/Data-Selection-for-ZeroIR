import pandas as pd
import numpy as np

from utils.global_invariants import TEST_METRICS_SMALL_PCTGS
from utils.utils import find_differences_numpy


def get_initial_pruning_idxs(selection_record_path, search_strat):
    initial_pruned_idxs = []
    
    df = pd.read_csv(selection_record_path, sep='\t')
    for metric in TEST_METRICS_SMALL_PCTGS: # ["ndcg@10", "ndcg@10.20%", "ndcg_cut_10.10%", "ndcg_cut_10.20%"]
        if metric in df.columns:
            initial_observations = df[metric].to_numpy()
            break
    
    if search_strat in ["multi-metrics", "single-metric", "single-metric-delete-used-source"]:
        initial_pruned_idxs = []
    elif search_strat == "multi-metrics-init-pruning":
        initial_pruned_idxs = np.argsort(initial_observations)[:len(initial_observations)//2].tolist()
    elif search_strat == "single-metric-init-pruning":
        initial_pruned_idxs = np.argsort(initial_observations)[:len(initial_observations)//4].tolist()
    else:
        raise ValueError(f"Unsupported search strategy: {search_strat}")
    
    return initial_pruned_idxs


def get_berri_idxs_to_prune_quarter(father_iter_flag, father_rep_np, existed_iter_info_dict, existed_pruned_berri_idxs):
    prune_idxs_list = existed_pruned_berri_idxs
    
    left_idxs = len(father_rep_np) - len(prune_idxs_list)
    if left_idxs > len(father_rep_np)//4:
        # (father_iter_flag, iter_flag, iter_metric, new_m1_value, new_rep, new_ckpt)
        iter_list = [(item[3], item[-2]) for item in list(existed_iter_info_dict.values()) if item[0]==father_iter_flag]
        iter_list.sort(key=lambda x: x[0])
        prune_rep_list = [x[1] for x in iter_list][:len(iter_list) // 4]
            
        for rep_np in prune_rep_list:
            prune_idx = find_differences_numpy(rep_np, father_rep_np)[0]
            prune_idxs_list.append(int(prune_idx))
                        
    return prune_idxs_list