import os
import pandas as pd

from utils.global_invariants import TEST_METRICS_SMALL_PCTGS



def init_selection_record_file(selection_record_path, training_berri_datasets, eval_metrics, eval_pctg_list, beir_list=[]):
    if not os.path.exists(selection_record_path):
        print("Init the selection_record_path:[{}]".format(selection_record_path))
        with open(selection_record_path, "w") as rew:
            berri_datasets_string = "\t".join(training_berri_datasets.keys())
            
            if len(beir_list) > 0:
                # random, polling
                eval_metrics_pctg_string = "\t".join([str(beir_list[i])+"."+eval_metrics[0]+"."+str(eval_pctg_list[i]) for i in range(len(beir_list))])
                rew.write("iteration\t{}\t{}\t{}\n".format(berri_datasets_string, eval_metrics_pctg_string, "update_flag"))
            else: 
                eval_metrics_pctg_string = "\t".join([str(x) + "." + str(y) for x in eval_metrics for y in eval_pctg_list])
                if len(eval_metrics) <= 1:
                    # backtrack, none
                    rew.write("iteration\t{}\t{}\n".format(berri_datasets_string, eval_metrics_pctg_string))
                else:
                    # single metric, multi metrics
                    rew.write("iteration\t{}\t{}\t{}\t{}\n".format(berri_datasets_string, eval_metrics_pctg_string, "father_iteration", "update_flag"))


def get_lines_num(selection_record_path):
    with open(selection_record_path, "r") as f:
        lines = f.readlines()
        return len(lines) - 1 # remove the header line


def read_selection_record_file(selection_record_path, num_berri_headers):
    df = pd.read_csv(selection_record_path, sep='\t')
    berri_headers = [col for col in df.columns if "_" in col][:num_berri_headers]
    X = df[berri_headers].to_numpy()
    for metric in TEST_METRICS_SMALL_PCTGS: # ["ndcg@10", "ndcg@10.20%", "ndcg_cut_10.10%", "ndcg_cut_10.20%"]
        if metric in df.columns:
            y = df[metric].to_numpy()
            break
    return X, y
    
    
    


def add_observation_to_selection_record_file(selection_record_path, 
                                             iteration, 
                                             selection_vector,
                                             eval_metrics_values_list,
                                             father_iteratioin=None,
                                             update_flag=" "): 
                                            
    record_line = []
    if father_iteratioin is None:
        if update_flag==" ":
            # backtrack, none
            record_line = [iteration] + list(selection_vector) + eval_metrics_values_list
        else:
            # random, polling
            record_line = [iteration] + list(selection_vector) + eval_metrics_values_list + [update_flag]
            
    else:
        # single metric
        record_line = [iteration] + list(selection_vector) + eval_metrics_values_list + [father_iteratioin] + [update_flag]
    record_df = pd.DataFrame([record_line])
    record_df.to_csv(selection_record_path, sep='\t', index=False, header=False, mode='a')




