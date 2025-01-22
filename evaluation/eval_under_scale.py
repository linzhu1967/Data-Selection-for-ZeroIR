import os
import torch
import torch.distributed as dist
import subprocess

from evaluation.parrallel_reranking import rerank_ddp_and_save
from evaluation.trec_evaluate import run_trec_eval

from utils.global_invariants import TEST_METRICS_DECIMAL, TEST_SMALL_PCTGS, TEST_LARGE_PCTG



# validation interface
def eval_under_pctg(pctg, eval_dataset, qrels_file, label_token_ids, eval_metrics, rerank_saved_path, 
                    num_gpus, device, new_model, iter_flag, training_args, extra_args, target_dataset_name=""):
    
    eval_dataset_name = extra_args.target_eval_dataset_name if len(target_dataset_name)==0 else target_dataset_name
    
    if torch.distributed.get_rank() in [0,-1]:
        if pctg in TEST_SMALL_PCTGS:
            print(f"Iteration[{iter_flag}]: Evaluate the model under a small percentage ({pctg}) of the ({eval_dataset_name}) test dataset.")
        elif pctg in TEST_LARGE_PCTG:
            print(f"Iteration[{iter_flag}]: Evaluate the model under a large percentage ({pctg}) of the ({eval_dataset_name}) test dataset.")
        else:
            print(f"Iteration[{iter_flag}]: Evaluate the model under an unknown percentage ({pctg}) of ({eval_dataset_name}) the test dataset.")
        
    # rerank
    if os.path.exists(rerank_saved_path):
        if torch.distributed.get_rank() in [0,-1]:
            print(f"###### iteration{iter_flag}_{eval_dataset_name}_{rerank_saved_path} already exists.")
    else:
        rerank_ddp_and_save(new_model, 
                            num_gpus,
                            device, 
                            eval_dataset, 
                            label_token_ids, 
                            training_args.per_device_eval_batch_size, 
                            rerank_saved_path)
    
    new_tensors = torch.zeros(len(eval_metrics)).to(device)
    if torch.distributed.get_rank() in [0,-1]:
        eval_results = run_trec_eval(rerank_saved_path, qrels_file)
        for idx, metric in enumerate(eval_metrics):
            if metric in eval_results:
                metric_value = eval_results[metric]
                print(f"###### iteration{iter_flag}_{eval_dataset_name}_{rerank_saved_path}_{metric}_{pctg}: {metric_value}")
                new_tensors[idx] = torch.tensor([metric_value], device="cuda:0")
    dist.barrier()
    if dist.is_initialized():
        dist.broadcast(tensor=new_tensors, src=0)
        new_values = [round(new_tensor.item(), TEST_METRICS_DECIMAL) for new_tensor in new_tensors]
    
    return new_values


def combine_into_total_eval_results(rerank_saved_path_list, total_rerank_saved_path, qrels_file, eval_metrics):
    cmd = "cat {} {} | sort | uniq".format(rerank_saved_path_list[0],
                                        rerank_saved_path_list[1])

    with open(total_rerank_saved_path, 'w') as outfile:
        subprocess.run(cmd, shell=True, stdout=outfile, check=True)
    
    if torch.distributed.get_rank() in [0,-1]:
        print(f"###### Combine the results of 10% and 20% into {total_rerank_saved_path}") 
        print(f"###### Evaluate the combined results of 10% and 20% on {qrels_file}.")   
    eval_results = run_trec_eval(total_rerank_saved_path, qrels_file)
    metric_values = [eval_results[metric] for metric in eval_metrics if metric in eval_results]

    return metric_values