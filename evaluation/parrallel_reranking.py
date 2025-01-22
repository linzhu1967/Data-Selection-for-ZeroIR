import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

import pandas as pd
from tqdm import tqdm
from itertools import chain



def rerank_ddp_and_save(model, gpus_num, device, eval_dataset, label_token_ids, per_device_eval_batch, rerank_saved_path):
    rank = dist.get_rank()
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=gpus_num, rank=rank)
    eval_loader = DataLoader(eval_dataset, batch_size=per_device_eval_batch, sampler=eval_sampler, num_workers=8)

    token_false_id, token_true_id = label_token_ids
    per_device_results = []
    if not isinstance(model, DDP):
        model = DDP(model, device_ids=[rank])
    model.eval()
    # assert torch.distributed.get_world_size() == 4, "Should be running on 4 GPUs."  
    with torch.inference_mode():
        for bt_i, batch in enumerate(tqdm(eval_loader)):                      
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model.module.generate(**inputs,
                                            max_new_tokens=1,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            )
            batch_scores = outputs.scores[0]
            batch_scores = batch_scores[:, [token_false_id, token_true_id]]
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)     # in (-,0]
            rel_scores = batch_scores[:, 1].exp().tolist()               
            # irrel_scores = batch_scores[:, 0].exp().tolist()          
            
            per_device_results.extend(zip(batch["qid"], batch["docid"], rel_scores))
    
    all_results_nested_list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(all_results_nested_list, per_device_results)
    all_results = list(chain.from_iterable(all_results_nested_list))

    if rank == 0:
        predicted_df = pd.DataFrame(all_results, columns=['qid', 'docid','score'])
        predicted_df["_1"] = "Q0"
        predicted_df["ranker"] = "MonoT5"
        predicted_df = predicted_df.groupby("qid").apply(lambda x:x.sort_values("score", ascending=False)).reset_index(drop=True)
        predicted_df = predicted_df.groupby('qid').apply(lambda x: x.drop_duplicates(subset='docid')).reset_index(drop=True)
        predicted_df["rank"]=predicted_df.groupby("qid").cumcount() + 1
        predicted_df = predicted_df[["qid", "_1", "docid", "rank", "score", "ranker"]]
        
        saved_path = rerank_saved_path
        predicted_df.to_csv(saved_path, index=False, sep="\t", header=False, float_format='%.15f')
    
    dist.barrier()
    
    return "[rerank_ddp_and_save] Done!"




