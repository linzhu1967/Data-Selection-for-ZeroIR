import os
import torch
import torch.distributed as dist


from transformers import (
    Seq2SeqTrainer,
)

from evaluation.parrallel_reranking import rerank_ddp_and_save
from evaluation.trec_evaluate import run_trec_eval

from utils.global_invariants import TEST_METRICS_DECIMAL


class RerankerTrainer(Seq2SeqTrainer):
    def __init__(self, *args, evaluate_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_args=evaluate_args
       
    # rewrite the evaluate method to return the validation loss and the predictions
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Custom evaluation function to override the default evaluation process.
        Evaluate the reranker on the validation dataset.
        """
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        self.model.eval()
        
        eval_metric = {"ndcg_cut_10": 0.0}
        local_rank = int(os.getenv('LOCAL_RANK', -1))
        device = torch.device("cuda", local_rank)
        rerank_saved_path=self.evaluate_args["rerank_saved_path"].format(str(self.state.epoch))
        rerank_ddp_and_save(self.model,
                            gpus_num=torch.cuda.device_count(),
                            device=device, 
                            eval_dataset=eval_dataset, 
                            label_token_ids=self.evaluate_args["label_token_ids"], 
                            per_device_eval_batch=self.args.per_device_eval_batch_size, 
                            rerank_saved_path=rerank_saved_path)
        
        new_tensors = torch.zeros(len(eval_metric)).to(device)
        
        if torch.distributed.get_rank() in [0,-1]:
            eval_results = run_trec_eval(rerank_saved_path, self.evaluate_args["qrels_file"])
            
            for idx, metric in enumerate(eval_metric.keys()):
                if metric in eval_results:
                    metric_value = eval_results[metric]
                    # print and record
                    print(f"###### epoch{self.state.epoch}_{metric}: {metric_value}")
                    with open(self.evaluate_args["selection_record_path"], "a") as f:
                        f.write(f"###### epoch{self.state.epoch}_{metric}: {metric_value}\n")
                        
                    new_tensors[idx] = torch.tensor([metric_value], device="cuda:0")
        
        dist.barrier()
        
        if dist.is_initialized():
            dist.broadcast(tensor=new_tensors, src=0)
            new_values = [round(new_tensor.item(), TEST_METRICS_DECIMAL) for new_tensor in new_tensors]
        
        if new_values[0] > self.evaluate_args["best_m1_value"]:
            print(f"****** epoch{self.state.epoch} update the [best_m1_value] and save the model")
            self.evaluate_args["best_m1_value"] = new_values[0]
            self.save_model(output_dir=self.evaluate_args["best_ckpt_saved_dir"])
            
        for idx, metric in enumerate(eval_metric.keys()):
            eval_metric[metric] = new_values[idx]
        
        print(f"Evaluation Results: {eval_metric}")
        
        return eval_metric