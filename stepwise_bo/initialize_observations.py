import os
import math
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    Trainer,
    Seq2SeqTrainer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Subset
import torch.distributed as dist

from pyserini.search import get_qrels_file

from evaluation.parrallel_reranking import rerank_ddp_and_save
from evaluation.trec_evaluate import run_trec_eval

from stepwise_bo.record_bo import init_selection_record_file
from stepwise_bo.record_bo import add_observation_to_selection_record_file
from stepwise_bo.record_bo import get_lines_num

from stepwise_bo.pruning import get_initial_pruning_idxs

        
        
def get_initial_observations_with_historical_status_and_pruning(training_berri_datasets, 
                             eval_rerank_beir_datasets, 
                             selection_record_path, 
                             initial_reranking_dir, 
                             training_args, 
                             extra_args, 
                             label_token_ids, 
                             device,
                             eval_pctg_list):
    """description:
    1. check and get the initial observations for the training_berri_datasets
    2. read the historical status from the selection_record_path
    """
    
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    
    if local_rank in [0,-1]:
        init_selection_record_file(selection_record_path, training_berri_datasets, extra_args.eval_metrics, eval_pctg_list)
    dist.barrier()

    observation_num = get_lines_num(selection_record_path)
    if observation_num >= len(training_berri_datasets): 
        print("Already have observations for [global_berri_datasets].")
        initial_pruned_idxs = get_initial_pruning_idxs(selection_record_path, extra_args.search_strat)
        return observation_num, initial_pruned_idxs
    
    # the fold to save the reranked results
    initial_reranking_target_dir = os.path.join(initial_reranking_dir, "{}.test{}.epoch{}/".format(extra_args.target_eval_dataset_name, 
                                                                                               eval_pctg_list[0],
                                                                                               int(training_args.num_train_epochs)))
    if local_rank in [0,-1] and not os.path.exists(initial_reranking_target_dir):
        os.makedirs(initial_reranking_target_dir)
    
    for berri_name, berri_dataset in training_berri_datasets.items():
        
        # vector representation of the selected dataset for training
        selection_index = berri_name.split("_")[0]
        if int(selection_index) < observation_num:
            continue
        
        # init the file to save the reranked berri_dataset results
        initial_rerank_saved_path = os.path.join(initial_reranking_target_dir, 
                                    "{}_to_{}.{}.tsv".format(berri_name, extra_args.target_eval_dataset_name, eval_pctg_list[0]))
        
        # init the selection_vector
        selection_vector = [0] * len(training_berri_datasets)
        len_berri_dataset = len(berri_dataset)
        if len_berri_dataset < extra_args.source_data_chunk*extra_args.select_chunk_number:
            selection_vector[int(selection_index)] = math.ceil(len_berri_dataset/extra_args.source_data_chunk)
            one_select_chunks_dataset = berri_dataset
        else:    
            selection_vector[int(selection_index)] = extra_args.select_chunk_number # 1
            # build subset of the dataset
            one_select_chunks_dataset = Subset(berri_dataset, list(range(extra_args.source_data_chunk*extra_args.select_chunk_number)))
        
        if local_rank in [0,-1]:
            print("# get initial observation on beir-[{}]".format(berri_name))
        
        if not os.path.exists(initial_rerank_saved_path):        
            # train
            config = AutoConfig.from_pretrained(extra_args.base_model)
            tokenizer = AutoTokenizer.from_pretrained(extra_args.base_model)
            
            seq2seq = any(
                True
                for architecture in config.architectures
                if 'ForConditionalGeneration' in architecture or 'T5Model' in architecture
            )

            if seq2seq:
                model = AutoModelForSeq2SeqLM.from_pretrained(extra_args.base_model)
                trainer_cls = Seq2SeqTrainer
                data_collator = DataCollatorForSeq2Seq(tokenizer)
            else:
                config.num_labels = 1
                config.problem_type = 'multi_label_classification'
                model = AutoModelForSequenceClassification.from_pretrained(
                    extra_args.base_model,
                    config=config,
                )
                trainer_cls = Trainer
                data_collator = DataCollatorWithPadding(tokenizer)
                
            trainer = trainer_cls(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=one_select_chunks_dataset,
                data_collator=data_collator,
            )
            train_metrics = trainer.train()
            print("[Initialization with berri_{}]: train_metrics {} ".format(berri_name, train_metrics.metrics))
            
            # rerank & evaluate
            print("# rerank & evaluate on beir-[{}]".format(extra_args.target_eval_dataset_name))
            model.eval()
            with torch.inference_mode():
                eval_dataset_list = eval_rerank_beir_datasets[extra_args.target_eval_dataset_name]
                small_pctg_index = 0
                small_eval_dataset = eval_dataset_list[small_pctg_index]
                rerank_ddp_and_save(model, 
                                    torch.cuda.device_count(),
                                    device, 
                                    small_eval_dataset, 
                                    label_token_ids, 
                                    training_args.per_device_eval_batch_size, 
                                    initial_rerank_saved_path)
            
            dist.barrier()
            del model
            del trainer_cls
            del trainer
            del train_metrics
            torch.cuda.empty_cache()
        
        
        # evaluate the reranked results        
        dist.barrier()
        if torch.distributed.get_rank() in [0,-1]:
            qrels_file = get_qrels_file(f"beir-v1.0.0-{extra_args.target_eval_dataset_name}-test")
            eval_metrics_values = []
            
            eval_results = run_trec_eval(initial_rerank_saved_path, qrels_file)
            print(eval_results.keys())
            for metric in extra_args.eval_metrics:
                if metric in eval_results:
                    metric_value = eval_results[metric]
                    eval_metrics_values.extend([metric_value, 0, 0])
                    print(f"###### {extra_args.target_eval_dataset_name}_{initial_rerank_saved_path}_{metric}: {metric_value}")
                else:
                    print("The metric [{}] is not in the eval_results.".format(metric))    
            add_observation_to_selection_record_file(selection_record_path, 0, selection_vector, eval_metrics_values)

    dist.barrier()
    initial_pruned_idxs = get_initial_pruning_idxs(selection_record_path, extra_args.search_strat)
    
    return len(training_berri_datasets), initial_pruned_idxs


