import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)
from torch.utils.data import Subset
import torch.distributed as dist

import math
import numpy as np
import json
import gc

from pyserini.search import get_qrels_file


from data.berri_datasets import BerriPairMonoT5Dataset
from data.beir_datasets import BeirBM25TopkMonoT5Dataset
from data.load_datasets import load_beir_queries, load_beir_corpus

from stepwise_bo.gp_ei import select_next_training_data
from stepwise_bo.initialize_observations import get_initial_observations_with_historical_status_and_pruning
from stepwise_bo.record_bo import add_observation_to_selection_record_file
from stepwise_bo.pruning import get_berri_idxs_to_prune_quarter


from evaluation.eval_under_scale import eval_under_pctg
from evaluation.eval_under_scale import combine_into_total_eval_results

from utils.extra_arguments import ExtraArguments
from utils.global_invariants import PREDICTION_TOKENS, BERRI_IDX_DICT 
from utils.global_invariants import TEST_PCTGS_DICT
from utils.global_invariants import BEIR_TEST_PCTG_DICT
from utils.utils import delete_model_folder
from utils.utils import custom_encoder



# Initialize the global variables
global_berri_datasets = {
    "0_agnews": None,
    "1_altlex": None,
    "2_cnn_dailymail": None,
    "3_coco_captions": None,
    "4_eli5_question_answer": None,
    "5_fever": None,
    "6_gigaword": None,
    "7_hotpotqa": None,
    "8_mdmcqa": None,
    "9_medical_sim": None,
    "10_npr": None,
    "11_nq": None,
    "12_oqa": None,
    "13_pubmedqa": None,
    "14_qrecc": None,
    "15_quora": None,
    "16_record": None,
    "17_scitldr": None,
    "18_searchQA_top5_snippets": None,
    "19_sentence-compression": None,
    "20_squad_pairs": None,
    "21_stackexchange_duplicate_questions_title-body_title-body": None,
    "22_stackexchange_duplicate_questions_title_title": None,
    "23_triviaqa": None,
    "24_wikihow": None,
    "25_wow": None,
    "26_xsum-multilexsum": None,
    "27_nan": None, 
}
global_beir_datasets = {
    "trec-covid": None,               
    "nfcorpus": None,                 
    "fiqa": None,                     
    "arguana": None,                  
    "webis-touche2020": None,         
    "dbpedia-entity": None,           
    "scidocs": None,                  
    "climate-fever": None,            
    "scifact": None,                  
}
global_berri_datasum_rep = None # Count the number of data chunks in each dataset in BERRI
global_berri_selected_data_rep = [0]*len(global_berri_datasets) # Record the number of data chunks currently selected

def update_global_berri_selected_data_rep(selected_idx):
    global global_berri_selected_data_rep
    global_berri_selected_data_rep[selected_idx] += 1


def read_global_berri_selected_data_rep(history_selected_berri_rep):
    global global_berri_selected_data_rep
    global_berri_selected_data_rep = history_selected_berri_rep


def select_dataset_with_delete(iter_selected_data_rep, father_data_rep, source_data_chunk):
    selected_index = np.where(iter_selected_data_rep != father_data_rep)[0].tolist()[0]
    s_berri_name = BERRI_IDX_DICT[selected_index]
    s_berri_dataset = global_berri_datasets[s_berri_name]
    
    # Determine the index
    begin_idx = global_berri_selected_data_rep[selected_index]*source_data_chunk
    if begin_idx+1*source_data_chunk < len(s_berri_dataset):
        end_idx = begin_idx+1*source_data_chunk  
    else:
        end_idx = len(s_berri_dataset)
        
    # Take a chunk from the berri dataset
    selected_data_dataset = Subset(s_berri_dataset, list(range(begin_idx, end_idx)))
    
    selected_data_jsonl_str = ""
    for i in range(begin_idx, end_idx):
        selected_data_jsonl_str += json.dumps(s_berri_dataset.data[i])+"\n"
        
    # update
    update_global_berri_selected_data_rep(selected_index)
    
    return selected_data_dataset, selected_data_jsonl_str


def init_berri_datasets(extra_args, tokenizer):
    global global_berri_datasets
    for berri_idx in global_berri_datasets.keys():
        print(f"# Loading the dataset BERRI-{berri_idx}......")
        datapath = extra_args.source_training_pairs_dir.format(berri_idx)
        if global_berri_datasets[berri_idx] is None:
            global_berri_datasets[berri_idx] = BerriPairMonoT5Dataset(datapath, tokenizer, extra_args.max_doc_length)


def init_berri_datasum_vector_representation(extra_args):
    global global_berri_datasum_rep
    if global_berri_datasum_rep is None:
        datasum_rep_list= []
        for i in global_berri_datasets.keys():
            sum_rep = math.ceil(len(global_berri_datasets[i])/extra_args.source_data_chunk)
            datasum_rep_list.append(sum_rep) 
    else:
        print("The [global_berri_datasum_rep] has been initialized.")

    global_berri_datasum_rep = tuple(datasum_rep_list)
    
    return global_berri_datasum_rep


def init_beir_datasets(extra_args, tokenizer, eval_pctg_list):
    global global_beir_datasets
    print(f"# Loading the dataset BEIR-{extra_args.target_eval_dataset_name} with {eval_pctg_list} bm25_top1000 pairs......")
    if global_beir_datasets[extra_args.target_eval_dataset_name] is None:
        global_beir_datasets[extra_args.target_eval_dataset_name] = []

    # load corpus and queries
    print(f"# Loading the BEIR-{extra_args.target_eval_dataset_name} corpus and queries......")
    corpus = load_beir_corpus(extra_args.target_eval_dataset_name)
    corpus = dict(zip(corpus['doc_id'], corpus['text']))
    queries = load_beir_queries(extra_args.target_eval_dataset_name)
    
    # read the three topk files with different data volume percentages (, , 100%) to the global_beir_datasets list
    for pctg in eval_pctg_list:
        if pctg != "100%":
            input_topk_file = extra_args.target_eval_bm25topk_dir + "run.beir-v1.0.0-{}-flat.trec".format(extra_args.target_eval_dataset_name+"."+pctg)
            global_beir_datasets[extra_args.target_eval_dataset_name].append(BeirBM25TopkMonoT5Dataset(input_topk_file, 
                                                                                                extra_args.target_eval_dataset_name, 
                                                                                                extra_args.eval_bm25topk, 
                                                                                                tokenizer, 
                                                                                                extra_args.max_doc_length,
                                                                                                corpus,
                                                                                                queries)
                                                                        )   
    


    
    
def sds_select(training_args, extra_args):  
    
    # ------------------------- Step-0 -------------------------
    # set random seed
    set_seed(training_args.seed)
    print("[seed]: ", training_args.seed)
    # set training arguments
    training_args.dataloader_shuffle=False              
    training_args.dataloader_drop_last=False            
    training_args.logging_dir='logs/training_logs'      
    training_args.logging_steps=1
    if len(extra_args.eval_pctg) == 0:
        extra_args.eval_pctg = BEIR_TEST_PCTG_DICT[extra_args.target_eval_dataset_name]

    
    # ------------------------- Step-1 -------------------------
    # Initialize ddp
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    num_gpus = torch.cuda.device_count()    
    # Check whether the environment is distributed (that is, local_rank is not -1).
    if local_rank != -1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://')
            print(f"Let's use {num_gpus} GPUs!")
        else:
            print("The distributed setting is disable.")
            sys.exit()
    else:
        print("Not using distributed mode")
    
    
    
    # ------------------------- Step-2 -------------------------
    # Create the [dir] for recording the reranking performance and the reranking outputs
    base_model_name = extra_args.base_model.split("/")[-1]
    selection_record_full_dir = os.path.join(extra_args.data_selection_record_dir, 
                                            "{}_gpu{}_bsize{}_gaccu{}_select{}_chunk{}/".format(base_model_name, \
                                                str(num_gpus), 
                                                str(training_args.per_device_train_batch_size),
                                                str(training_args.gradient_accumulation_steps),
                                                str(extra_args.select_chunk_number),
                                                str(extra_args.source_data_chunk),), 
                                            "random{}/".format(str(training_args.seed)),
                                            )
    if local_rank in [-1,0] and not os.path.exists(selection_record_full_dir):
        os.makedirs(selection_record_full_dir)
        
    # Create the [file] for recording the iterative reranking performance on the target dataset (BEIR)
    selection_record_name = "BERRI_to_{}.test{}.{}.epoch{}".format(extra_args.target_eval_dataset_name,
                                                                   extra_args.eval_pctg,
                                                                   extra_args.search_strat,
                                                                   int(training_args.num_train_epochs)
                                                                   )
    selection_record_path = os.path.join(selection_record_full_dir, "{}.tsv".format(selection_record_name))
    selection_status_path = os.path.join(selection_record_full_dir, "{}.status.json".format(selection_record_name))
    

    # Create the [dir] for saving the reranking outputs in the initialization and the selection process (preparing for trec test)
    intial_reranking_output_dir= os.path.join(selection_record_full_dir, "reranking_outputs/BERRI_initial_observations/")
    if local_rank in [-1,0] and not os.path.exists(intial_reranking_output_dir):
        os.makedirs(intial_reranking_output_dir)
    reranking_output_dir= os.path.join(selection_record_full_dir, "reranking_outputs/{}/".format(selection_record_name))
    if local_rank in [-1,0] and not os.path.exists(reranking_output_dir):
        os.makedirs(reranking_output_dir)
        
    # Create the [dir] for saving rerankers        
    reranker_saved_dir = os.path.join(training_args.output_dir,
                                      "{}_gpu{}_bsize{}_gaccu{}_select{}_chunk{}/".format(base_model_name, \
                                          str(num_gpus), 
                                          str(training_args.per_device_train_batch_size),
                                          str(training_args.gradient_accumulation_steps),
                                          str(extra_args.select_chunk_number),
                                          str(extra_args.source_data_chunk)),
                                      "random{}/{}/".format(str(training_args.seed), selection_record_name),
                                      )
    if local_rank in [-1,0] and not os.path.exists(reranker_saved_dir):
        os.makedirs(reranker_saved_dir)
        
        
    # TODO
    # create the [file] for saving the selected data
    selected_data_saved_dir = os.path.join(selection_record_full_dir, "selected_berri_data/")
    if local_rank in [-1,0] and not os.path.exists(selected_data_saved_dir):
        os.makedirs(selected_data_saved_dir)
    selected_data_saved_file = os.path.join(selected_data_saved_dir, "{}.json".format(selection_record_name))
    seperate_selected_data_saved_file_template = "iter{iter_flag}.{selection_record_name}.json"
    



    # ------------------------- Step-3 -------------------------
    # Metrics
    metric_1 = extra_args.eval_metrics[0]   # "ndcg_cut_10"
        
    # Create the [dir] for saving the best rerankers' ckpt   
    best_m1_ckpt_saved_dir = os.path.join(reranker_saved_dir, "best_m1_{}_ckpt/".format(metric_1))
    if local_rank in [-1,0]:
        if not os.path.exists(best_m1_ckpt_saved_dir):
            os.makedirs(best_m1_ckpt_saved_dir)

    # State the [dir] for saving the candidate rerankers' ckpt
    candidate_ckpt_saved_dir = os.path.join(reranker_saved_dir, "iter{}_ckpt/")
    
        
    
    # ------------------------- Step-4 -------------------------
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(extra_args.base_model)
    # Get the eval_pctg_list
    eval_pctg_list = TEST_PCTGS_DICT[extra_args.eval_pctg]
    
    # Initialize the berri datasets for training
    if local_rank in [-1,0]:
        print("# Initializing berri datasets......")
    init_berri_datasets(extra_args, tokenizer)
    
    # Initialize the berri datasum vector representation
    init_berri_datasum_vector_representation(extra_args)
    
    # Initialize the beir datasets for testing
    if local_rank in [-1,0]:
        print("# Initializing beir datasets[{}]......".format(extra_args.target_eval_dataset_name))
    init_beir_datasets(extra_args, tokenizer, eval_pctg_list)
    dist.barrier()
    
    
    
    # ------------------------- Step-5 -------------------------
    # Initialize the label_token_ids and the qrels_file for different percentage
    token_false, token_true = [None, None]
    model_name = extra_args.base_model.split("/")[-1].lower()
    if model_name in PREDICTION_TOKENS:
        token_false, token_true = PREDICTION_TOKENS[model_name]
    else:
        print("The model name is not in the PREDICTION_TOKENS.")
        sys.exit()
        
    label_token_ids = [tokenizer.get_vocab()[token_false],tokenizer.get_vocab()[token_true]]
    # Initialize the eval_dataset_list and the qrels_file
    eval_dataset_list = global_beir_datasets[extra_args.target_eval_dataset_name]
    qrels_file = ""
    if torch.distributed.get_rank() in [0,-1]:
        qrels_file = get_qrels_file(f"beir-v1.0.0-{extra_args.target_eval_dataset_name}-test")
    # small pctg and its evaluation dataset
    small_pctg_index = 0
    small_pctg = eval_pctg_list[small_pctg_index]
    small_eval_dataset = eval_dataset_list[small_pctg_index]
    # large pctg and its evaluation dataset
    large_pctg_index = 1
    large_pctg = eval_pctg_list[large_pctg_index]
    large_eval_dataset = eval_dataset_list[large_pctg_index]
    # total pctg
    total_pctg_index = 2
    total_pctg = eval_pctg_list[total_pctg_index]



    # ------------------------- Step-6 -------------------------
    # Initialization for BO
    total_iters = extra_args.iter_number if extra_args.iter_number else 100
    total_loss = 0.0
    iter_flag = 0           
    father_iter_flag = 0    
    flagged_child_father_dict = {}  
    
    best_m1 = 0.0           # best ndcg_cut_10
    best_m1_iter_flag = 0   
    best_m1_berri_selected_data_rep = np.array([0]*len(global_berri_datasets.keys()))
    
    best_iter_list = [best_m1_iter_flag]
                                        # (father_iter_flag, iter_flag, iter_metric, new_m1_value, new_rep, new_ckpt)
    iter_info_dict = {best_m1_iter_flag:(None, best_m1_iter_flag, None, best_m1, best_m1_berri_selected_data_rep, extra_args.base_model)} 

    pruned_berri_idxs = []  
    
    # Get initial observations for data selection and initially prune quarter
    initial_observations_num, initial_pruned_idxs = get_initial_observations_with_historical_status_and_pruning(global_berri_datasets, 
                                                                                                                global_beir_datasets, 
                                                                                                                selection_record_path, 
                                                                                                                intial_reranking_output_dir, 
                                                                                                                training_args, 
                                                                                                                extra_args, 
                                                                                                                label_token_ids, 
                                                                                                                device,
                                                                                                                eval_pctg_list)

    
    if extra_args.initial_pruning or "init-pruning" in extra_args.search_strat:
        print("Initial pruning......")
        print("Initial pruning num: ", len(initial_pruned_idxs))
        pruned_berri_idxs = initial_pruned_idxs

    dist.barrier()
    
    
    # ------------------------- Step-7 -------------------------
    if initial_observations_num > len(BERRI_IDX_DICT.keys()):
        # selection_status_path
        with open(selection_status_path, "r") as r_json_file:  
            historical_status = json.load(r_json_file)
            father_iter_flag = historical_status["father_iter_flag"]
            iter_flag = historical_status["iter_flag"]
            best_iter_list = historical_status["best_iter_list"]
            iter_info_dict = historical_status["iter_info_dict"]
            iter_info_dict = {int(k):tuple(v) for k,v in iter_info_dict.items()} 
            pruned_berri_idxs = historical_status["pruned_berri_idxs"]
            best_m1 = historical_status["best_m1"]
            best_m1_iter_flag = historical_status["best_m1_iter_flag"]
            best_m1_berri_selected_data_rep = np.array(historical_status["best_m1_berri_selected_data_rep"])
            flagged_child_father_dict = historical_status["flagged_child_father_dict"]
        
        read_global_berri_selected_data_rep(historical_status["selected_berri_rep"])
        
        
        
        print(f"We have {initial_observations_num} initial observations, " \
            f"and then we will select data from the {iter_flag+1}-th iteration for the rest search space.")
    dist.barrier()
    
    # ------------------------- Step-8 -------------------------
    # Begin training with data selection based on BO
    # Select data based on the best_m1 or best_m2
    while best_iter_list and iter_flag < total_iters:        
        # father iteration
        father_info = iter_info_dict[father_iter_flag] # (father_iter_flag, iter_flag, iter_metric, new_m1_value, new_rep, new_ckpt)
        father_m1 = father_info[3]
        father_rep = father_info[4] if isinstance(father_info[4], np.ndarray) else np.array(father_info[4])
        father_model_ckpt = father_info[-1]
        # select data
        new_rep = select_next_training_data(selection_record_path=selection_record_path,
                                            current_berri_selected_rep=father_rep,
                                            berri_datasum_rep=global_berri_datasum_rep,
                                            selected_sum=extra_args.select_chunk_number,
                                            pruned_berri_idxs=pruned_berri_idxs,
                                            used_berri_rep=global_berri_selected_data_rep)
        
        # check whether new_m1_rep is empty
        if not isinstance(new_rep, np.ndarray):
            if local_rank in [-1,0]:
                print("For the father_rep {}, the search space is empty.".format(father_rep))
                if not father_model_ckpt == extra_args.base_model:
                    if os.path.exists(father_model_ckpt):
                        print(f"Delete the father_model_ckpt: [{father_model_ckpt}]")
                        delete_model_folder(father_model_ckpt)
            dist.barrier()
                
            # update the pruned_berri_idxs
            pruned_berri_idxs = get_berri_idxs_to_prune_quarter(father_iter_flag, father_rep, iter_info_dict, pruned_berri_idxs)
            print("[1/4] Add new pruned_berri_idxs: [{pruned_berri_idxs}]")
            
            # update the best_iter_list and father_iter_flag
            best_iter_list.remove(father_iter_flag)
            if best_iter_list:
                father_iter_flag = best_iter_list[-1]
                print(f"Update the father_iter_flag with the following value: [{father_iter_flag}]")
            else:
                break
            
            continue
        
        # start a new iteration
        iter_flag+=1
        
        # the new selected data from new_m1_rep
        new_selected_dataset, new_selected_data_jsonl_str = select_dataset_with_delete(new_rep, father_rep, extra_args.source_data_chunk)

        # train distributedly
        if local_rank in [-1,0]:
            print("--------------------------------------------------------------")
            print(f"# iteration{iter_flag}_selected_data_rep: {new_rep}")
            print("--------------------------------------------------------------")
            print("# training......")
        
        new_model = AutoModelForSeq2SeqLM.from_pretrained(father_model_ckpt)
        trainer = Seq2SeqTrainer(
            model=new_model,
            args=training_args,
            data_collator=DataCollatorForSeq2Seq(tokenizer),
            train_dataset=new_selected_dataset,
        )
        train_metrics = trainer.train()
        print(f"###### iteration{iter_flag}_train_metrics: {train_metrics}")
        
        # Distributed testing of the new_m1_model's reranking performance on the 10%~20% test data
        print("# rerank & evaluate on beir-[{}]".format(extra_args.target_eval_dataset_name))    
        new_model.eval()
        
        with torch.inference_mode():
            rerank_saved_path = os.path.join(reranking_output_dir, "iter{}.{}.tsv".format(str(iter_flag), small_pctg))
            new_m1_value, new_m2_value = eval_under_pctg(small_pctg, small_eval_dataset, qrels_file, label_token_ids, 
                                                         extra_args.eval_metrics, rerank_saved_path, num_gpus, device, 
                                                         new_model, iter_flag, training_args, extra_args)

            new_ckpt = candidate_ckpt_saved_dir.format(iter_flag)   
            iter_metric = metric_1 if new_m1_value > father_m1 else "" 
            update_flag = iter_metric
            # add the current iter info into the iter_info_dict                                    
            iter_info_dict[iter_flag] = (father_iter_flag, iter_flag, iter_metric, new_m1_value, new_rep, new_ckpt)
            
            if len(iter_metric) != 0:# if new_m1_value > father_m1:                  
                # update the best_iter_list
                best_iter_list.append(iter_flag)
                
                # save model
                if torch.distributed.get_rank() in [0,-1] and not os.path.exists(new_ckpt):
                    os.makedirs(new_ckpt)
                dist.barrier()
                trainer.save_model(new_ckpt)

            # additionally, compare with the best
            if new_m1_value > best_m1:
                best_m1_iter_flag = iter_flag
                best_m1_berri_selected_data_rep = new_rep
                trainer.save_model(best_m1_ckpt_saved_dir)
                best_m1 = new_m1_value
                update_flag = update_flag+"_bestm1"
                
            
            if len(iter_metric) != 0:
                # save the flagged child-father relationship
                flagged_child_father_dict[iter_flag] = father_iter_flag
                
                # TODO: SAVE THE CURRENT SELECTED DATA to the seperate file 
                if torch.distributed.get_rank() in [0,-1]:
                    seperate_selected_data_saved_path = os.path.join(selected_data_saved_dir, 
                                                                     seperate_selected_data_saved_file_template.format(\
                                                                         iter_flag=iter_flag,
                                                                         selection_record_name=selection_record_name))    
                    with open(seperate_selected_data_saved_path, "w") as f:
                        f.write(new_selected_data_jsonl_str)
                        
                    
            if torch.distributed.get_rank() in [0,-1]:
                metrics_values = [new_m1_value, 0, 0, new_m2_value, 0, 0]
                add_observation_to_selection_record_file(selection_record_path, iter_flag, new_rep, 
                                                        metrics_values, father_iter_flag, update_flag)
            dist.barrier()    
            
        # save the current status to the file
        if torch.distributed.get_rank() in [0,-1]:
            status_dict = {
                "father_iter_flag": father_iter_flag,
                "iter_flag": iter_flag,
                "best_iter_list": best_iter_list,
                "flagged_child_father_dict": flagged_child_father_dict,
                "pruned_berri_idxs": pruned_berri_idxs,
                "selected_berri_rep": global_berri_selected_data_rep,
                "best_m1": best_m1,
                "best_m1_iter_flag": best_m1_iter_flag,
                "best_m1_berri_selected_data_rep": best_m1_berri_selected_data_rep,
                "iter_info_dict": iter_info_dict,
            }
            try:
                json_string = json.dumps(status_dict, indent=4, default=custom_encoder)
            except TypeError as e:
                print("[Save status]: A non-serializable object was found:", e)
            else:
                with open(selection_status_path, "w") as json_file: 
                    json_file.write(json_string)
        dist.barrier()

        if local_rank in [-1,0]:
            print("--------------------------------------------------------------")
            print("iter_flag")
            print(iter_flag)
            print("--------------------------------------------------------------")
            print("best_iter_list")
            print(best_iter_list)
            print("father_iter_flag")
            print(father_iter_flag)
            print("--------------------------------------------------------------")
            
        
        # update the father_iter_flag for the next iteration        
        if new_m1_value > father_m1:
            father_iter_flag = iter_flag
            if torch.distributed.get_rank() in [0,-1]:
                print("--------------------------------------------------------------")
                print(f"[Iter_flag {iter_flag}]: Update father_iter_flag to {father_iter_flag} due to new_m1_value[{new_m1_value}] > father_m1[{father_m1}].")     


        dist.barrier()


    
    del new_model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

        
    # ------------------------- Step-8.4 -------------------------
    # Evaluate the best_m1 model on the 100% test data
    best_m1_model = AutoModelForSeq2SeqLM.from_pretrained(best_m1_ckpt_saved_dir).to(device)
    best_m1_large_pctg_rerank_saved_path = os.path.join(reranking_output_dir, "iter{}.{}.tsv".format(str(best_m1_iter_flag), large_pctg))
    best_m1_model_large_pctg_m1_value, best_m1_model_large_pctg_m2_value = eval_under_pctg(large_pctg, large_eval_dataset, qrels_file, label_token_ids, 
                                                                                extra_args.eval_metrics, best_m1_large_pctg_rerank_saved_path, 
                                                                                num_gpus, device, best_m1_model, best_m1_iter_flag, training_args, 
                                                                                extra_args)

    if torch.distributed.get_rank() in [0,-1]:
        # m1
        rerank_saved_path_list = [os.path.join(reranking_output_dir, "iter{}.{}.tsv".format(str(best_m1_iter_flag), small_pctg)), 
                                    best_m1_large_pctg_rerank_saved_path]
        total_rerank_saved_path = os.path.join(reranking_output_dir, "iter{}.{}.tsv".format(str(best_m1_iter_flag), total_pctg))
        total_metrics_values = combine_into_total_eval_results(rerank_saved_path_list, total_rerank_saved_path, qrels_file, extra_args.eval_metrics)
        eval_metrics_values_list = [best_m1, best_m1_model_large_pctg_m1_value, total_metrics_values[0], 
                                    "-", best_m1_model_large_pctg_m2_value, total_metrics_values[1]]
        add_observation_to_selection_record_file(selection_record_path, best_m1_iter_flag, best_m1_berri_selected_data_rep, 
                                                    eval_metrics_values_list)

        print("$Final [global_berri_selected_data_rep]:$")
        print(global_berri_selected_data_rep)
        

        




if __name__ == "__main__":
    parser = HfArgumentParser((Seq2SeqTrainingArguments, ExtraArguments))
    training_args, extra_args = parser.parse_args_into_dataclasses()
    sds_select(training_args, extra_args)

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
