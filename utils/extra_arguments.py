from dataclasses import dataclass, field
from typing import List



@dataclass
class ExtraArguments:
    retrieval_approach: str = field(
        default="reranking",
        metadata={
            "help": "It means the exact approch used for text retrieval, such as reranking and dense_retreival."
        },
    )
    base_model: str = field(
        default="castorini/monot5-large-msmarco",
        metadata={
            "help": "The backbone of the retrieval model. Download from huggingface."
        },
    )
    source_training_pairs_dir: str = field(
        default="/BERRI/tart_full_final_train/original_json/{}_st_train_ranker_input.json",        
        metadata={
            "help": "The folder of files containing pairs of query, passage and label (positive/negative) in TSV format."
        },
    )
    target_eval_bm25topk_dir: str = field(
        default="/BEIR/bm25_top1000/",
        metadata={
            "help": "The folder of files containing pairs of query and candidate relevant passage scored by BM25 in TSV format."
        },
    )
    data_selection_record_dir: str = field(
        default="/BERRI/tart_full_final_train/exp_dataselect_performance/",
        metadata={
            "help": "The directory to record the selection process."
        },
    )
    target_eval_dataset_name: str = field(
        default="fiqa",
        metadata={
            "help": "One of the BEIR datasets to be evaluated."
        },
    )
    eval_bm25topk: int = field(
        default=1000,
        metadata={
            "help": "The number of topk passages retrieved and scored by BM25."
        },
    )
    iter_number: int = field(
        default=100,
        metadata={
            "help": "Number of iterations for BO."
        },
    )
    eval_pctg: str = field(
        default="",
        metadata={
            "help": "The percentage of test data to be evaluated. It can be 10% or 20%."
        },
    )
    search_strat: str = field(
        default="",
        metadata={
            "help": "The search strategy used for reducing the optimization space of BO. \
                For example, None, 'backtrack', 'multi-metrics', 'multi-metrics-init-pruning', \
                    'single-metric-init-pruning', 'single-metric-delete-used-source'."
        },
    )
    source_data_chunk: int = field(
        default=640,
        metadata={
            "help": "Number of data in each chunk."
        },
    )
    select_chunk_number: int = field(
        default=1,
        metadata={
            "help": "Number of chunks in each selection process."
        },
    )
    max_doc_length: int = field(
        default=0,
        # default=300,
        metadata={
            "help": "Maximum document length. Documents exceding this length will be truncated."
        },
    )
    eval_metrics: List[str] = field(
        default_factory=lambda: ["ndcg_cut_10"],
        metadata={
            "help": "The evaluation metrics to be used. These can be 'ndcg_cut_10', 'P_10', 'map', and etc."
        },
    )
    initial_pruning: bool = field(
        default=False,
        metadata={
            "help": "Whether to use initial pruning."
        },
    )