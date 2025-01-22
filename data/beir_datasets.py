from torch.utils.data import Dataset

from datasets import load_dataset
from functools import partial
import pandas as pd
import csv

from data.load_datasets import load_beir_queries, load_beir_corpus





class BeirBM25TopkMonoT5Dataset(Dataset):
    def __init__(self, input_topk_file, dataset_name, topk, tokenizer, max_length, corpus, queries):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length if max_length==0 else max_length
        self.prompt = "Query: {query} Document: {text} Relevant:"
        self.outputs = ['false', 'true']
        self.data = self.read_data(input_topk_file, dataset_name, corpus, queries, topk)
        
    
    def read_data(self, input_topk_file, dataset_name, corpus, queries, topk=1000):
        # corpus = load_beir_corpus(dataset_name)
        # corpus = dict(zip(corpus['doc_id'], corpus['text']))
        # queries = load_beir_queries(dataset_name)
        
        topk_df = pd.read_csv(
            input_topk_file,
            sep=r"\s+",
            quoting=csv.QUOTE_NONE, 
            keep_default_na=False,
            names=("qid", "_1", "docid", "rank", "score", "ranker"),
            dtype=str,
        )
        # Handling strings containing double quotes
        if dataset_name == "climate-fever":
            # Remove the quotation marks before and after
            topk_df.loc[topk_df['docid'].str.contains('\"'), 'docid'] = topk_df['docid'].str[1:-1]
            # Replace double quotes with single quotes
            topk_df.loc[topk_df['docid'].str.contains('\"'), 'docid'] = topk_df['docid'].str.replace('\"\"', '\"', regex=False)
            
        # # for debug, only use the first query
        # topk_df = topk_df[topk_df["qid"] == list(topk_df["qid"])[0]]
        # Converts run to float32 and subtracts a large number to ensure the BM25 scores are lower than those provided by the neural ranker.
        topk_df["score"] = topk_df["score"].astype("float32").apply(lambda x: x-10000)
        # Reranks only the top-k documents for each query
        subset = topk_df[["qid", "docid"]].groupby("qid").head(topk).apply(lambda x: [x["qid"], queries[x["qid"]], x["docid"], corpus[x["docid"]]], axis=1)        
        data = subset.values.tolist()
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        prompts = self.prompt.format(query=data_item[1], text=data_item[3])
        tokenized = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        
        return {
            "qid": data_item[0],
            "docid": data_item[2],
            "input_ids": tokenized['input_ids'].squeeze(),
            "attention_mask": tokenized['attention_mask'].squeeze(),
        }