import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from functools import partial






class BerriPairMonoT5Dataset(Dataset):
    def __init__(self, datapath, tokenizer, max_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length if max_length==0 else max_length
        self.prompt = "Query: {query} Document: {text} Relevant:"
        self.outputs = ['false', 'true']
        self.data = self.read_data(datapath)
    
    def read_data(self, datapath):
        data = load_dataset(
            'json', 
            data_files=datapath,
            )["train"]
        
        def pack_pairs(pairs, prompt_template, token_false, token_true):
            examples = {
                'label': [],
                'text': [],
            }
            for i in range(len(pairs['query'])):
                examples['text'].append(prompt_template.format(query=pairs["query"][i], text=pairs["document"][i]))
                examples['label'].append([token_false, token_true][int(pairs["label"][i])])

            return examples

        if "text" in data.features.keys():
            pass
        else:
            data = data.map(
                partial(pack_pairs, prompt_template=self.prompt, token_false=self.outputs[0], token_true=self.outputs[1]),
                remove_columns=('query', 'document', 'label'),
                batched=True,
            )
            
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        tokenized = self.tokenizer(
            text=data_item['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized['input_ids'].squeeze(), 
            "attention_mask": tokenized['attention_mask'].squeeze(),
            "labels": torch.tensor(self.tokenizer(data_item['label'])['input_ids'], dtype=torch.long),   
        }
    

        