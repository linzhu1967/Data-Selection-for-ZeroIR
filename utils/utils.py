import random
import numpy as np
import torch
import os
import shutil
import subprocess



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
def distribute_eggs(eggs, baskets):
    if baskets == 1:
        return [[eggs]]
    else:
        distributions = []
        for eggs_in_basket in range(eggs + 1):
            for distribution in distribute_eggs(eggs - eggs_in_basket, baskets - 1):
                distributions.append([eggs_in_basket] + distribution)
        return distributions
    
    
def delete_model_folder(folder_path: str):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")
            
            
def find_differences_numpy(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    differences = np.where(arr1 != arr2)[0]
    
    return differences


def custom_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def merge_jsonl_files(file_paths, output_file_path):
    cat_command = ['cat'] + file_paths
    with open(output_file_path, 'w') as outfile:
        subprocess.run(cat_command, stdout=outfile)
        

def delete_jsonl_files(file_paths):
    for file_path in file_paths:
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")