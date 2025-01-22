# Data-Selection-for-ZeroIR
This is the official code for the SDS method for data selection in ZeroIR.

## SDS-ranker
We finetune the SDS-ranker on the subset of source data selected by our SDS method. The SDS-ranker is built on the monoT5 large model (see https://huggingface.co/castorini/monot5-large-msmarco).

## Source Datasets
- We use BERRI, a collection of retrieval datasets from various tasks, as source datasets. The official BERRI can be found at https://github.com/facebookresearch/tart?tab=readme-ov-file#dataset-berri.
- Since the SDS-ranker belongs to the cross-encoder architecture, we use the TART-full training data of BERRI.
- According to the instruction texts, we classify each instance into its original dataset.
- 28 datasets are correctly classified, including AGNews, Altlex, CNN Daily Mail, etc. (see )
- MS MARCO dataset is removed because it was used in the pretraining of monoT5, the initialization of our model.

## Target Datasets
- We use 9 publicly available BEIR datasets. The BEIR can be found at https://github.com/beir-cellar/beir. 
