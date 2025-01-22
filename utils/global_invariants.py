PREDICTION_TOKENS= {
    'monot5-large-msmarco': ['▁false', '▁true'],
    # 'castorini/monot5-small-msmarco-10k':   ['▁false', '▁true'],
    # 'castorini/monot5-small-msmarco-100k':  ['▁false', '▁true'],
    'monot5-base-msmarco':  ['▁false', '▁true'],
    # 'castorini/monot5-base-msmarco-10k':    ['▁false', '▁true'],
    # 'castorini/monot5-large-msmarco':       ['▁false', '▁true'],
    # 'castorini/monot5-large-msmarco-10k':   ['▁false', '▁true'],
    # 'castorini/monot5-base-med-msmarco':    ['▁false', '▁true'],
    # 'castorini/monot5-3b-med-msmarco':      ['▁false', '▁true'],
    'monot5-3b-msmarco-10k':      ['▁false', '▁true'],
    # 'castorini/monot5-3b-msmarco':          ['▁false', '▁true'],
    # 'unicamp-dl/mt5-base-en-msmarco':       ['▁no'   , '▁yes'],
    # 'unicamp-dl/mt5-base-mmarco-v2':        ['▁no'   , '▁yes'],
    # 'unicamp-dl/mt5-base-mmarco-v1':        ['▁no'   , '▁yes'],
}

BERRI_IDX_DICT = {
    0: "0_agnews",
    1: "1_altlex",
    2: "2_cnn_dailymail",
    3: "3_coco_captions",
    4: "4_eli5_question_answer",
    5: "5_fever",
    6: "6_gigaword",
    7: "7_hotpotqa",
    8: "8_mdmcqa",
    9: "9_medical_sim",
    10: "10_npr",
    11: "11_nq",
    12: "12_oqa",
    13: "13_pubmedqa",
    14: "14_qrecc",
    15: "15_quora",
    16: "16_record",
    17: "17_scitldr",
    18: "18_searchQA_top5_snippets",
    19: "19_sentence-compression",
    20: "20_squad_pairs",
    21: "21_stackexchange_duplicate_questions_title-body_title-body",
    22: "22_stackexchange_duplicate_questions_title_title",
    23: "23_triviaqa",
    24: "24_wikihow",
    25: "25_wow",
    26: "26_xsum-multilexsum",
    27: "27_nan",
}

BEIR_TEST_PCTG_DICT = {
    "trec-covid":           "20%",               
    "nfcorpus":             "20%",                 
    "fiqa":                 "10%",                     
    "arguana":              "10%",                  
    "webis-touche2020":     "20%",         
    "dbpedia-entity":       "20%",           
    "scidocs":              "10%",                  
    "climate-fever":        "10%",            
    "scifact":              "20%",                  
}

TEST_PCTGS_DICT = {
    "10%":                  ["10%", "90%", "100%"],
    "20%":                  ["20%", "80%", "100%"]
}
TEST_SMALL_PCTGS =          ["10%", "20%"]
TEST_LARGE_PCTG =           ["90%", "80%"]
TEST_TOTAL_PCTG =           ["100%"]

TEST_METRICS_DECIMAL = 4

TEST_METRICS_SMALL_PCTGS =  ["ndcg@10", "ndcg@10.20%", "ndcg_cut_10.10%", "ndcg_cut_10.20%"]