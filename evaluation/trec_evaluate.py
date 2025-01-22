import subprocess



def run_trec_eval(run_file, qrels_file, relevance_threshold=1, remove_unjudged=False):
    args = [
        "python3",
        "-m",
        "pyserini.eval.trec_eval",
        "-c",
        f"-l {relevance_threshold}",
        "-m" , "all_trec",
        "-m", "judged.10",
        "-m", "ndcg_cut.10",
    ]

    if remove_unjudged:
        args.append("-remove-unjudged")
    args += [qrels_file, run_file]

    result = subprocess.run(args, stdout=subprocess.PIPE)
    # print(result.stdout.decode("utf-8"))
    metrics = {}
    for line in result.stdout.decode("utf-8").split("\n"):
        for metric in [
            "recip_rank",
            "recall_1000",
            "num_q",
            "num_ret",
            "ndcg_cut_10",
            "ndcg_cut_20",
            "map",
            "P_10",
            "judged_10",
        ]:
            # the space is to avoid getting metrics such as ndcg_cut_100 instead of ndcg_cut_10 as but start with ndcg_cut_10
            if line.startswith(metric + " ") or line.startswith(metric + "\t"):
                metrics[metric] = float(line.split("\t")[-1])
    print("metrics:")
    print(metrics)
    return metrics
