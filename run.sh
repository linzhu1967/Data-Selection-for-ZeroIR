beir_list=("trec-covid" "nfcorpus" "fiqa" "arguana" "webis-touche2020" "dbpedia-entity" "scidocs" "climate-fever" "scifact")
epoch=4
chunk_size=640

for beir_item in "${beir_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 nohup \
    python -m torch.distributed.launch --nproc_per_node 6 --master_port 12200 \
    train_ranker_by_stepwise_data_selection.py \
        --seed 9991 \
        --base_model "castorini/monot5-large-msmarco" \
        --target_eval_dataset_name "$beir_item" \
        --iter_number 150 \
        --search_strat "single-metric-delete-used-source" \
        --source_data_chunk $chunk_size \
        --num_train_epochs $epoch \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 160 \
        --output_dir "./saved/rankers" \
        --eval_metrics "ndcg_cut_10" \
    > ./logs/adapt_ranker_to_${beir_item}_batch8_ep${epoch}_test% 2>&1 & \
    PID=$!
    wait $PID 
done