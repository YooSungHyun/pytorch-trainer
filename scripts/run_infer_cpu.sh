python3 cpu_infer.py \
    --seed=42 \
    --data_path=./raw_data/pollution_test_data1.csv \
    --model_path=./cpu_outputs/epoch-0050.ckpt \
    --metric_on_cpu=True