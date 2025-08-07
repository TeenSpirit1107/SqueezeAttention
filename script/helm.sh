# formatted=$(printf "%g" $i)  # Commented out as $i is not defined
cd helm
rm -rf prod_env/cache/*
sam_num=300
model=llama2-7b-32k
ini_size=0.4
KV_class3=0.25
TASK=xsum
path_percent=$(echo "scale=0; ($KV_class3 * 100)/1" | bc)
JSONL=../pred_${TASK}/${model}/${ini_size}/${path_percent}/${TASK}.jsonl
OUTPUT=${TASK}_${model}_${ini_size}_${path_percent}_result
ARCH=llama

# Check if input JSONL file exists
if [ ! -f "$JSONL" ]; then
    echo "Error: Input file $JSONL does not exist!"
    exit 1
fi
# pre_path=/home/user/wangzihao/kv_pruner/helm/benchmark_output/runs
pre_path=/local/ymteng/SqueezeAttention/helm/benchmark_output/runs
mkdir ${pre_path}/${OUTPUT}
mkdir ${pre_path}/${OUTPUT}/eval_cache
cp benchmark_output/runs/latest/eval_cache/* ${pre_path}/${OUTPUT}/eval_cache

python scripts/offline_eval/import_results.py together ${JSONL} --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/${TASK}/run_specs_${ARCH}.conf --local-path . --max-eval-instances ${sam_num} --num-train-trials=1 --suite ${OUTPUT} -n 1

helm-summarize --suite ${OUTPUT}
cd ../

python get_result_helm.py \
	--input_path ./helm/benchmark_output/runs/${OUTPUT}/groups/latex/core_scenarios_accuracy.tex \
	--output_path temp_result \
	--model GPT-NeoX \