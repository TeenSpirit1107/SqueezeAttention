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

# Dynamically determine model display name for get_result_helm.py
case $model in
    *llama2*|*llama-2*)
        MODEL_DISPLAY_NAME="Llama 2"
        ;;
    *llama3*|*llama-3*)
        MODEL_DISPLAY_NAME="Llama 3"
        ;;
    *llama*)
        MODEL_DISPLAY_NAME="LLaMA"
        ;;
    *gpt-neox*|*neox*)
        MODEL_DISPLAY_NAME="GPT-NeoX"
        ;;
    *gpt*)
        MODEL_DISPLAY_NAME="GPT"
        ;;
    *)
        # Default fallback - extract base model name
        MODEL_DISPLAY_NAME=$(echo $model | sed 's/-[0-9]*[bk]*$//' | sed 's/.*\///g' | tr '[:lower:]' '[:upper:]')
        ;;
esac

echo "Using model: $model, Display name: $MODEL_DISPLAY_NAME"

# Check if input JSONL file exists
if [ ! -f "$JSONL" ]; then
    echo "Error: Input file $JSONL does not exist!"
    exit 1
fi
# pre_path=/home/user/wangzihao/kv_pruner/helm/benchmark_output/runs
pre_path=/local/ymteng/SqueezeAttention/helm/benchmark_output/runs
mkdir -p ${pre_path}/${OUTPUT}/eval_cache
# Copy eval_cache files if they exist
if [ "$(ls -A benchmark_output/runs/latest/eval_cache/ 2>/dev/null)" ]; then
    cp benchmark_output/runs/latest/eval_cache/* ${pre_path}/${OUTPUT}/eval_cache
else
    echo "No eval_cache files found in latest directory, starting with empty cache"
fi

python scripts/offline_eval/import_results.py meta ${JSONL} --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/${TASK}/run_specs_${ARCH}.conf --local-path prod_env --max-eval-instances ${sam_num} --num-train-trials=1 --suite ${OUTPUT} -n 1

helm-summarize --suite ${OUTPUT}
cd ../

python get_result_helm.py \
	--input_path ./helm/benchmark_output/runs/${OUTPUT}/groups/latex/core_scenarios_accuracy.tex \
	--output_path temp_result \
	--model_name "$MODEL_DISPLAY_NAME"