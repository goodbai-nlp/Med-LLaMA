export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="/opt/conda/bin:$PATH"
source ~/.bashrc

BasePath=/baixuefeng
MODEL=${BasePath}/data/pretrained-models/llama-7b
DataPath=${BasePath}/data
DataSetName=pubmed-abs

lr=2e-5
OUTPUT_DIR=${BasePath}/output/exp.PubMedLLaMA/Pretrain-${DataSetName}-PubMedLLaMA-7b-lr-${lr}-totalbsz128

if [ ! -d ${OUTPUT_DIR} ];then
  mkdir -p ${OUTPUT_DIR}
else
  read -p "${OUTPUT_DIR} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf ${OUTPUT_DIR}; mkdir -p ${OUTPUT_DIR};;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed run.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --data_path ${DataPath}/${DataSetName} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name ${MODEL} \
    --use_fast_tokenizer False \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --evaluation_strategy "steps" \
    --logging_steps 100 \
    --save_strategy "steps" \
    --eval_steps 200 \
    --save_steps 20000 \
    --greater_is_better False \
    --save_total_limit 3 \
    --num_train_epochs 3 \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --tf32 True \
    --overwrite_output_dir \
    --data_cache_dir ${DataPath}/${DataSetName}/.cache \
    --report_to "tensorboard" 2>&1 | tee ${OUTPUT_DIR}/training.log
