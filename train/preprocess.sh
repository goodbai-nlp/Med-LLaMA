export PATH="/root/micromamba/bin:$PATH"
source ~/.bashrc

BasePath=/baixuefeng
ModelCate=llama-7b
MODEL=${BasePath}/data/pretrained-models/${ModelCate}

DataPath=${BasePath}/data
DataSetName=pubmed-abs

DataPath=${BasePath}/data/TaskData
DataSetName=debug-pubmedqa

export HF_DATASETS_CACHE=${DataPath}/${DataSetName}/.cache

lr=2e-5

OUTPUT_DIR=${BasePath}/output/exp.MedLLaMA/Preprocess-${DataSetName}-${ModelCate}

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


python run.py \
    --data_path ${DataPath}/${DataSetName} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name ${MODEL} \
    --use_fast_tokenizer False \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --evaluation_strategy "steps" \
    --logging_steps 100 \
    --greater_is_better False \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10 \
    --num_train_epochs 3 \
    --logging_first_step True \
    --gradient_checkpointing \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --tf32 True \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 10 \
    --data_cache_dir ${DataPath}/${DataSetName}/.cache \
    --report_to "tensorboard" 2>&1 | tee ${OUTPUT_DIR}/training.log
