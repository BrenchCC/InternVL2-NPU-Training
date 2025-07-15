PYTHONPATH=/opt/tiger/InternVL-Train/internvl_chat
PER_DEVICE_BATCH_SIZE=1
MODEL_DIR=/mnt/bn/brench-hl-volume2/internvl2_mllm_training/base_models/InternVL2-8B
OUTPUT_DIR=/mnt/bn/brench-hl-volume2/internvl2_mllm_training/model_save_ckpts/internvl2_8b_test1
DATA_DIR=/mnt/bn/brench-hl-volume2/internvl2_mllm_training/meta_files/meta_test.json
source /usr/local/Ascend/ascend-toolkit/set_env.sh

set -x

BATCH_SIZE=128
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / ARNOLD_WORKER_GPU / ARNOLD_WORKER_NUM))

MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=${ARNOLD_WORKER_0_PORT//,/}
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 16
# number of nodes: 1
# batch size per gpu: 2
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1000
torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_port=${MASTER_PORT} \
  /opt/tiger/InternVL-Train/internvl_chat/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${MODEL_DIR} \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ${DATA_DIR} \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1000 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 2 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "/opt/tiger/InternVL-Train/internvl_chat/zero_stage3_config.json" \
  --report_to "wandb" \
  --max_dynamic_patch 16 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"