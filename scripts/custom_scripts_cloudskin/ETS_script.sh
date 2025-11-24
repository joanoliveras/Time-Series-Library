model_name=ETS
pred_len=10
seq_len=2880

python -u run.py \
  --output_path /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/30m_inference/results \
  --task_name ets \
  --is_training 0 \
  --root_path /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/30m_inference \
  --data_path preprocessed_data.csv \
  --data_iterate True \
  --model $model_name \
  --model_id ets \
  --data ets \
  --target pipelines_status_realtime_pipeline_latency \
  --pred_len $pred_len \
  --seq_len $seq_len \
  --c_out 1