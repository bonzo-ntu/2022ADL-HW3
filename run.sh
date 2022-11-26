python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file ./data/train.jsonl\
    --validation_file ./data/public.jsonl \
    --source_prefix "summarize: " \
    --output_dir ./result/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --gradient_accumulation_steps=4 \
    --eval_accumulation_steps=4 \
    --adafactor \
    --learning_rate 1e-3 \
    --warmup_ratio 0.1 \