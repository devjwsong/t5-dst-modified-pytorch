python src/modified_train.py \
    --seed=0 \
    --data_dir="data" \
    --cached_dir="cached" \
    --data_name="multiwoz_zeroshot" \
    --trg_domain=TRG_DOMAIN \
    --model_name="t5-small" \
    --train_prefix="train" \
    --valid_prefix="valid" \
    --test_prefix="test" \
    --slot_descs_prefix="slot_descs" \
    --num_epochs=5 \
    --train_batch_size=32 \
    --eval_batch_size=16 \
    --num_workers=4 \
    --src_max_len=512 \
    --trg_max_len=64 \
    --max_extras=5 \
    --learning_rate=1e-4 \
    --warmup_ratio=0.0 \
    --max_grad_norm=1.0 \
    --min_delta=1e-4 \
    --patience=3 \
    --sep_token="<sep>" \
    --gpu="0" \
    --log_dir="./"