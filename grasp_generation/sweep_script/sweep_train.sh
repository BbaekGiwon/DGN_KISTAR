shopt -s expand_aliases

python scripts/kistar_sweep.py \
        --n_iter 6000 \
        --batch_size_each 500 \
        --max_total_batch_size 500 \
        --object_code_list "ddg-gd_banana_poisson_002" \
        --overwrite \

