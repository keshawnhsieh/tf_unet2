#!/usr/bin/env bash

python build_data.py \
            lcr_who_thr \
            /home/nas/sample_library/js/labeled3 \
            H51G004017 \
            --size 256 \
            --threshold 0.0 \
            --random_seed 1234

python train.py \
            --exp lcr_who_thr \
            --gpu 2