#!/usr/bin/env bash

for (( c=10; c<=34; c++ ))
do
    python predict.py \
                test/H51G004017.tif \
                lcr_who_thr/models/model-$c.ckpt \
                --gpu 1;
done
