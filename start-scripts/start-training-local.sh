#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/train.py \
 --device cpu --batch-size 4 --image-size 72 --pooling-layers 0 --encoding-layers 3 --data-types tas \
 --img-names 20cr-train.h5 --mask-names single_temp_mask.h5 \
 --data-root-dir /home/joe/PycharmProjects/climatereconstructionAI/data/20cr/ \
 --mask-dir /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/masks/ \
 --snapshot-dir /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/snapshots/temperature/20cr-lstm-test1/ \
 --log-dir /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/logs/temperature/20cr-lstm/ \
 --lstm-steps 0 \
 --prev-next-steps 0 \
 --out-channels 1 \
 --max-iter 1000000 \
 --save-model-interval 10000 \
 --log-interval 1000