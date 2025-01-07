#!/bin/bash

fullpath=$(readlink --canonicalize --no-newline $BASH_SOURCE)
cur_dir=$(cd `dirname ${fullpath}`; pwd)
# echo ${cur_dir}

export NGAP_ROOT=${cur_dir}

export PATH="${NGAP_ROOT}/code/build/bin:${PATH}"
export PATH="${NGAP_ROOT}/code/src/asyncap/bin:${PATH}"
export PATH="${NGAP_ROOT}/hscompile/build:${PATH}"


# export CUDA_VISIBLE_DEVICES=0
# sudo nvidia-smi -pm 1
# sudo nvidia-smi -i 0 -pl 200 # power limit, 3060ti: 200W, 3090: 350W 
# sudo nvidia-smi -i 0 -lgc 1695 # lock-gpu-clocks
# nvidia-smi --format=csv --query-gpu=clocks.sm,temperature.gpu,fan.speed,power.draw,power.limit -lms 100 -i 0
# sudo nvidia-smi -rgc # reset clock
# sudo systemctl isolate multi-user.target
