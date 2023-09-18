#!/bin/bash

time ${NGAP_ROOT}/scripts/run_throughput_gpu_sota_best.sh # 3.5 hrs
time ${NGAP_ROOT}/scripts/run_throughput_runahead.sh # 3.5 hrs
time ${NGAP_ROOT}/scripts/run_throughput_gpu_nap_defalut.sh # 1 hrs
time ${NGAP_ROOT}/scripts/run_throughput_gpu_nap_best.sh # 1 hrs

time ${NGAP_ROOT}/scripts/run_throughput_cpu.sh  # 5.5 hrs

