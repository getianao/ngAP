#!/bin/bash


time ${NGAP_ROOT}/scripts/run_throughput_gpu_sota_best_oneinput.sh
time ${NGAP_ROOT}/scripts/run_throughput_runahead_oneinput.sh
time ${NGAP_ROOT}/scripts/run_throughput_gpu_nap_defalut_oneinput.sh
time ${NGAP_ROOT}/scripts/run_throughput_gpu_nap_best_oneinput.sh

time ${NGAP_ROOT}/scripts/run_throughput_cpu_oneinput.sh

