#!/bin/bash

DIR=$(cd `dirname $0`; pwd)
cd ${DIR}

VALI=--validation

mkdir -p ./results/raw/throughput_gpu_nap_best_oneinput

# config: [NAP-Best], apps: [part 1, part 2]
APPS="app_spec_ngap_new_quickvalidation_part1" \
CONFIGS="exec_config_ngap_groups_best_oneinput" \
./run_throughput.sh --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 --csvdest=./results/raw/throughput_gpu_nap_best_oneinput/throughput_gpu_napbest_oneinput_part1.csv

APPS="app_spec_ngap_new_quickvalidation_part2" \
CONFIGS="exec_config_ngap_groups_best_oneinput" \
./run_throughput.sh --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 --csvdest=./results/raw/throughput_gpu_nap_best_oneinput/throughput_gpu_napbest_oneinput_part2.csv

# config: [NAP-Best'], apps: [part 3]
APPS="app_spec_ngap_new_quickvalidation_part3" \
CONFIGS="exec_config_ngap_groups_best_4degree_oneinput" \
./run_throughput.sh --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 --csvdest=./results/raw/throughput_gpu_nap_best_oneinput/throughput_gpu_napbest_oneinput_part3.csv