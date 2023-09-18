#!/bin/bash

DIR=$(cd `dirname $0`; pwd)
cd ${DIR}

VALI=--validation

mkdir -p ./results/raw/throughput_gpu_nap_default_adp

# config: [NAP-default], apps: [part 1, part 2]
APPS="app_spec_ngap_new_quickvalidation_part1" \
CONFIGS="exec_config_ngap_groups_nap_default" \
./run_throughput.sh --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 --csvdest=./results/raw/throughput_gpu_nap_default_adp/throughput_gpu_nap_default_part1.csv

APPS="app_spec_ngap_new_quickvalidation_part2" \
CONFIGS="exec_config_ngap_groups_nap_default" \
./run_throughput.sh --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 --csvdest=./results/raw/throughput_gpu_nap_default_adp/throughput_gpu_nap_default_part2.csv

# config: [NAP-default'], apps: [part 3]
APPS="app_spec_ngap_new_quickvalidation_part3" \
CONFIGS="exec_config_ngap_groups_nap_default_4degree" \
./run_throughput.sh --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 --csvdest=./results/raw/throughput_gpu_nap_default_adp/throughput_gpu_nap_default_part3.csv