#!/bin/bash

DIR=$(cd `dirname $0`; pwd)
cd ${DIR}

# VALI=--validation


mkdir -p ./results/raw/throughput_gpu_runahead_oneinput

# config: [runahead], apps: [part 1]
APPS="app_spec_ngap_new_quickvalidation_part1" \
CONFIGS="exec_config_ngap_groups_design_sota_runahead_oneinput" \
./run_throughput.sh  --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 \
--csvdest=./results/raw/throughput_gpu_runahead_oneinput/throughput_gpu_runahead_oneinput_part1.csv

# config: [runahead], apps: [part 2]
APPS="app_spec_ngap_new_quickvalidation_part2" \
CONFIGS="exec_config_ngap_groups_design_sota_runahead_oneinput" \
./run_throughput.sh  --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 \
--csvdest=./results/raw/throughput_gpu_runahead_oneinput/throughput_gpu_runahead_oneinput_part2.csv


# config: [runahead'], apps: [part 3]
APPS="app_spec_ngap_new_quickvalidation_part3" \
CONFIGS="exec_config_ngap_groups_design_sota_runahead_4degree_oneinput" \
./run_throughput.sh  --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI} --timeout-mins=60 \
--csvdest=./results/raw/throughput_gpu_runahead_oneinput/throughput_gpu_runahead_oneinput_part3.csv
