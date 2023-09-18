#!/bin/bash

DIR=$(cd `dirname $0`; pwd)
cd ${DIR}

# VALI=--validation

mkdir -p ../results/raw/throughput_cpu_oneinput

# config: [cpu], apps: [part 1]
APPS="app_spec_ngap_new_quickvalidation_part1" \
CONFIGS="exec_config_ngap_groups_design_cpu_oneinput" \
./run_throughput.sh  --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 \
--csvdest=./results/raw/throughput_cpu_oneinput/throughput_cpu_part1.csv

# config: [cpu], apps: [part 2]
APPS="app_spec_ngap_new_quickvalidation_part2" \
CONFIGS="exec_config_ngap_groups_design_cpu_oneinput" \
./run_throughput.sh  --keywords=../../code/scripts/collect_keyword_list_throughput.txt \
${VALI}  --timeout-mins=60 \
--csvdest=./results/raw/throughput_cpu_oneinput/throughput_cpu_part2.csv