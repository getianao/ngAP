#!/bin/bash

DIR=$(cd `dirname $0`; pwd)
cd ${DIR}

# VALI=--validation


# exp_pipeline-20230402-221741


mkdir -p ../results/raw/ncu/l1cache
mkdir -p ../results/raw/ncu/memory

# config: [aasinterval], apps: [part 1, part 2]
APPS="app_spec_ngap_new_quickvalidation_part1" \
CONFIGS="exec_config_table3_pipeline_groups_design_NAP_ncu" \
./run_throughput.sh \
--ncu \
--keywords=../../gpunfa_code/scripts/collect_keyword_list_ncu.txt \
${VALI}  --timeout-mins=60  \
--csvdest=-part1.csv

APPS="app_spec_ngap_new_quickvalidation_part2" \
CONFIGS="exec_config_table3_pipeline_groups_design_NAP_ncu" \
./run_throughput.sh \
--ncu \
--keywords=../../gpunfa_code/scripts/collect_keyword_list_ncu.txt \
${VALI}  --timeout-mins=60  \
--csvdest=-part2.csv