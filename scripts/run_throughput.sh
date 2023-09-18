#!/bin/bash

DIR=$(cd `dirname $0`; pwd)
cd ${DIR}

if [ -z "${CONFIGS}" ] || [ -z "${APPS}" ]; then
  echo "Either CONFIGS or APPS is empty"
  exit 1
fi

mkdir -p ../raw_results/log/


IFS=',' # Use the IFS (Internal Field Separator) variable to set the delimiter
configs_arr=(${CONFIGS})
apps_arr=(${APPS})


for config in ${configs_arr[@]}; do
    for app in ${apps_arr[@]}; do
        LOG=../raw_results/log/"exp-`date "+%Y%m%d-%H%M%S"`.log"
        ./run_experiments.sh ${app} ${config}  $@ 2>&1 | tee ${LOG}
    done
done
