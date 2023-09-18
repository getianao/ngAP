#!/bin/bash

DIR=$(cd `dirname $0`; pwd)
cd ${DIR}



FOLDER="exp-`date "+%Y%m%d-%H%M%S"`"

cd ../raw_results
if [ ! -d ${FOLDER} ]; then
    mkdir ${FOLDER} && cd ${FOLDER}
else
    cd ${FOLDER}
fi

cp ../../code/scripts/configs/* .

echo "Running Experiments... This will take several hours. "



APP_SPEC=$1
EXEC_CONFIG=$2


python ../../code/scripts/launch_exps.py -b ${APP_SPEC} -f ${EXEC_CONFIG} -e --clean ${@:3}

echo "Experiments finished. "


if [ $? -eq 0 ]; then
    echo "Collecting experiment raw data."
    python ../../code/scripts/collect_results.py -b ${APP_SPEC} -f ${EXEC_CONFIG} ${@:3}
else
    echo "Experiments terminate abnormally. "
    exit 1
fi

cd ${NGAP_ROOT}







