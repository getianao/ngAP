#!/bin/bash

fullpath=$(readlink --canonicalize --no-newline $BASH_SOURCE)
cur_dir=$(cd `dirname ${fullpath}`; pwd)
# echo ${cur_dir}

export NGAP_ROOT=${cur_dir}

export PATH="${NGAP_ROOT}/code/build/bin:${PATH}"
export PATH="${NGAP_ROOT}/hscompile/build:${PATH}"


