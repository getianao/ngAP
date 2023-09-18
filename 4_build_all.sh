#!/bin/bash

# GPU Schemes
cd ${NGAP_ROOT}/code && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

# CPU Schemes
cd ${NGAP_ROOT}/hscompile/lib/hyperscan && mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc .. 
make -j
cd ${NGAP_ROOT}/hscompile/lib/mnrl/C++
sed -i 's/CC = .*/CC = g++-5/g' Makefile     # requires GCC-5.
make                                         # If an error occurs, try to run it again  
cd ${NGAP_ROOT}/hscompile && mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc       \
    -DHS_SOURCE_DIR=${NGAP_ROOT}/hscompile/lib/hyperscan    \
    -DMNRL_SOURCE_DIR=${NGAP_ROOT}/hscompile/lib/mnrl/C++   \
    ..
make -j