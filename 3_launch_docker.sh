#!/bin/bash

docker run -it --rm --gpus all -v ${NGAP_ROOT}:/ngAP ngap-ae:latest /bin/bash