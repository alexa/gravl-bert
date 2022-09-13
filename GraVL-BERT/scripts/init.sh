#!/usr/bin/env bash
CUDA_HOME=/usr/local/cuda-11.1
cd ./common/lib/roi_pooling/
python setup.py build_ext --inplace
cd ../../../

#cd ./refcoco/data/datasets/refer/
#make
#cd ../../../../



