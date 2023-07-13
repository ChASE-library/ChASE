#!/bin/bash

folder="build-nccl"
NCCL_FLAG=ON

if [ -d "$folder" ]
then
    rm -rf $folder
fi

cmake -B $folder -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DBUILD_WITH_EXAMPLES=ON -DCHASE_OUTPUT=ON -DENABLE_NCCL=${NCCL_FLAG}
cmake --build $folder -j
