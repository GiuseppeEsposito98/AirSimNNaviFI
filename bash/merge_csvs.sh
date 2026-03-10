#!/bin/bash

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
# global_PWD="/media/STORAGE/g.esposito"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}

job_id=0

python NaviAPPFI/analysis/merge_states_41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.py --fsim_log FSIM_prova_0 --target_lyr 0