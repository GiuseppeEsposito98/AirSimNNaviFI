#!/bin/bash

DIR="$1"
lyr="$2"
hardening="$3"
Sim_dir=${global_PWD}/${DIR}_${hardening}/JOBID_N${lyr}/

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
# global_PWD="/media/STORAGE/g.esposito"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}

job_id=0

python3 ${global_PWD}/map_tool_box/AirSimNNaviFI/analysis/postprocess.py --fsim_log ${Sim_dir} --target_lyr ${lyr} 