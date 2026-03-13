#!/bin/bash

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
# global_PWD="/media/STORAGE/g.esposito"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}

job_id=0

python map_tool_box/AirSimNNaviFI/bash/postprocess.py --fsim_log FSIM --target_lyr 0