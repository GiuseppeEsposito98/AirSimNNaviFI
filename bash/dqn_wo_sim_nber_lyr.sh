#!/bin/bash

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}

job_id=0
hardening="base"

DIR="$1"
lyr="$2"
trials="$3"
hardening="$4"
# commit="newest"
echo "hardening: ${hardening}"

Sim_dir=${global_PWD}/${DIR}_${hardening}/JOBID_N${lyr}/

mkdir -p ${Sim_dir}
cd ${Sim_dir}
cp -r ${global_PWD}/map_tool_box/models/AirSim_Navigation/DRL_beta ${Sim_dir}
cp ${global_PWD}/map_tool_box/AirSimNNaviFI/Fault_simulations/FI_config.json ${Sim_dir}

sed -i -E "s/(\"neurons_rand_single_layer\"\s*:\s*\{[^}]*?\"layer\"\s*:\s*)[0-9]+/\1$lyr/" ${Sim_dir}FI_config.json
sed -i -E "s/(\"neurons_rand_single_layer\"\s*:\s*\{[^}]*?\"trials\"\s*:\s*)[0-9]+/\1$trials/" ${Sim_dir}FI_config.json

python3 ${global_PWD}/map_tool_box/AirSimNNaviFI/Fault_simulations/dqn_NBER_lyr.py \
    --fsim_config ${Sim_dir}FI_config.json \
    --target_layer $lyr \
    --trials $trials \
    --fsim_log_name ${Sim_dir} \
    --hardening ${hardening}


python3 ${global_PWD}/map_tool_box/AirSimNNaviFI/analysis/postprocess.py --fsim_log ${Sim_dir} --target_lyr ${lyr} 