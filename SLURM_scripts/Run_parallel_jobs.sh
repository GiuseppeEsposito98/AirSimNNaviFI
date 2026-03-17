#!/bin/bash

PWD=`pwd`
export PYTHONPATH="$PWD"
global_PWD="$PWD"
DIR="$1"
hardening="$2"
# commit="$2"
# hardening="$3"

mkdir -p ${global_PWD}/${DIR}_${hardening}

input_args=(0 1 2 3 4 5 6 7)
array_size=${#input_args[@]}

for ((i=0; i<$array_size; i++)); do
    sbatch --output=${global_PWD}/${DIR}_${hardening}/lyr${input_args[$((i))]}_stdo_%A_%a.log --error=${global_PWD}/${DIR}_${hardening}/lyr${input_args[$((i))]}_stde_%A_%a.log ${global_PWD}/map_tool_box/AirSimNNaviFI/SLURM_scripts/dqn_wo_sim_nber_lyr.sbatch ${DIR} ${input_args[$((i))]} 10 ${hardening}
done 
# ${input_args[$((i))]}
