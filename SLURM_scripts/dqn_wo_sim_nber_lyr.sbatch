#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=ResultPostprocessing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giuseppe.esposito@polito.it


source ~/miniconda3/bin/activate
conda deactivate

conda activate airsimnew

cd  ~/iasl_experimental/experimental/reinforcement_learning

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}

job_id=0

DIR="$1"
lyr="$2"
trials="$3"
# commit="41293c68fe7e15560d26ba8fa6c1bf377a7df4fd"

Sim_dir=${global_PWD}/${DIR}/JOBID${job_id}_W${lyr}/

mkdir -p ${Sim_dir}

cp ../models/navislim_release/v0/model.zip ${Sim_dir}
cp ../models/navislim_release/v0/configuration.json ${Sim_dir}
cp ./NaviAPPFI/Fault_simulations/FI_config.json ${Sim_dir}

sed -i -E "s/(\"neurons_rand_single_layer\"\s*:\s*\{[^}]*?\"layer\"\s*:\s*)[0-9]+/\1$lyr/" ${Sim_dir}FI_config.json
sed -i -E "s/(\"neurons_rand_single_layer\"\s*:\s*\{[^}]*?\"trials\"\s*:\s*)[0-9]+/\1$trials/" ${Sim_dir}FI_config.json

python NaviAPPFI/Fault_simulations/dqn_NBER_lyr.py \
    --fsim_config ${Sim_dir}FI_config.json \
    --target_layer $lyr \
    --trials $trials \
    --fsim_log_name ${Sim_dir} \
    # --hardening Ranger

# python NaviAPPFI/analysis/merge_states_41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.py --fsim_log ${global_PWD}/${DIR}

echo
echo "All done. Checking results:"