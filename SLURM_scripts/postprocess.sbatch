#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu
#SBATCH --ntasks-per-node=1
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
commit="41293c68fe7e15560d26ba8fa6c1bf377a7df4fd"

Sim_dir=${global_PWD}/${DIR}

python ${global_PWD}/map_tool_box/AirSimNNaviFI/bash/postprocess.py --fsim_log ${Sim_dir} --target_lyr ${lyr}


echo
echo "All done. Checking results:"