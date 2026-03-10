# NaviAPPFI
NaviAPPFI performs Fault Injection campaigns on the learnable parameters of the MLP for the authonomous drone navigation. The MLP has been trained with Deep Q Network (DQN) reinforcement learning algorithm.

# Prerequisites
python=3.12.7
NVIDIA compatible drivers version:  ???

# Framework Setup
Install miniconda environmet if you already have it ignore this step
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### Subsection 1: Setup that requires Airsim support
1. Download the NaviSlim Repo
```bash
git clone https://github.com/WreckItTim/rl_drone.git
```

2. Follow the steps provided in the README.md file in NaviSlim repo from step 2 on..

3. Once you have successfully setup NaviSlim repo, download NaviAPPFI repo inside rl_drone repo
```bash
cd rl_drone
git clone https://github.com/GiuseppeEsposito98/NaviAPPFI.git
```

4. Inside the NaviAPPFI download the NaviSlimPytorchFI repo
```bash
cd NaviAPPFI
git clone https://github.com/GiuseppeEsposito98/NaviSlimPytorchFI.git
```

5. If you don't have the conda environment already activated, activate it and set the right repo paths with the following snippet of code

```bash
source ~/miniconda3/bin/activate airsim
cd ..
python -m pip install -e ./NaviAPPFI/NaviSlimPytorchFI/
```

### Subsection 2: Setup that does not requires Airsim support but uses the synthetic data
1. Download the zip file named iasl_environment.zip that I shared on the slack group
2. Unzip iasl_environment.zip
3. Organize the root folder using the following commands

```bash
cd iasl_environment/reinforcement_learning
mv models/ ../models
mv data/ ../data
mv utils/ ../utils
```

4. Follow the steps provided in the README.md file in NaviSlim repo from step 2 on..

5. Once you have successfully setup NaviSlim repo, download NaviAPPFI repo inside rl_drone repo
```bash
git clone https://github.com/GiuseppeEsposito98/NaviAPPFI.git
```

6. Inside the NaviAPPFI download the NaviSlimPytorchFI repo
```bash
cd NaviAPPFI
git clone https://github.com/GiuseppeEsposito98/NaviSlimPytorchFI.git
```

7. If you don't have the conda environment already activated, activate it and set the right repo paths with the following snippet of code

```bash
source ~/miniconda3/bin/activate airsim
cd ..
python -m pip install -e ./NaviAPPFI/NaviSlimPytorchFI/
```

# How to use this framework?

### Subsection 1: Simulation running with Airsim from Microsoft
1. Deactivate the base conda environmet and activate the airsim environment
2. Change in your terminal the directory to the rl_drone directory
3. Run the Fsim command

```bash
bash ./NaviAPPFI/bash/dqn.sh < log_folder_name >
```

### Subsection 2: Simulation running without Airsim from Microsoft
1. Deactivate the base conda environmet and activate the airsim environment
2. Change in your terminal the directory to the iasl_environment/reinforcement_learning directory
3. Run the Fsim command

```bash
bash ./NaviAPPFI/bash/dqn_wo_sim.sh < log_folder_name >
```

The results will be saved in the folder < log_folder_name >/test
Specifically, the < log_folder_name >  folder will contain:
1. The folder test, storing the results of the fault injection:

    - F_<fault_idx>_results folder which, in turn, contains a detailes description of the inference output for each episode and for each step. Specifically (i) F_<fault_idx>_results.json contains results for fsim outcome analysis and (ii) states__part_0.json contains the information related to only the current fault and not reporting the results of the golden run.

    - golden_states folder which only contains (i) states__part_0.json which contains information and statistics gathered form the current (golden) run and (ii) observations__part_0 is the file storing the observations across all episodes and all steps in a numpy format. 

    - ckpt_FI.json still needs to be optimized and setup but its aim is to rerun simulations when they get stuck for any reason. 

    - fault_list.csv contains the list of injected faults.

    - fsim_report.csv contains general statistics gathered from the whole simulation scattered at the fault level. 

2. model.zip which contains the weights for the Reinforcement Learning-based Navigation model
3. configuration.json file which contains the configurations of the (i) drone simulator and (2) drone experimental setup

