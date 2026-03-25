# NaviAPPFI
NaviAPPFI performs Fault Injection campaigns on the learnable parameters of the MLP for the authonomous drone navigation. The MLP has been trained with Deep Q Network (DQN) reinforcement learning algorithm.

# Prerequisites
python=3.12.7

# Framework Setup
Install miniconda environmet if you already have it ignore this step
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### Subsection 1: Setup the repositories tree
1. Clone the Repository for the Airsim-based autonomous drone simulator and swith to the compatible commit
```bash
git clone https://github.com/WreckItTim/map_tool_box.git
git switch  --detach 34fa84a8bd309587f59b561cd472d907dc4fa601
```

2. Create a python environment (you can use also anaconda or miniconda) with python version 3.12.7 and activate it
```bash
conda create -n airnnfi python==3.12.7
conda activate airnnfi
```

3. Clone NaviAPPFI repo inside rl_drone repo
```bash
cd map_tool_box
git clone https://github.com/GiuseppeEsposito98/AirSimNNaviFI.git
```

4. Install the packages with the env_setup.bat file, provided in the AirSimNNaviFI folder. It is strictly required to do that through the bat file to meet the library dependences constraints. If you are using windows the bat file is an executable itself so you need to type the file path on the Command Line Interface (CLI) and run it. For Linux users, you can execute the bat file as it is a bash file.
```bash
bash AirSimNNaviFI/env_setup.bat
```

4. Inside the NaviAPPFI clone the NaviSlimPytorchFI repo
```bash
cd AirSimNNaviFI
git clone https://github.com/GiuseppeEsposito98/NaviSlimPytorchFI.git
cd ..
```

5. If you don't have the conda environment already activated, activate it and set the right repo paths with the following snippet of code

```bash
python -m pip install -e ./AirSimNNaviFI/NaviSlimPytorchFI/
```

### Subsection 2: Download the dataset and Navigation Neural Network weights
1. According to what is described in map_tool_box repo, download the dataset available at zip files data/maps/AirSimNH/sensors/DepthV1.zip of this dropbox: 
    https://www.dropbox.com/scl/fo/vg3t52glaj0yqk9njlc4d/AB17lboB6pdP84wh-tt7OkI?rlkey=ra6u86nrj28kea1dh84emw6hy&st=vagz7xrp&dl=0 
2. Download the (Neural Network) NN weights and setup files available in models folder of the same Dropbox. 
3. Sort the items within the correct directories: 
    - The dataset DepthV1.zip goes in the downloaded repo folder maps/AirSimNH/sensors/ (if sensors parent folder is not available, you need to create it). 
    - For the downloaded files for the NN, you can just drag and drop the models folder (downloaded from Dropbox) in the main repo directory (map_tool_box). 
4. Unzip the DepthV1.zip 
5. Remove the file file_map.p that you may find in DepthV1 unzipped folder 


# How to use this framework?

### Subsection 1: Run in local

1. Activate the conda environment
```bash
conda activate airnnfi
```
2. Run the command 
```bash
bash map_tool_box/AirSimNNaviFI/bash/dqn_wo_sim_nber_lyr.sh FSIM 0 10 base
```
To run the fault injection campaign:
- Saving the data in FSIM folder
- Targeting the first layer of the Neural Network
- Executing 10 trial per injected fault

### Subsection 2: Run on a SLURM-based HPC system

1. Activate the conda environment
```bash
conda activate airnnfi
```
3. Customize the sbatch script at map_tool_box/AirSimNNaviFI/SLURM_scripts/dqn_wo_sim_nber_lyr.sbatch, based on your system requirements

2. Run the command 
```bash
bash map_tool_box/AirSimNNaviFI/SLURM_scripts/Run_parallel_jobs.sh FSIM base
```
To launch 8 jobs (1 per Neural Network layer) and save the data on FSIM folder.
