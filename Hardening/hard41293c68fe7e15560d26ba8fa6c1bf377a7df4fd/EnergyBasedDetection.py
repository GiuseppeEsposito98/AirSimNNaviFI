import json
import os
import utils.global_methods as gm
import map_data.map_methods as mm
from ComputeTrainStats import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

def inference_on_train_split():

    # params set from arguments passed in python call
    model_dir = 'models/navislim_release/v0/' # directory to model which has json configuration and sb3 pytorch neural networks
    output_dir = f'{model_dir}evaluations/'
    extra_out_name = ''
    results_path = f'{output_dir}{extra_out_name}'
    results_file_path = f'{results_path}evaluation__train.p'
    json_energies = f"{results_path}train_energies.json"
    # print(os.path.exists(json_energies))
    # if os.path.exists(results_file_path) and os.path.exists(json_energies):
    #     # results = pickle.load(open(results_file_path, 'rb'))
    #     # stats_per_ep = json.load(open(f"{json_energies}", 'r'))
    #     pass
    # else:
    req_stat = ['Energy']
    train_configuration = setup_inference_on_train()
    stats_per_ep = inference(train_configuration=train_configuration, req_stat=req_stat)
    with open(f"{json_energies}", "w", encoding="utf-8") as f:
        json.dump(stats_per_ep, f, indent=4, ensure_ascii=False)
    
    unr_stats = list()
    for ep in stats_per_ep:
        unr_stats.append(stats_per_ep[ep]['Energy'])
    print(stats_per_ep)
    fig, ax = plt.subplots(1,1,figsize=(10,7))
    sns.kdeplot(data=unr_stats, ax = ax)

    perc75, perc90, perc95 = np.percentile(unr_stats, [75, 90, 95])

    print(f'75th percentile: {perc75}')
    print(f'90th percentile: {perc90}')
    print(f'95th percentile: {perc95}')

    ax.vlines([perc75, perc90, perc95], ymin=0, ymax=0.5, colors=['red', 'green', 'blue'])

    fig.savefig(f"{results_path}train_energies.png")


def main():
    inference_on_train_split()

if __name__ == '__main__':
    main()