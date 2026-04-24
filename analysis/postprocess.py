import pandas as pd
import os
import math
from copy import deepcopy
from itertools import zip_longest
from tqdm import tqdm
import csv
import pickle
import numpy as np
import argparse
import json

def get_argparse():
    parser = argparse.ArgumentParser(description='DQN configuration')
    parser.add_argument('--fsim_log', help='Folder path to process')
    parser.add_argument('--target_lyr', help='Target layer')
    return parser

action_list = [
    'RotateClockwise',
    'RotateCounter',
    'StrafeRight1',
    'StrafeRight2',
    'StrafeRight4',
    'StrafeRight8',
    'StrafeRight16',
    'StrafeRight32',
    'StrafeLeft1',
    'StrafeLeft2',
    'StrafeLeft4',
    'StrafeLeft8',
    'StrafeLeft16',
    'StrafeLeft32',
    'Forward1',
    'Forward2',
    'Forward4',
    'Forward8',
    'Forward16',
    'Forward32',
]

def pk_read(path):
	return pickle.load(open(path, 'rb'))

def euclidean(p1, p2):
    return np.sqrt( 
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2 +
        (p1[2] - p2[2])**2
    )

def manhattan(p1, p2):
    return (
        abs(p1[0] - p2[0]) +
        abs(p1[1] - p2[1]) +
        abs(p1[2] - p2[2])
    )

def energy_from_buffer(buffer):

    buf = np.asarray(buffer).reshape(4,3)
    p0 = buf[2]
    p1 = buf[1]
    p2 = buf[0]
    dp1 = p1 - p0
    dp2 = p2 - p1

    E = np.dot(dp1, dp1) + np.dot(dp2, dp2)
    return E

def cross_product_2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def has_turn(a, b, c):
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    cross = cross_product_2d(v1, v2)
    return cross != 0

def step_ratio(A, B, bounds):
    x0, y0 = A
    x1, y1 = B
    xmin, xmax, ymin, ymax = bounds

    dx = x1 - x0
    dy = y1 - y0

    step_size = math.hypot(dx, dy)
    
    if step_size == 0:
        return 0.0

    # Direzione normalizzata
    ux = dx / step_size
    uy = dy / step_size

    t_values = []

    if ux != 0:
        t1 = (xmin - x0) / ux
        t2 = (xmax - x0) / ux
        t_values.extend([t1, t2])
    if uy != 0:
        t3 = (ymin - y0) / uy
        t4 = (ymax - y0) / uy
        t_values.extend([t3, t4])

    t_pos = [t for t in t_values if t > 0]
    max_t = float('inf')

    for t in t_pos:
        x = x0 + t * ux
        y = y0 + t * uy
        if xmin <= x <= xmax and ymin <= y <= ymax:
            max_t = min(max_t, t)

    max_step_size = max_t if max_t != float('inf') else 0.0

    ratio = (step_size / max_step_size) * 100 if max_step_size > 0 else 0.0
    # if ratio > 0.0:
    #     print(step_size)
    return ratio

# def show_paths(golden_scenario, faulty_scenario, fault_id, ep_id, fsim_log):
#     # print(len(golden_scenario))
#     golden_eval_path, _, _, _, _ = golden_scenario
#     faulty_eval_path, _, _, _, _ = faulty_scenario

#     golden_eval_path = np.array(golden_eval_path, dtype=np.int16)
#     faulty_eval_path = np.array(faulty_eval_path, dtype=np.int16)

#     airsim_map = 'AirSimNH'
#     sensor_names = [
#         'DepthV1',
#     ]
#     resolution= [144,256] # height, width of observation arrays (images)

#     datamap = mm.DataMap(airsim_map)

#     _, golden_path_animation = datamap.data_at_path(sensor_names, golden_eval_path, 
#                     make_animation=True, return_data=True, ncols=2, resolution=resolution, 
#                     sensor_psuedonames={'DepthV1':'Depth Sensor'}, include_nulls=False)
#     _, faulty_path_animation = datamap.data_at_path(sensor_names, faulty_eval_path, 
#                     make_animation=True, return_data=True, ncols=2, resolution=resolution, 
#                     sensor_psuedonames={'DepthV1':'Depth Sensor'}, include_nulls=False)

#     # save as gif
#     # if not os.path.exists(f'map_tool_box/AirSimNNaviFI/analysis/{fsim_log}/plots'):
#     #     os.path.mkdir(f'map_tool_box/AirSimNNaviFI/analysis/{fsim_log}/plots')
#     # if not os.path.exists(f'{fsim_log}/plots/golden_ep{ep_id}.gif'):
#     golden_path_animation.save(f'{fsim_log}/plots/golden_ep{ep_id}.gif', writer='imagemagick', fps=2)
#     faulty_path_animation.save(f'{fsim_log}/plots/fault{fault_id}_ep{ep_id}.gif', writer='imagemagick', fps=2)

def evaluate_path(scenario):
    # print(type(scenario[0]['point']))
    # scenario_path = Point(scenario[0]['point'])
    positions = [tuple([scenario[0]['point'].x, scenario[0]['point'].y, scenario[0]['point'].z]) for step in scenario]
    
    turns_original = 0
    turns_pivot = 0
    total_dx = 0
    total_dy = 0
    stops = 0
    ca_act = 0

    vectors = []
    for i in range(len(positions) - 1):
        dx = positions[i+1][0] - positions[i][0]
        dy = positions[i+1][1] - positions[i][1]
        current_action = action_list[positions[i][2]]
        hyp_dist=''
        if 'Forward' in current_action:    
            hyp_dist = ''.join(char for char in current_action if char.isdecimal())
            # print('=================')
            # print(np.int16(hyp_dist))
            # print(dx)
            if np.int16(hyp_dist) > dx or np.int16(hyp_dist) > dy:
                ca_act += 1
            if i > 0 and 'Forward' not in action_list[positions[i-1][2]]:
                turns_pivot +=1
        # print(dy)

        total_dx += abs(dx)
        total_dy += abs(dy)

        if dx != 0 or dy != 0:
            vectors.append((dx, dy))
        else:
            stops += 0
        # print(vectors)
    for i in range(len(vectors) - 1):
        cp = cross_product_2d(vectors[i], vectors[i+1])
        if cp != 0:
            turns_original += 1
    
    total_manhattan = total_dx + total_dy
    energy=None

    return turns_original, total_dx, total_dy, total_manhattan, stops, ca_act, turns_pivot, energy


def evaluate_sim(sim, golden_path_length_per_ep_tot=None, golden_sim=None, fault_id = None, fsim_log=None):
    turns_original_per_ep = dict()
    stops_per_ep = dict()
    path_length_per_ep_x = dict()
    path_length_per_ep_y = dict()
    path_length_per_ep_tot = dict()
    ca_act_per_ep = dict()
    turns_pivot_per_ep = dict()
    energy_per_ep = dict()
    for ep in range(len(sim)):
        turns_original, total_dx, total_dy, total_manhattan, stops, ca_act, turns_pivot, energy = evaluate_path(sim[ep])
        turns_original_per_ep[ep] = turns_original
        turns_pivot_per_ep[ep] = turns_pivot
        stops_per_ep[ep] = stops
        ca_act_per_ep[ep] = ca_act
        energy_per_ep[ep] = energy
        

        path_length_per_ep_x[ep] = total_dx
        path_length_per_ep_y[ep] = total_dy
        path_length_per_ep_tot[ep] = total_manhattan

    return turns_original_per_ep, path_length_per_ep_x, path_length_per_ep_y, path_length_per_ep_tot, stops_per_ep, ca_act_per_ep, turns_pivot_per_ep, energy_per_ep

def main(args):

    root_path = f'{args.fsim_log}'
    file_name = 'evaluation__test.p'
    idx = 0
    csv_file = f'{root_path}/summary.csv'

    golden_file_path = f'{root_path}/Golden_results/evaluation__test.p'
    golden_file_path_det = f'{root_path}/Golden_results.json'

    header = ['fault_id',
                'episode_id',
                'golden_steps',
                'faulty_steps',
                'golden_number_of_turns',
                'golden_number_of_turns_pivot',
                'golden_number_of_stops',
                'golden_ca_act',
                'faulty_number_of_turns',
                'faulty_number_of_turns_pivot',
                'faulty_number_of_stops',
                'faulty_ca_act',
                'golden_path_length',
                'faulty_path_length',
                'longer_path',
                'golden_termination',
                'faulty_termination',
                'step_wise_euclidean_max',
                'step_wise_euclidean_avg',
                'step_wise_euclidean_min',
                'step_wise_euclidean_tot',
                'golden_energy',
                'faulty_energy',
                'agrees',
                'faulty_obs_mean',
                'golden_obs_mean',
                'faulty_obs_min',
                'golden_obs_min',
                'faulty_obs_var',
                'golden_obs_var',
                'faulty_rel_distances',
                'golden_rel_distances',
                'faulty_max_q',
                'golden_max_q',
                'faulty_min_q',
                'golden_min_q',
                'faulty_skw_q',
                'golden_skw_q',
                'faulty_kur_q',
                'golden_kur_q',
                'faulty_max_prob',
                'golden_max_prob',
                'faulty_min_prob',
                'golden_min_prob',
                'faulty_skw_prob',
                'golden_skw_prob',
                'faulty_kur_prob',
                'golden_kur_prob',
                # 'faulty_max_prob',
                # 'golden_max_prob',
                # 'faulty_skw_prob',
                # 'golden_skw_prob',
                ]


    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

    golden_data = pk_read(golden_file_path)
    with open(golden_file_path_det, 'r') as f:
        golden_data_det = json.load(f)

    # print(golden_data)
    golden_turns_per_ep, golden_path_length_per_ep_x, golden_path_length_per_ep_y, golden_path_length_per_ep_tot, golden_stops_per_ep, golden_ca_act_per_ep, golden_turns_pivot_per_ep, golden_energy_per_ep = evaluate_sim(golden_data)

    data_sheet = pd.DataFrame()

    # Iterate over the available data
    for folder in tqdm([file for file in os.listdir(root_path) if file.startswith('F_') and os.path.isdir(os.path.join(root_path, file))]):

        idx += 1
        folder_path = os.path.join(root_path, folder)
        file_path = os.path.join(folder_path, file_name)

        faulty_file_path_det = os.path.join(folder_path, f'{folder}.json')

        # Initialize the template of the info to collect at the episode level
        template = {
                'fault_id':folder.split('_')[1],
                'episode_id': None,
                'golden_steps' : None, 
                'faulty_steps' : None,
                'golden_number_of_turns': None,
                'golden_number_of_turns_pivot': None,
                'golden_number_of_stops': None,
                'golden_ca_act': None,
                'faulty_number_of_turns': None,
                'faulty_number_of_turns_pivot': None,
                'faulty_number_of_stops': None,
                'faulty_ca_act': None,
                'golden_path_length': None,
                'faulty_path_length': None,
                'longer_path': None,             
                'golden_termination': None,             
                'faulty_termination': None,             
                'step_wise_euclidean_max' : None,
                'step_wise_euclidean_avg' : None,
                'step_wise_euclidean_min' : None,
                'step_wise_euclidean_tot' : None,
                'golden_energy': None,
                'faulty_energy':None,
                'agrees': None,
                'faulty_obs_mean': None,
                'golden_obs_mean': None,
                'faulty_obs_min': None,
                'golden_obs_min': None,
                'faulty_obs_var': None,
                'golden_obs_var': None,
                'faulty_rel_distances':None,
                'golden_rel_distances':None,
                'faulty_max_q':None,
                'golden_max_q':None,
                'faulty_min_q':None,
                'golden_min_q':None,
                'faulty_skw_q':None,
                'golden_skw_q':None,
                'faulty_kur_q':None,
                'golden_kur_q':None,
                'faulty_max_prob':None,
                'golden_max_prob':None,
                'faulty_min_prob':None,
                'golden_min_prob':None,
                'faulty_skw_prob':None,
                'golden_skw_prob':None,
                'faulty_kur_prob':None,
                'golden_kur_prob':None,
            }
        
        if os.path.exists(file_path):
            faulty_data = pk_read(file_path)
            with open(faulty_file_path_det, 'r') as f:
                faulty_data_det = json.load(f)

            # print(faulty_data[0])
            # Evaluate the number of turns and the path length in the faulty scenario
            faulty_turns_per_ep, faulty_path_length_per_ep_x, faulty_path_length_per_ep_y, faulty_path_length_per_ep_tot, faulty_stops_per_ep, faulty_ca_act_per_ep, faulty_turns_pivot_per_ep, faulty_energy_per_ep = evaluate_sim(faulty_data, golden_path_length_per_ep_tot, golden_sim=golden_data, fault_id = folder.split('_')[1], fsim_log=root_path)
            # print(faulty_path_length_per_ep_tot)
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)

                for episode_id in range(len(faulty_data)):
                    faulty_sim = faulty_data[episode_id]

                    faulty_sim_det = faulty_data_det[f'ep{episode_id+1}']
                    golden_sim_det = golden_data_det[f'ep{episode_id+1}']
                    
                    euc_stats = []
                    # manh_stats = []
                    # euc_ratio_stats = []

                    max_euc = None
                    avg_euc = None
                    min_euc = None
                
                    golden_sim = golden_data[episode_id]

                    # Keep track of which path is longer between golden and faulty
                    longer_path = 'same'
                    
                    agrees = list()
                    
                    for golden_step, faulty_step in zip_longest(range(len(golden_sim_det['actions'])), range(len(faulty_sim_det['actions'])), fillvalue='Not avail'):
                        
                        # if the golden path is shorter
                        if golden_step =='Not avail':
                            golden_current_step_detail = deepcopy(golden_last_step)
                            faulty_current_step_detail = faulty_sim[faulty_step+1]
                            longer_path = 'faulty'
                            agree = False
                            

                        # if the faulty path is shorter
                        elif faulty_step == 'Not avail':
                            faulty_current_step_detail = deepcopy(faulty_last_step)
                            golden_current_step_detail = golden_sim[golden_step+1]
                            longer_path = 'golden'
                            agree = False
                        
                        else:
                            golden_current_step_detail = golden_sim[golden_step+1]
                            faulty_current_step_detail = faulty_sim[faulty_step+1]
                                    
                            golden_last_step = deepcopy(golden_current_step_detail)
                            faulty_last_step = deepcopy(faulty_current_step_detail)
                            agree = golden_sim_det['actions'][golden_step] == faulty_sim_det['actions'][faulty_step]
                        
                        agrees.append(agree)

                        # position in the faulty path
                        fp = (faulty_current_step_detail['point'].x, faulty_current_step_detail['point'].y, faulty_current_step_detail['point'].z)

                        # position in the golden path
                        gp = (golden_current_step_detail['point'].x, golden_current_step_detail['point'].y, golden_current_step_detail['point'].z)

                        # compute step-wise distances
                        euc_distance = euclidean(gp, fp)

                        euc_stats.append(euc_distance)

                    golden_termination = golden_data[episode_id][-1]['end']
                    faulty_termination = faulty_data[episode_id][-1]['end']


                    # compute average, min and max per episode
                    max_euc = max(euc_stats)
                    avg_euc = sum(euc_stats)/len(euc_stats)
                    min_euc = min(euc_stats)
                    tot_euc = sum(euc_stats)
                    
                    # save info
                    template['golden_number_of_turns'] = golden_turns_per_ep[episode_id]
                    template['golden_number_of_turns_pivot'] = golden_turns_pivot_per_ep[episode_id]
                    template['golden_ca_act'] = golden_ca_act_per_ep[episode_id]
                    template['golden_path_length'] = golden_path_length_per_ep_tot[episode_id]

                    template['longer_path'] = longer_path
                    template['golden_termination'] = golden_termination
                    template['faulty_termination'] = faulty_termination

                    template['step_wise_euclidean_max']=max_euc
                    template['step_wise_euclidean_avg']=avg_euc
                    template['step_wise_euclidean_min']=min_euc
                    template['step_wise_euclidean_tot']=tot_euc

                    template['faulty_steps'] = len(faulty_sim)
                    template['episode_id'] = episode_id
                    template['golden_steps'] = len(golden_sim)

                    template['faulty_number_of_turns'] = faulty_turns_per_ep[episode_id]
                    template['faulty_number_of_turns_pivot'] = faulty_turns_pivot_per_ep[episode_id]
                    template['faulty_number_of_stops'] = faulty_stops_per_ep[episode_id]
                    template['faulty_ca_act'] = faulty_ca_act_per_ep[episode_id]
                    template['faulty_path_length'] = faulty_path_length_per_ep_tot[episode_id]

                    template['agrees'] = agrees

                    template['faulty_obs_mean'] = faulty_sim_det['obs_mean']
                    template['golden_obs_mean'] = golden_sim_det['obs_mean']

                    template['faulty_obs_min'] = faulty_sim_det['obs_min']
                    template['golden_obs_min'] = golden_sim_det['obs_min']

                    template['faulty_obs_var'] = faulty_sim_det['obs_std']
                    template['golden_obs_var'] = golden_sim_det['obs_std']

                    template['faulty_rel_distances'] = faulty_sim_det['rel_distance']
                    template['golden_rel_distances'] = golden_sim_det['rel_distance']
                    # max_q, min_q, skw_q, kur_q, max_prob, min_prob, skw_prob, kur_prob

                    template['faulty_max_q'] = faulty_sim_det['max_q']
                    template['golden_max_q'] = golden_sim_det['max_q']

                    template['faulty_min_q'] = faulty_sim_det['min_q']
                    template['golden_min_q'] = golden_sim_det['min_q']

                    template['faulty_skw_q'] = faulty_sim_det['skw_q']
                    template['golden_skw_q'] = golden_sim_det['skw_q']

                    template['faulty_kur_q'] = faulty_sim_det['kur_q']
                    template['golden_kur_q'] = golden_sim_det['kur_q']

                    template['faulty_max_prob'] = faulty_sim_det['max_prob']
                    template['golden_max_prob'] = golden_sim_det['max_prob']

                    template['faulty_min_prob'] = faulty_sim_det['min_prob']
                    template['golden_min_prob'] = golden_sim_det['min_prob']

                    template['faulty_skw_prob'] = faulty_sim_det['skw_prob']
                    template['golden_skw_prob'] = golden_sim_det['skw_prob']

                    template['faulty_kur_prob'] = faulty_sim_det['kur_prob']
                    template['golden_kur_prob'] = golden_sim_det['kur_prob']


                    writer.writerow(template)
                    f.flush()
                    
    fault_list_path = os.path.join(root_path, 'fault_list.csv')
    fault_list = pd.read_csv(fault_list_path)
    fault_list['fault_id'] = fault_list['Unnamed: 0']

    root_csv_path = os.path.join(root_path, 'summary.csv')
    data = pd.read_csv(root_csv_path)

    root_csv_path = os.path.join(root_path, 'fsim_report.csv')
    general_data = pd.read_csv(root_csv_path)
    general_data['fault_id'] = general_data['Unnamed: 0']

    data = pd.merge(data, fault_list, on='fault_id', how='inner')
    general_data = pd.merge(general_data, fault_list, on='fault_id', how='inner')

    general_data['SRD (%)'] = ((general_data['golden_goal_prob']-general_data['faulty_goal_prob'])/general_data['golden_goal_prob'])*100

    target_metrics=[
        'delta_number_of_turns',
        'delta_number_of_stops',
        'TDI (%)',
        'Relative_dx_change (%)',
        'Relative_dy_change (%)',
        'MPD (m)',
        'Maximum Step-wise Eucledian distance (m)',
        'Total Step-wise Eucledian distance (m)',
    ]

    complexity_metrics = [
        'golden_number_of_turns',
        'golden_norm_turns (%)',
        'golden_path_length (m)',
        'golen_turns_per_unit_distance (T/L)',
    ]

    data['Travel distance per turn'] = data['golden_path_length'] / data['golden_number_of_turns']
    data.replace([np.inf, -np.inf, np.nan], 100, inplace=True)
    data['golen_turns_per_unit_distance (T/L)'] = data['golden_number_of_turns'] / data['golden_path_length']
    data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    data['golden_norm_turns (%)'] = data['golden_number_of_turns'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    data['delta_number_of_stops'] = data['faulty_number_of_stops'] - data['golden_number_of_stops']
    data['delta_number_of_turns'] = abs(data['faulty_number_of_turns'] - data['golden_number_of_turns'])
    data['TDI (%)'] = (abs(data['faulty_path_length'] - data['golden_path_length'])/data['golden_path_length'])*100

    data.loc[(data['golden_path_length']==0) & (data['faulty_path_length']!=0),'TDI (%)'] = 100
    data.loc[(data['golden_path_length']==0) & (data['faulty_path_length']==0),'TDI (%)'] = 0


    mapping_columns={
        'step_wise_euclidean_avg':'MPD (m)',
        'step_wise_euclidean_max':'Maximum Step-wise Eucledian distance (m)',
        'step_wise_euclidean_tot':'Total Step-wise Eucledian distance (m)'
    }

    data.rename(columns=mapping_columns, inplace=True)


    for col in data.columns:
        if col not in ['fault_id', 'golden_termination',
        'faulty_termination', 'label']:
            try:
                data[col]=data[col].astype(float)
            except:
                print(f"Colunm: '{col}' contains formatting mistakes")
                print(data[col].unique())

    # print(data.columns)
    percent_faulty = data['faulty_termination'].value_counts(normalize=True) * 100
    percent_golden = data['golden_termination'].value_counts(normalize=True) * 100

    data['label'] = np.nan

    data['Complexity']=data['Travel distance per turn']
    bins=5
    bin_labels = [_+1 for _ in range(bins)]
    data['Complexity bin'] = pd.cut(data['Complexity'], bins=bins, labels=bin_labels)

    root_root = '/'.join(args.fsim_log.split('/')[-2:])
    
    if not os.path.exists(f"map_tool_box/AirSimNNaviFI/analysis/l{args.target_lyr}"):
        os.mkdir(f"{root_root}/map_tool_box/AirSimNNaviFI/analysis/l{args.target_lyr}")

    data[(data['golden_termination']=='Goal')].groupby(['ber', 'Complexity bin'])[['MPD (m)', 'TDI (%)']].mean().reset_index().to_csv(f'map_tool_box/AirSimNNaviFI/analysis/l{args.target_lyr}/manh_success_complexity.csv')
    data.groupby(['ber', 'Complexity bin'])[['MPD (m)', 'TDI (%)']].mean().reset_index().to_csv(f'map_tool_box/AirSimNNaviFI/analysis/l{args.target_lyr}/Path_deviation.csv')
    general_data.groupby(['ber'])[['SRD (%)']].mean().reset_index().to_csv(f'map_tool_box/AirSimNNaviFI/analysis/l{args.target_lyr}/SRD.csv')

if __name__ == '__main__':
    arguments = get_argparse()
    main(arguments.parse_args())