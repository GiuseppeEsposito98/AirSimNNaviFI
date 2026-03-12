from map_tool_box.modules import Utils
from pathlib import Path
import numpy as np
import random
import copy
import os
import sys
import torch
import scipy

SUCCESS_REASONS = ['Goal']

# step through an entire episode with environment, generationg actions from model
# environment is an Episodic class object in the Environment.py module
# model can be any object that has a predict() function as defined below
def play_episode(environment, model, save_observations=False, save_qvalues=False, episode=0, Fsim_setup = None, run='', req_stat=None, input_list=None, calib_iter = None):
    
    actions = list()

    obs_mean = list()
    obs_std = list()
    obs_min = list()
    
    max_prob = list()
    skw_prob = list()

    # start episode
    observations, state = environment.start()

    # initialize and log state variables
    if save_observations: # include observations? (this is memory heavy)
        state['observations'] = copy.deepcopy(observations)

    obs_mean.append(float(np.mean(observations['img'][0,:,:])))
    obs_min.append(float(np.min(observations['img'][0,:,:])))
    obs_std.append(float(np.var(observations['img'][0,:,:])))
    # questo slicing è da controllare

    states = [state]
    
    # iterate through each step
    terminate = False
    while(not terminate):
        
        action_value, q_values = model.predict2(observations)
        
        actions.append(int(action_value))
    
        # take step based on action value
        returned_values = environment.step(action_value)
        if len(returned_values) == 3:
            observations, terminate, state = returned_values
        if len(returned_values) == 4:
            observations, reward, terminate, state = returned_values
        if len(returned_values) == 5:
            observations, reward, terminate, truncate, state = returned_values
        state['action_value'] = action_value

        if save_qvalues:
            state['q_values'] = q_values

        probs = scipy.special.softmax(q_values)
        max_prob.append(float(np.max(probs)))
        skw_prob.append(float(scipy.stats.skew(probs)))

        # log state variables
        if save_observations: # include observations? (this is memory heavy)
            state['observations'] = copy.deepcopy(observations)
        
        obs_mean.append(float(np.mean(observations['img'][0,:,:])))
        obs_min.append(float(np.min(observations['img'][0,:,:])))
        obs_std.append(float(np.var(observations['img'][0,:,:])))

        states.append(state)

        if req_stat != None:
            stat = dict()
            for metric in req_stat:
                if metric == 'Energy':
                    if len(path) >=4:
                        cum_energy_per_step += energy_from_buffer(observation_data['vec'])
                        stat[metric] = cum_energy_per_step
                    else:
                        stat[metric] = -1

    # end episode
    environment.end()
    print(actions)

    if Fsim_setup:
        Fsim_setup.FI_report.update_report(episode, len(states), states[-1]['end'], actions, obs_mean, obs_min, obs_std, max_prob, skw_prob)

    if req_stat == None:
        return states
    else:
        return states, stat

# step through several episodes with environment and model
# calculates navigation accuracy
def eval(environment, model, write_path=None, save_observations=False, print_freq=10,
         ckpt_freq=10, goal_threshold=4, output_progress=False, start_path_idx=None, end_path_idx=None,
         fault=None, FI_setup=None, run='', req_stat=None, calib_iter = None, save_qvalues=False):

    # reset spawner -- sets any vars as needed to check if there are more paths to evaluate
    environment.spawner.reset()

    # keep playing a new episode while spawner has more paths to evaluate on
    if write_path is not None and os.path.exists(write_path):
        episodes = Utils.pickle_read(write_path)
        # for states in episodes:
        #     print(states)
        #     sys.exit()
        accuracy = 100*np.mean([states[-1]['end'] in SUCCESS_REASONS for states in episodes])
        if output_progress:
            progress = f'continuing eval from file of size {len(episodes)} and accuracy {accuracy:.2f}%'
            job_name = Utils.get_global('job_name')
            job_note = Utils.get_global('job_note')
            if job_name is not None:
                Utils.update_progress(job_name, job_note + ' ' + progress)
            print(progress)
        print(f'final accuracy: {accuracy:.2f}% with {len(episodes)} episodes')
    else:
        episodes = []
        # sys.exit()

    # set spawner index to length of episodes
    environment.spawner.skip_to(len(episodes))
    episode_idx = 0
    # iterate through each episode
    while(environment.spawner.has_more_paths()):
        n_episodes = len(episodes)

        # output progress
        if output_progress and n_episodes % print_freq == 0 and n_episodes > 0:

            accuracy = 100*np.mean([states[-1]['end'] in SUCCESS_REASONS for states in episodes])
            progress = f'on episode {n_episodes} with accuracy {accuracy:.2f}%'
            job_name = Utils.get_global('job_name')
            job_note = Utils.get_global('job_note')
            print(f'final accuracy: {accuracy:.2f}% with {len(episodes)} episodes')
            if job_name is not None:
                Utils.update_progress(job_name, job_note + ' ' + progress)
            print(progress)

        # checkpoint progress
        if write_path is not None and n_episodes % ckpt_freq == 0:
            Utils.pickle_write(write_path, episodes)

        # step though single episode
        if req_stat == None:
            states = play_episode(environment, model, save_observations, save_qvalues, episode = episode_idx, Fsim_setup=FI_setup, run=run, req_stat=req_stat, calib_iter=calib_iter)
            episodes.append(states)
        else:
            states, stats = play_episode(environment, model, save_observations, save_qvalues, episode = episode_idx, Fsim_setup=FI_setup, run=run, req_stat=req_stat, calib_iter=calib_iter)
            stats_per_ep[episode_idx] = {}
            for stat in req_stat:
                stats_per_ep[episode_idx][stat] = stats[stat]/len(rewards)
            episodes.append([states, stats[stat]/len(rewards)])
        episode_idx+=1
        # print(episodes)
        # sys.exit()

    # checkpoint progress
    if write_path is not None:
        Utils.pickle_write(write_path, episodes)

    n_successes = 0
    for episode in episodes:
        target = episode[0]['initial_target']
        for state in episode:
            point = state['point']
            distance = point.distance(target)
            if distance <= goal_threshold:
                n_successes += 1
                break
    accuracy = 100 * n_successes / len(episodes)
    if output_progress:
        progress = f'finished evaluations with size {len(episodes)} and accuracy {accuracy:.2f}%'
        job_name = Utils.get_global('job_name')
        job_note = Utils.get_global('job_note')
        if job_name is not None:
            Utils.update_progress(job_name, job_note + ' ' + progress)
        print(progress)
    print(f'final accuracy: {accuracy:.2f}% with {len(episodes)} episodes')
    return accuracy, episodes

# step through several episodes with environment and model
# calculates navigation accuracy
def eval_set(environment, model, start_path_idx, end_path_idx, write_path=None, save_observations=False, print_freq=10,
         ckpt_freq=10, goal_threshold=4, output_progress=False, fault=None, FI_setup=None, run='', req_stat=None, calib_iter = None):

    # reset spawner -- sets any vars as needed to check if there are more paths to evaluate
    environment.spawner.reset()

    # keep playing a new episode while spawner has more paths to evaluate on
    if write_path is not None and os.path.exists(write_path):
        episodes = Utils.pickle_read(write_path)
        if output_progress:
            progress = f'continuing eval from file of size {len(episodes)}'
            job_name = Utils.get_global('job_name')
            job_note = Utils.get_global('job_note')
            if job_name is not None:
                Utils.update_progress(job_name, job_note + ' ' + progress)
            print(progress)
    else:
        episodes = {}

    # set spawner index to length of episodes
    path_idx = start_path_idx + len(episodes)
    environment.spawner.skip_to(path_idx)

    # iterate through each episode
    n_episodes = len(episodes)
    while(path_idx < end_path_idx):
        n_episodes = len(episodes)

        # output progress
        if output_progress and n_episodes % print_freq == 0 and n_episodes > 0:
            progress = f'on episode {path_idx} of {end_path_idx}'
            job_name = Utils.get_global('job_name')
            job_note = Utils.get_global('job_note')
            if job_name is not None:
                Utils.update_progress(job_name, job_note + ' ' + progress)
            print(progress)

        # checkpoint progress
        if write_path is not None and n_episodes % ckpt_freq == 0:
            Utils.pickle_write(write_path, episodes)

        # step though single episode
        states = play_episode(environment, model, save_observations, save_qvalues, fault, FI_setup, run, req_stat, calib_iter)

        # aggregate results of episode
        episodes[path_idx] = states

        path_idx += 1

    # checkpoint progress
    if write_path is not None:
        Utils.pickle_write(write_path, episodes)

    n_successes = 0
    for path_idx in episodes:
        episode = episodes[path_idx]
        target = episode[0]['initial_target']
        for state in episode:
            point = state['point']
            distance = point.distance(target)
            if distance <= goal_threshold:
                n_successes += 1
                break
    accuracy = 100 * n_successes / len(episodes)

    if output_progress:
        progress = f'finished evaluations with size {len(episodes)} and accuracy {accuracy:.2f}'
        job_name = Utils.get_global('job_name')
        job_note = Utils.get_global('job_note')
        if job_name is not None:
            Utils.update_progress(job_name, job_note + ' ' + progress)
        print(progress)
    
    return accuracy, episodes

# model that selects a random discrete action index
class RandomDiscrete:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def predict(self, observations):
        return random.randint(0, self.n_actions-1)