from map_tool_box.modules import Utils
from pathlib import Path
import numpy as np
import random
import copy
import os
from map_tool_box.AirSimNNaviFI.analysis import complexity

SUCCESS_REASONS = ['Goal']

# step through an entire episode with environment, generationg actions from model
# environment is an Episodic class object in the Environment.py module
# model can be any object that has a predict() function as defined below
def play_episode(environment, model, save_observations=False, save_qvalues=False):

    # start episode
    observations, state = environment.start()

    # initialize and log state variables
    if save_observations: # include observations? (this is memory heavy)
        state['observations'] = copy.deepcopy(observations)
    states = [state]
    
    # iterate through each step
    terminate = False
    while(not terminate):
        
        # get action from model based on observations
        if save_qvalues:
            action_value, q_values = model.predict2(observations)
        else:
            action_value = model.predict(observations)
    
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

        # log state variables
        if save_observations: # include observations? (this is memory heavy)
            state['observations'] = copy.deepcopy(observations)
        states.append(state)

    # end episode
    environment.end()
    
    return states

# step through several episodes with environment and model
# calculates navigation accuracy
def eval(environment, model, write_path=None, save_observations=False, print_freq=10,
         ckpt_freq=10, goal_threshold=4, output_progress=False, start_path_idx=None, end_path_idx=None, save_qvalues=False):

    # reset spawner -- sets any vars as needed to check if there are more paths to evaluate
    environment.spawner.reset()

    # keep playing a new episode while spawner has more paths to evaluate on
    if write_path is not None and os.path.exists(write_path):
        episodes = Utils.pickle_read(write_path)
        accuracy = 100*np.mean([states[-1]['end'] in SUCCESS_REASONS for states in episodes])
        if output_progress:
            progress = f'continuing eval from file of size {len(episodes)} and accuracy {accuracy:.2f}%'
            job_name = Utils.get_global('job_name')
            job_note = Utils.get_global('job_note')
            if job_name is not None:
                Utils.update_progress(job_name, job_note + ' ' + progress)
            print(progress)
    else:
        episodes = []

    # set spawner index to length of episodes
    environment.spawner.skip_to(len(episodes))

    # iterate through each episode
    while(environment.spawner.has_more_paths()):
        n_episodes = len(episodes)

        # output progress
        if output_progress and n_episodes % print_freq == 0 and n_episodes > 0:
            accuracy = 100*np.mean([states[-1]['end'] in SUCCESS_REASONS for states in episodes])
            progress = f'on episode {n_episodes} with accuracy {accuracy:.2f}%'
            job_name = Utils.get_global('job_name')
            job_note = Utils.get_global('job_note')
            if job_name is not None:
                Utils.update_progress(job_name, job_note + ' ' + progress)
            print(progress)

        # checkpoint progress
        if write_path is not None and n_episodes % ckpt_freq == 0:
            Utils.pickle_write(write_path, episodes)

        # step though single episode
        states = play_episode(environment, model, save_observations, save_qvalues)

        # aggregate results of episode
        episodes.append(states)

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
    
    return accuracy, episodes

# step through several episodes with environment and model
# calculates navigation accuracy
def eval_set(environment, model, start_path_idx, end_path_idx, write_path=None, save_observations=False, print_freq=10,
         ckpt_freq=10, goal_threshold=4, output_progress=False):

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
        states = play_episode(environment, model, save_observations, save_qvalues)

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
