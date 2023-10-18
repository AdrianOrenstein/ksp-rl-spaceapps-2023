"""This file runs an experiment based on parameters specified in a configuration file."""

import time
import traceback
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import krpc
from utils.sweeper import Sweeper
from utils.helpers import validate_output_folder
from envs.ksp_detumbling import GameEnv, SimplerGameEnv, DummyEnv
from agents.linear_agent import DifferentialDiscountedQlearningAgent

env_map = {'Detumbler': 'GameEnv',
           'SimplerDetumbler': 'SimplerGameEnv',
           'Dummy': 'DummyEnv'}
agent_map = {'DiffDiscQ': 'DifferentialDiscountedQlearningAgent'}


def process_observation(raw_obs):
    return raw_obs


def log_data(interval, current_timestep, current_run,
             exp_type, log, env, agent, centered_values, save_weights,
             exp_name, exp_id, nonlinear):
    if save_weights:
        index = current_timestep // interval
        log['weights'][current_run][index] = agent.weights
        log['avgrew'][current_run][index] = agent.avg_reward


def save_final_weights(nonlinear, run_idx, log, agent, exp_name, exp_id):
    if nonlinear:
        agent.save_trained_model(f'{exp_name}_{exp_id}_{run_idx}')
    else:
        log['weights_final'][run_idx] = agent.weights
    if hasattr(agent, "avg_reward"):
        log['avgrew_final'][run_idx] = agent.avg_reward


def print_experiment_summary(log, exp_type):
    if exp_type == 'control':
        tqdm.write('RewardRate_total\t= %f' % (np.mean(log['reward'])))
        tqdm.write('RewardRate_last50%%\t= %f\n' % np.mean(log['reward'][:, log['reward'].shape[1] // 2:]))
        tqdm.write('RewardRate_last10%%\t= %f\n' % np.mean(log['reward'][:, log['reward'].shape[1] // 10 * 9:]))


def run_experiment_one_config(config):
    """
    Runs N independent experiments for a particular parameter configuration.

    Args:
        config: a dictionary of all the experiment parameters
    Returns:
        log: a dictionary of quantities of interest
    """
    exp_name = config['exp_name']
    exp_type = config['exp_type']
    env_name = config['env_name']
    agent_name = config['agent_name']
    num_runs = config['num_runs']
    max_steps = config['num_max_steps']
    eval_every_n_steps = config['eval_every_n_steps']
    save_weights = config.get('save_weights', 0)
    num_weights = config['num_weights']
    store_max_action_values = config.get('store_max_action_values', False)

    log = {'reward': np.zeros((num_runs, max_steps + 1), dtype=np.float32),
           'angle': np.zeros((num_runs, max_steps + 1), dtype=np.float32),
           'action': np.zeros((num_runs, max_steps + 1), dtype=np.float32),
           'weights_final': np.zeros((num_runs, num_weights), dtype=np.float32),
           'avgrew_final': np.zeros(num_runs, dtype=np.float32),
           }
    if save_weights:
        log['avgrew'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1), dtype=np.float32)
        log['weights'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1,
                                num_weights), dtype=np.float32)
    if store_max_action_values:
        log['max_value_per_step'] = np.zeros((num_runs, max_steps // 10 + 1), dtype=np.float32)

    assert env_name in env_map, f'{env_name} not found.'
    assert agent_name in agent_map, f'{agent_name} not found.'

    for run in range(num_runs):
        config['rng_seed'] = run
        agent = getattr(sys.modules[__name__], agent_map[agent_name])(**config)
        if env_name == 'Dummy':
            env = DummyEnv()
        else:
            env = getattr(sys.modules[__name__], env_map[env_name])(krpc.connect(name="Tracker"))
        obs, _ = env.reset(seed=config['rng_seed'])
        action = agent.start(process_observation(obs))

        for t in tqdm(range(max_steps + 1)):
            # logging relevant data at regular intervals
            if t % eval_every_n_steps == 0:
                log_data(interval=eval_every_n_steps, current_timestep=t,
                         current_run=run, exp_type=exp_type, log=log,
                         env=env, agent=agent,
                         save_weights=save_weights, nonlinear=False,
                         exp_name=exp_name, exp_id=config['exp_id'],
                         centered_values=None)
            # the environment and agent step
            next_obs, reward, _, _ = env.step(action)
            # print(f'action: {action}\nobs: {next_obs}')
            # print(f'reward: {reward}')
            action = agent.step(reward, process_observation(next_obs))
            # logging the reward at each step
            log['reward'][run][t] = reward
            # logging some data for debugging
            log['action'][run][t] = action
            log['angle'][run][t] = np.arctan2(next_obs[0], next_obs[1])     # this is the *next* angle
            # print(np.rad2deg(np.arctan2(next_obs[0], next_obs[1])), env.roll())

        save_final_weights(nonlinear=False,
                           run_idx=run, log=log, agent=agent,
                           exp_name=exp_name, exp_id=config['exp_id'])

    print_experiment_summary(log, exp_type)
    return log


parser = argparse.ArgumentParser(description="Run an experiment based on parameters specified in a configuration file")
parser.add_argument('--config-file',  # required=True,
                    default='config_files/pendulum/test.json',
                    help='location of the config file for the experiment (e.g., config_files/test_config.json)')
parser.add_argument('--cfg-start', default=0)
parser.add_argument('--cfg-end', default=-1)
parser.add_argument('--output-path', default='results/test_exp/')
args = parser.parse_args()
print(args.config_file, args.output_path)
path = validate_output_folder(args.output_path)

sweeper = Sweeper(args.config_file)
cfg_start_idx = int(args.cfg_start)
cfg_end_idx = int(args.cfg_end) if args.cfg_end != -1 else sweeper.total_combinations

print(f'\n\nRunning configurations {cfg_start_idx} to {cfg_end_idx}...\n\n')

start_time = time.time()

for i in range(cfg_start_idx, cfg_end_idx):
    config = sweeper.get_one_config(i)
    config['exp_id'] = i
    config['output_folder'] = path
    # print(f'Starting at: {time.localtime(start_time)}')
    print(config)

    try:
        log = run_experiment_one_config(config)
        log['params'] = config
    except Exception as e:
        print('\n***\n')
        print(traceback.format_exc())
        print('***\nException occurred with this parameter configuration, moving on now\n***\n')
    else:
        filename = f"{config['exp_name']}_{config['exp_id']}"
        print(f'Saving experiment log in: {filename}.npy\n**********\n')
        np.save(f'{path}{filename}', log)
    finally:
        print("Time elapsed: {:.2} minutes\n\n".format((time.time() - start_time) / 60))
        os.system('sleep 0.5')

end_time = time.time()
print("Total time elapsed: {:.2} minutes".format((end_time - start_time) / 60))
