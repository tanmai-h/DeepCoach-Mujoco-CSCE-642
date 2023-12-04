import gym
import numpy as np
import time
import argparse
import os
from memory_buffer import MemoryBuffer
from feedback import Feedback
from agents.selector import agent_selector
from simulated_teacher.selector import teacher_selector
from tools.functions import load_config_data
from simulated_teacher.pick_and_fetch import PF
from simulated_teacher.fetch_push import FP
from simulated_teacher.fetch_and_slide import FS

# Read program args
parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default='pick_and_fetch', help='We use the same configs for all 3 domains')
parser.add_argument('--exp-num', default='-1')

parser.add_argument("--use-pf", action='store_true', help="Use Pick and Fetch Domain")
parser.add_argument("--use-fp", action='store_true', help="Use Fetch and Push Domain")
parser.add_argument("--use-fs", action='store_true', help="Use Fetch and Slide Domain")

args = parser.parse_args()

config_file = args.config_file
exp_num = args.exp_num

# Load common parameters from config file
config = load_config_data('config_files/' + config_file + '.ini')
# print('config_files/' + config_file + '.ini')
config_exp_setup = config['EXP_SETUP']

environment = config_exp_setup['environment']
network = config_exp_setup['network_type']
error_prob = config_exp_setup['error_prob']
env_config_file = config_exp_setup['env_config_file']

# Load network and method parameters
network_folder = network
if network == 'HD':
    version = config_exp_setup['version']
    network_folder += '/' + version
else:
    version = ''

config = load_config_data('config_files/' + network_folder + '/' + environment + '/' + env_config_file + '.ini')
print('config_files/' + network_folder + '/' + environment + '/' + env_config_file + '.ini')
config_graph = config['GRAPH']
config_buffer = config['BUFFER']
config_general = config['GENERAL']

# Load teacher parameters
config = load_config_data('config_files/' + network_folder + '/' + environment + '/teacher.ini')
config_teacher = config['TEACHER']
config_feedback = config['FEEDBACK']

eval_save_folder = '/' + network_folder

eval_save_path = config_exp_setup['eval_save_path']
evaluate = config_exp_setup.getboolean('evaluate')
train = config_exp_setup.getboolean('train')
use_simulated_teacher = config_exp_setup.getboolean('use_simulated_teacher')
render = config_exp_setup.getboolean('render')
count_down = config_exp_setup.getboolean('count_down')
save_results = config_exp_setup.getboolean('save_results')
save_graph = config_exp_setup.getboolean('save_graph')
show_FPS = config_exp_setup.getboolean('show_FPS')
max_num_of_episodes = config_exp_setup.getint('max_num_of_episodes')
max_time_steps_episode = float(config_exp_setup['max_time_steps_episode'])
history_training_rate = config_buffer.getint('history_training_rate')
use_memory_buffer = config_buffer.getboolean('use')
render_delay = float(config_general['render_delay'])

if not use_memory_buffer:
    eval_save_folder += '_no_buffer'

output_reward_results_name = '/Enhanced_results_' + args.config_file + exp_num + '_'

pf = None if not args.use_pf else PF()
if args.use_fp:
    pf = FP()
elif args.use_fs:
    pf = FS()
# Create environment
env = gym.make(environment) if not pf else pf

# Create teacher
if use_simulated_teacher:
    teacher = teacher_selector(network,
                               version,
                               dim_a=config_teacher.getint('dim_a'),
                               action_lower_limits=config_teacher['action_lower_limits'],
                               action_upper_limits=config_teacher['action_upper_limits'],
                               loc=config_teacher['loc'],
                               error_prob=error_prob,
                               teacher_parameters=config_general['simulated_teacher_parameters'],
                               config_general=config_general,
                               config_teacher=config_teacher,
                               pickFetchEnv=pf)
observation_input_shape = env.observation_space.shape
print('env.observation_space=', env.observation_space, env)
# Create agent
agent = agent_selector(network,
                       version,
                       train_ae=config_graph.getboolean('train_autoencoder'),
                       load_policy=config_exp_setup.getboolean('load_graph'),
                       learning_rate=float(config_graph['learning_rate']),
                       dim_a=config_graph.getint('dim_a'),
                       fc_layers_neurons=config_graph.getint('fc_layers_neurons'),
                       loss_function_type=config_graph['loss_function_type'],
                       policy_loc=config_graph['policy_loc'] + exp_num + '_',
                       action_upper_limits=config_graph['action_upper_limits'],
                       action_lower_limits=config_graph['action_lower_limits'],
                       e=config_graph['e'],
                       config_graph=config_graph,
                       config_general=config_general,observation_input_shape=observation_input_shape)

# Create memory buffer
buffer = MemoryBuffer(min_size=config_buffer.getint('min_size'),
                      max_size=config_buffer.getint('max_size'))

# Create saving directory if it does no exist
if save_results:
    if not os.path.exists(eval_save_path + eval_save_folder):
        os.makedirs(eval_save_path + eval_save_folder)

# Initialize variables
total_reward, total_feedback, total_time_steps = [], [], []
r, total_r, t_counter, h_counter, last_t_counter = 0, 0, 0, 0, 0

# Print general general information
print('\nExperiment number:', exp_num)
print('Environment:', environment)

if network == 'HD':
    print('Network:', network, version, '\n')
else:
    print('Network:', network, '\n')

time.sleep(3)

# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)

# Iterate over the maximum number of episodes
init_time = time.time()
print("Max num read from the config", max_num_of_episodes)
for i_episode in range(max_num_of_episodes):
    print('Starting episode number', i_episode)
    agent.new_episode()
    observation = pf.reset()

    # Iterate over the episode
    for t in range(int(max_time_steps_episode)):
        if render:
            env.render()  # Make the environment visible
            time.sleep(render_delay)  # Add delay to rendering if necessary

        # Map action from state
        action = agent.action(np.expand_dims(observation, axis=0))

        # Act
        observation, reward, done, info = env.step(action)

        # Accumulate reward
        r += reward

        # Get feedback signal
        if use_simulated_teacher:
            h = teacher.get_feedback_signal(observation, action, t_counter)
        # else:
        #     h = human_feedback.get_h()

        # Update weights
        if train:
            if np.any(h):  # if any element is not 0
                agent.update(h, np.expand_dims(observation, axis=0))
                if not use_simulated_teacher:
                    print("feedback", h)
                h_counter += 1
                # Add state action-label pair to memory buffer
                if use_memory_buffer:
                    last_step = agent.last_step(observation)
                    if last_step is not None:
                        buffer.add(last_step)

                    # Train sampling from buffer
                    if buffer.initialized():
                        batch = buffer.sample(batch_size=config_buffer.getint('sampling_size'))
                        agent.batch_update(batch)

            # Train every k time steps
            if buffer.initialized() and t % history_training_rate == 0:
                batch = buffer.sample(batch_size=config_buffer.getint('sampling_size'))
                agent.batch_update(batch)

        t_counter += 1

        # Computations needed to do at each time step
        agent.time_step(t)

        # Calculate FPS
        if t % 100 == 0 and t != 0 and show_FPS:
            fps = (t_counter - last_t_counter) / (time.time() - init_time)
            init_time = time.time()
            last_t_counter = t_counter
            print('\nFPS:', fps, '\n')

        # End of episode
        if done:
            if evaluate:
                total_r += r
                print('Episode Reward:', '%.3f' % r)
                print('\n', i_episode, 'avg reward:', '%.3f' % (total_r / (i_episode + 1)), '\n')
                print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))
                if save_results:
                    total_reward.append(r)
                    total_feedback.append(h_counter)
                    total_time_steps.append(t_counter)
                    np.save(eval_save_path + eval_save_folder + output_reward_results_name + 'reward', total_reward)
                    np.save(eval_save_path + eval_save_folder + output_reward_results_name + 'feedback', total_feedback)
                    np.save(eval_save_path + eval_save_folder + output_reward_results_name + 'time', total_time_steps)

            if save_graph:
                agent.save_params()

            if render:
                time.sleep(1)

            h_counter = 0
            r = 0
            print('Total time (s):', '%.3f' % (time.time() - init_time))
            break
