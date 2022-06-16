import os
import sys
import argparse
import time
import datetime
import json
import random
import numpy as np
import torch
import tensorflow as tf

from bnp_options import *
from utils import *
from eval import *
from env.room_env import RoomEnv
from env.atari_env import AtariEnv


def get_args():
    parser = argparse.ArgumentParser(
        description='Trains BNP learning of options')

    # Options settings
    parser.add_argument('--K', type=int, default=1,
                        help='initial number of options before the truncation')
    parser.add_argument('--tolerance', type=float, default=0.5,
                        help='threshold below which an option is considered unused')
    parser.add_argument('--check-options-usage', type=int, default=10,
                        help='Number of epochs between checks of options usage to add a new option.')
    parser.add_argument('--fixed-options', action="store_true", default=False,
                        help='prevent any option from being added')

    # Networks settings
    parser.add_argument('--hidden-layer-sizes-policy', nargs='*', default=[16],
                        help='number of hidden units per layer in policy network')
    parser.add_argument('--hidden-layer-sizes-termination', nargs='*', default=[16],
                        help='number of hidden units per layer in termination network')
    parser.add_argument('--LSTM-hidden-layer-size', type=int, default=32,
                        help='dimension of LSTM hidden state')
    parser.add_argument('--LSTM-MLP-hidden-layer-sizes', nargs='*', default=[32, 32],
                        help='number of hidden units per layer in MLP after LSTM')
    parser.add_argument('--action-space', type=str, default='discrete',
                        help='discrete or continuous action space')

    # Training settings
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--clip', type=float, default=5.,
                        help='gradient clipping')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--max-epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Used to seed random number generators')

    # Relaxation settings
    parser.add_argument('--relaxation-type', type=str, default='GS',
                        help='GS for Gumbel-Softmax (IGR not implemented)')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='Gumbel-Softmax temperature')
    parser.add_argument('--temperature-ratio', type=float, default=0.999,
                        help='Decay rate of GS temperature')
    parser.add_argument('--entropy-factor', type=float, default=5.,
                        help='Initial entropy factor')
    parser.add_argument('--entropy-ratio', type=float, default=0.995,
                        help='Decay rate of entropy factor')

    # Environment settings
    parser.add_argument('--env-type', type=str, default='room',
                        help='type of environment')
    parser.add_argument('--nb-rooms', type=int, default=6,
                        help='number of rooms in the room environment')
    parser.add_argument('--nb-traj', type=int, default=1000,
                        help='number of trajectories in the expert dataset')
    parser.add_argument('--noise-level', type=float, default=0.,
                        help='noise percentage in expert trajectories')
    parser.add_argument('--max-steps', type=int, default=6,
                        help='maximum number of steps in an expert trajectory')
    parser.add_argument('--demo-file', type=str, default='',
                        help='path to the expert trajectories file')
    parser.add_argument('--atari-env-name', type=str, default='',
                        help='name of the atari env')

    # Misc settings
    parser.add_argument('--save-dir', type=str, default='',
                        help='directory where model and config are saved')
    parser.add_argument('--results-file', type=str, default=None,
                    help='file where results are saved')
    
    # Baseline settings
    parser.add_argument('--baseline-ddo', action="store_true", default=False,
                        help='train ddo baseline instead of bnp options')

    args = parser.parse_args()
    return args


def setup_env(args):
    n_rooms = args.nb_rooms
    max_steps = args.max_steps

    if args.env_type == 'room':
        env = RoomEnv(rng=rng_env, n_rooms=n_rooms, max_steps=max_steps)
        data = env.generate_expert_trajectories(n_traj=args.nb_traj, noise_level=args.noise_level, 
                                                max_steps=args.max_steps, action_seed=action_seed)
    elif args.env_type == 'atari':
        if args.atari_env_name == '':
            env_name = args.demo_file.split('/')[-2]
            env_name = env_name[0].upper() + env_name[1:]
        else:
            env_name = args.atari_env_name
        env = AtariEnv('{}-ramNoFrameskip-v3'.format(env_name), path=args.demo_file)
        data = env.get_expert_trajectories(max_steps=args.max_steps)
    else:
        raise AssertionError("environment not defined.")

    return env, data


def split_train_test(data, rng_split, split=0.01):
    n_traj = len(data[0])
    perm = rng_split.permutation(n_traj)
    test_indices = perm[:int(split*n_traj)]
    test_data_states = data[0][test_indices]
    test_data_actions = data[1][test_indices]
    test_data_rewards = None if data[2] is None else data[2][test_indices]
    test_data = (test_data_states, test_data_actions, test_data_rewards)
    train_indices = perm[int(split*n_traj):]
    train_data_states = data[0][train_indices]
    train_data_actions = data[1][train_indices]
    train_data_rewards = None if data[2] is None else data[2][train_indices]
    train_data = (train_data_states, train_data_actions, train_data_rewards)
    return train_data, test_data


def train_baseline(trajectories, k, statedim, actiondim,
          super_iterations=2000, sub_iterations=100, learning_rate=1e-3):

    if type(statedim) != tuple:
        statedim = (statedim,)
    if type(actiondim) != tuple:
        actiondim = (actiondim,)
    model = AtariRAMModel(k, statedim=statedim, actiondim=actiondim)
    with tf.variable_scope("optimizer2"):
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        model.sess.run(tf.initialize_all_variables())
    model.train(opt, trajectories, super_iterations, sub_iterations)
    return model


if __name__ == "__main__":

    args = get_args()
    params = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.baseline_ddo:
        sys.path.append('../segment-centroid')
        from segmentcentroid.tfmodel.AtariRAMModel import AtariRAMModel

    # Creating folder for this run
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_time_s = time.time()
    run_ID = "{}_{}".format(args.env_type, datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
    if args.save_dir == '':
        run_dir = "runs/{}".format(run_ID)
    else:
        run_dir = args.save_dir
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(json.dumps(params, indent=4))

    # This will be used to generate the seeds for other RNGs.
    rng_master = np.random.RandomState(args.random_seed)
    np.random.seed(args.random_seed) # there were some issue with reproducibility
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    env_seed = rng_master.randint(100000)
    action_seed = rng_master.randint(100000)
    split_seed = rng_master.randint(100000)
    rng_env = np.random.RandomState(env_seed)
    rng_split = np.random.RandomState(split_seed)

    # Environment setup
    env, data = setup_env(args)
    train_data, test_data = split_train_test(data, rng_split)

    # Training
    if not args.baseline_ddo:
        model = BNPOptions(train_data, env.state_dim, env.action_dim, device, rng=rng_master, **vars(args))
        model.train()
    else:
        # Change data format to fit multilevel discovery
        train_data = [[(state.astype("float64"), action.astype("float64"))
                       for state, action in zip(states, actions)]
                      for states, actions in zip(train_data[0], train_data[1])]
        model = train_baseline(train_data, args.K, env.state_dim, train_data[0][0][1].shape[-1], super_iterations=args.max_epochs)

    model.save(os.path.join(run_dir, "checkpoint.pth"))

    # Evaluation
    if not args.baseline_ddo:
        score = compute_score(model, env, test_data, device)
    else:
        score = compute_score_baseline(model, env, test_data)
    print("Achieved a score of {}.".format(score))

    if args.results_file is not None:
        with open(args.results_file, 'a') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')
            f.write(str(score))
            f.write(' ')
            f.write(str(model.K))
            f.write(' ')
            f.write("[{}]".format(' '.join([str(epoch) for epoch in model.new_option_hist])))
            f.write('\n')