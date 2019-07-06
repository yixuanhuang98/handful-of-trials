from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
#from gym.wrappers.monitoring import VideoRecorder
import gym
from dotmap import DotMap

import time
import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import dmbrl.misc.MBExp
from dmbrl.misc.run import run

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
import pybullet_envs.bullet.racecarGymEnv as e

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 100
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 8
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 2,1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):  # ppo for choice model

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

# env = gym.make('Pendulum-v0').unwrapped
# ppo = PPO()
# all_ep_r = []

# for ep in range(EP_MAX):
#     s = env.reset()
#     buffer_s, buffer_a, buffer_r = [], [], []
#     ep_r = 0
#     for t in range(EP_LEN):    # in one episode
#         env.render()
#         a = ppo.choose_action(s)
#         s_, r, done, _ = env.step(a)
#         buffer_s.append(s)
#         buffer_a.append(a)
#         buffer_r.append((r+8)/8)    # normalize reward, find to be useful
#         s = s_
#         ep_r += r

#         # update ppo
#         if (t+1) % BATCH == 0 or t == EP_LEN-1:
#             v_s_ = ppo.get_v(s_)
#             discounted_r = []
#             for r in buffer_r[::-1]:
#                 v_s_ = r + GAMMA * v_s_
#                 discounted_r.append(v_s_)
#             discounted_r.reverse()

#             bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
#             buffer_s, buffer_a, buffer_r = [], [], []
#             ppo.update(bs, ba, br)
#     if ep == 0: all_ep_r.append(ep_r)
#     else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
#     print(
#         'Ep: %i' % ep,
#         "|Ep_r: %i" % ep_r,
#         ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
#     )

# plt.plot(np.arange(len(all_ep_r)), all_ep_r)
# plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)
    # env = e.RacecarGymEnv(isDiscrete=False ,renders=True)
    # print('bbbbbbbbbb')
    # print(env)
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    #env = e.RacecarGymEnv(isDiscrete=False ,renders=True)
    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def get_ppo():#):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    print('enter main function')
    args1 = ['run.py', '--alg=ppo2', '--env=RacecarBulletEnv-v0', '--num_timesteps=1e3']#, '--load_path=/Users/huangyixuan/models/racecar_ppo2', '--play']
    # if 4e5 ,it uses total_night
    arg_parser = common_arg_parser()
    args1, unknown_args = arg_parser.parse_known_args(args1)
    print('unknown_args')
    print(unknown_args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    print('extra')
    print(extra_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        #configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args1.log_path, format_strs=[])

    model, env = train(args1, extra_args)
    return model
    #env = e.RacecarGymEnv(isDiscrete=False ,renders=True)
    

    # if args.save_path is not None and rank == 0:
    #     save_path = osp.expanduser(args.save_path)
    #     model.save(save_path)

    # if args.play:
    #     logger.log("Running trained model")
    #     obs = env.reset()

    #     state = model.initial_state if hasattr(model, 'initial_state') else None
    #     dones = np.zeros((1,))

    #     episode_rew = 0
    #     while True:
    #         if state is not None:
    #             actions, _, state, _ = model.step(obs,S=state, M=dones)
    #         else:
    #             actions, _, _, _ = model.step(obs)
    #         print(actions)

    #         obs, rew, done, _ = env.step(actions)
    #         episode_rew += rew[0] if isinstance(env, VecEnv) else rew
    #         env.render()
    #         done = done.any() if isinstance(done, np.ndarray) else done
    #         if done:
    #             print('episode_rew={}'.format(episode_rew))
    #             episode_rew = 0
    #             obs = env.reset()

#     env.close()

#     return model

# if __name__ == '__main__':
#     main(sys.argv)



class Agent:
    """An general class for RL agents.
    """
    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will 
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the 
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env
        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None
        self.ppo_policy = get_ppo()
        self.ppo_choice = PPO()
        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]

    def sample(self, horizon, policy, record_fname=None):
        """Samples a rollout from the agent.

        Arguments: 
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        # configure logger, disable logging in child MPI processes (with rank > 0)
        
        # beginning the ppo function
        # print('enter main function')
        # arg_parser = common_arg_parser()
        # args, unknown_args = arg_parser.parse_known_args(args)
        # extra_args = parse_cmdline_kwargs(unknown_args)

        # if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        #     rank = 0
        # #configure_logger(args.log_path)
        # else:
        #     rank = MPI.COMM_WORLD.Get_rank()
        #     configure_logger(args.log_path, format_strs=[])

        # model, env = train(args, extra_args)

        # end of the ppo function

        # video_record = record_fname is not None
        # video_record = False
        # recorder = None if not video_record else gym.wrappers.Monitor(self.env, record_fname)

        
        # #horizon = 200
        # print('horizon')
        # print(horizon)

        # all_ep_r = []
        # total_ob = []
        # total_ac = []

        # for ep in range(EP_MAX):
        #     times, rewards = [], []
        #     print('reset in agent')
        #     O, A, reward_sum, done = [self.env.reset()], [], 0, False
        #     print('ob')
        #     print(O)
        #     policy.reset()
        #     buffer_s, buffer_a, buffer_r = [], [], []
        #     ep_r = 0
        #     for t in range(horizon):    # in one episode
        #         #env.render()
        #         start = time.time()
        #         a = self.ppo_choice.choose_action(O[t])
        #         print('action')
        #         print(a)
                
        #         if(a >= 0):
        #             true_action = policy.act(O[t], t)
        #         else:
        #             O_new = np.zeros((1,2))
        #             O_new[0,:] = O[t]
        #             step_action, _ , _ , _ = self.ppo_policy.step(O_new)
        #             true_action = step_action[0]
        #         A.append(true_action)
        #         print('Observation')
        #         print(O[t])
        #         print('action')
        #         print(A[t])
        #         times.append(time.time() - start)
        #         #abs(carpos[1])
        #         previous_reward = 0
        #         if(abs(O[t][1]) > 1):
        #             previous_reward = - 1 * abs(O[t][1]) 
        #         if self.noise_stddev is None:
        #             obs, reward, done, info = self.env.step(A[t])
        #         else:
        #             action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
        #             action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
        #             obs, reward, done, info = self.env.step(action)
        #         reward += previous_reward
        #         O.append(obs)
        #         reward_sum += reward
        #         rewards.append(reward)
        #         if(done):   # why we could not delete it
        #             break

        #         #s_, r, done, _ = env.step(a)
        #         if(ep >= 10):
        #             total_ob.append(obs)
        #             total_ac.append(a)
        #         buffer_s.append(obs)
        #         buffer_a.append(a)
                
        #         buffer_r.append((reward+8)/8)    # normalize reward, find to be useful
        #         #s = s_
        #         ep_r += reward
        #         if (t+1) % BATCH == 0 or t == horizon-1:
        #             v_s_ = self.ppo_choice.get_v(O[-1])
        #             discounted_r = []
        #             for r in buffer_r[::-1]:
        #                 v_s_ = r + GAMMA * v_s_
        #                 discounted_r.append(v_s_)
        #             discounted_r.reverse()

        #             bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
        #             buffer_s, buffer_a, buffer_r = [], [], []
        #             self.ppo_choice.update(bs, ba, br)
            
        #     if(ep >= 15):
        #         final_output = []
        #         final_output = np.concatenate((total_ob,total_ac),axis = 1)
        #         #np.savetxt('/home/guest/txt_result/test',(total_final)) 
        #         np.savetxt('/home/guest/txt_result/1e5_1',(final_output))
        #     if ep == 0: all_ep_r.append(ep_r)
        #     else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        #     print(
        #         'Ep: %i' % ep,
        #         "|Ep_r: %i" % ep_r,
        #         ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        #     )
            

        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        for t in range(horizon):
            if video_record:
                recorder.capture_frame()
            start = time.time()
            A.append(policy.act(O[t], t))
            times.append(time.time() - start)

            if self.noise_stddev is None:
                obs, reward, done, info = self.env.step(A[t])
            else:
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
                obs, reward, done, info = self.env.step(action)
            O.append(obs)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
