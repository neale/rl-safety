"""
Algorithm for Proximal Policy Optimization.

Note that this comes from my (Carroll's) self-training exercises.
It should probably be replaced with OpenAI baselines.
"""

import os
import gym
import logging
from types import SimpleNamespace
from collections import namedtuple
from functools import wraps

import numpy as np
import tensorflow as tf

from .wrappers import AutoResetWrapper
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

logger = logging.getLogger(__name__)

DEFAULT_LOGDIR = os.path.join(__file__, '../../data/tmp')
DEFAULT_LOGDIR = os.path.abspath(DEFAULT_LOGDIR)


def ppo_cnn(scaled_images, **kwargs):
    activ = tf.nn.elu
    conv1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=1, init_scale=np.sqrt(2), **kwargs))
    conv2 = activ(conv(conv1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    conv3 = activ(conv(conv2, 'c3', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    conv3 = conv_to_fc(conv3)
    return activ(linear(conv3, 'fc1', n_hidden=512, init_scale=0.01))


def ppo_cnn_lstm(scaled_images, **kwargs):
    activ = tf.nn.elu
    conv1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=1, init_scale=np.sqrt(2), **kwargs))
    conv2 = activ(conv(conv1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    conv3 = activ(conv(conv2, 'c3', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    conv3 = conv_to_fc(conv3)
    # try w/o LSTM first
    return activ(linear(conv3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class PolicyPPO(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(PolicyPPO, self).__init__(*args, **kwargs,
                                        cnn_extractor=ppo_cnn,
                                        feature_extraction="cnn")

class StablePPO2(object):
    def __init__(self, envs, logdir=DEFAULT_LOGDIR, saver_args={}, **kwargs):
        for key, val in kwargs.items():
            if (not key.startswith('_') and hasattr(self, key) and
                    not callable(getattr(self, key))):
                setattr(self, key, val)
            else:
                raise ValueError("Unrecognized parameter: '%s'" % (key,))

        self.logdir = logdir
        self.envs = [AutoResetWrapper(env) for env in envs]
        self.env = DummyVecEnv([lambda: env for env in envs])
        
        n_cpu = 4
        self.env = SubprocVecEnv([lambda: gym.make('BreakoutDeterministic-v3') for i in range(n_cpu)])
        
        self.op = SimpleNamespace()
        self.num_steps = 0
        self.num_episodes = 0
        self.save_path = os.path.join(logdir, 'model')
        #self.restore_checkpoint(logdir)

        """ ----------
        gamma : ndarray
            Set of discount factors used to calculate the discounted rewards.
        lmda : float or ndarray
            Discount factor for generalized advantage estimator. If an array,
            it should be the same shape as gamma.
        policy_discount_weights : ndarray
            Relative importance of the advantages at the different discount
            factors in the policy loss function. Should sum to one.
        value_discount_weights : ndarray
            Relative importance of the advantages at the different discount
            factors in the value loss function. Should sum to one.
        vf_coef : float
            Overall coefficient of the value loss in the total loss function.
            Would be redundant with `value_discount_weights` if we didn't
            force that to sum to one.
        learning_rate : float
        entropy_reg : float
        entropy_clip : float
            Used in entropy regularization. The regularization effectively doesn't
            turn on until the entropy drops below this level.
        max_gradient_norm : float
        eps_clip : float
            The PPO clipping for both policy and value losses. Note that this
            implies that the value function has been scaled to roughly unit value.
        rescale_policy_eps : bool
            If true, the policy clipping is scaled by ε → (1-π)ε
        min_eps_rescale : float
            Sets a lower bound on how much `eps_clip` can be scaled by.
            Only relevant if `rescale_policy_eps` is true.
        reward_clip : float
            Clip absolute rewards to be no larger than this.
            If zero, no clipping occurs.
        value_grad_rescaling : str
            One of [False, 'smooth', 'per_batch', 'per_state'].
            Sets the way in which value function is rescaled with entropy.
            This makes sure that the total gradient isn't dominated by the
            value function when entropy drops very low.
        policy_rectifier : str
            One of ['relu', 'elu'].
        """
        self.gamma = 0.99
        self.lmda = 0.9  # generalized advantage estimation parameter
        self.policy_discount_weights = np.array([1.0], dtype=np.float32)
        self.value_discount_weights = np.array([1.0], dtype=np.float32)

        self.learning_rate = 3e-4
        self.entropy_reg = 1e-2
        self.entropy_clip = 1.0  # don't start regularization until it drops below this
        self.vf_coef = 1.0
        self.max_gradient_norm = 1.0
        self.eps_clip = 0.1  # PPO clipping for both value and policy losses
        self.rescale_policy_eps = False
        self.min_eps_rescale = 1e-3  # Only relevant if rescale_policy_eps = True
        self.reward_clip = 0.0
        self.value_grad_rescaling = 'smooth'  # one of [False, 'smooth', 'per_batch', 'per_state']
        self.policy_rectifier = 'relu'  # or 'elu' or ...more to come

        self.steps_per_env = 20
        self.envs_per_minibatch = 4
        self.epochs_per_batch = 3
        self.total_steps = 5e6
        self.report_every = 5000
        self.save_every = 10000

        ppo_args = { 
                #'policy': PolicyPPO,
                'policy': CnnPolicy,
                'env': self.env,
                'gamma': self.gamma,
                'n_steps': self.steps_per_env,
                'ent_coef': self.entropy_reg,
                'learning_rate': self.learning_rate,
                'vf_coef': self.vf_coef, 
                'max_grad_norm': self.max_gradient_norm,
                'lam': self.lmda,
                'nminibatches': self.envs_per_minibatch,
                'noptepochs': self.epochs_per_batch,
                'cliprange': self.eps_clip, ## TODO
                'cliprange_vf': self.eps_clip, ## TODO
                'verbose': 1,
                'tensorboard_log': logdir,
                '_init_setup_model': True,
                'policy_kwargs': None,
                'full_tensorboard_log': True
        }

        self.model = PPO2(**ppo_args) # Nature Policy
    
    def train(self, total_steps=None):
        print ('logging to ', self.logdir)
        self.model.learn(total_timesteps=int(self.total_steps), log_interval=1, tb_log_name=self.logdir)
        self.model.save(self.save_path)
    
    def log_episode(self, info):
        self.num_episodes += 1
        summary = tf.Summary()
        summary.value.add(tag='episode/reward', simple_value=info['episode_reward'])
        summary.value.add(tag='episode/length', simple_value=info['episode_length'])
        summary.value.add(tag='episode/completed', simple_value=self.num_episodes)
        self.logger.add_summary(summary, self.num_steps)
        logger.info(
            "Episode %i: length=%i, reward=%0.1f",
            self.num_episodes, info['episode_length'], info['episode_reward'])
