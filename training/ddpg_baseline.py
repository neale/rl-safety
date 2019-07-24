"""
Algorithm for Proximal Policy Optimization.

Note that this comes from my (Carroll's) self-training exercises.
It should probably be replaced with OpenAI baselines.
"""

import os
import logging
from types import SimpleNamespace
from collections import namedtuple
from functools import wraps

import numpy as np
import tensorflow as tf

from .wrappers import AutoResetWrapper
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

logger = logging.getLogger(__name__)

DEFAULT_LOGDIR = os.path.join(__file__, '../../data/tmp')
DEFAULT_LOGDIR = os.path.abspath(DEFAULT_LOGDIR)


def ddpg_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    conv1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=1, init_scale=np.sqrt(2), **kwargs))
    conv2 = activ(conv(conv1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    conv3 = activ(conv(conv2, 'c3', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    conv3 = conv_to_fc(conv3)
    return activ(linear(conv3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def ddpg_cnn_lstm(scaled_images, **kwargs):
    activ = tf.nn.relu
    conv1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=1, init_scale=np.sqrt(2), **kwargs))
    conv2 = activ(conv(conv1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    conv3 = activ(conv(conv2, 'c3', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    conv3 = conv_to_fc(conv3)
    # try w/o LSTM first
    return activ(linear(conv3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class PolicyDDPG(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(PolicyDDPG, self).__init__(*args, **kwargs,
                                        cnn_extractor=ddpg_cnn,
                                        feature_extraction="cnn")

class StableDDPG(object):
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
        self.memory_policy = None 
        self.eval_env = None
        self.nb_train_steps = 50 
        self.nb_rollout_steps = 100
        self.nb_eval_steps = 100
        self.param_noise = None
        self.action_noise = None
        self.normalize_observations = False
        self.tau = 0.001
        self.batch_size = 128 
        self.param_noise_adaption_interval = 50
        self.normalize_returns = False
        self.enable_popart = False
        self.observation_range = (-5.0, 5.0)
        self.critic_l2_reg = 0.0 
        self.return_range = (-np.inf, np.inf)
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.clip_norm = None
        self.reward_scale = 1.0
        self.render = False
        self.render_eval = False
        self.memory_limit = None
        self.buffer_size = 50000
        self.random_exploration = 0.0
        self.verbose = 0
        
        ddpg_args = { 
                'policy': PolicyDDPG,
                'env': self.env,
                'gamma': self.gamma,
                'memory_policy': self.memory_policy,
                'eval_env': self.eval_env,
                'nb_train_steps': self.nb_train_steps,
                'nb_rollout_steps': self.nb_rollout_steps,
                'nb_eval_steps': self.nb_eval_steps,
                'param_noise': self.param_noise,
                'action_noise': self.action_noise,
                'normalize_observations': self.normalize_observations,
                'tau': self.tau,
                'batch_size': self.batch_size,
                'param_noise_adaption_interval': self.param_noise_adaption_interval,
                'normalize_returns': self.normalize_returns,
                'enable_popart': self.enable_popart,
                'observation_range': self.observation_range,
                'critic_l2_reg': self.critic_l2_reg,
                'return_range': self.return_range,
                'actor_lr': self.actor_lr,
                'critic_lr': self.critic_lr,
                'clip_norm': self.clip_norm,
                'reward_scale': self.reward_scale,
                'render': self.render,
                'render_eval': self.render_eval,
                'memory_limit': self.memory_limit,
                'buffer_size': self.buffer_size,
                'random_exploration': self.random_exploration,

                'verbose': 1,
                'tensorboard_log': self.logdir,
                '_init_setup_model': True,
                'policy_kwargs': None,
                'full_tensorboard_log': True
        }

        self.model = DDPG(**ddpg_args) # Nature Policy
    
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
