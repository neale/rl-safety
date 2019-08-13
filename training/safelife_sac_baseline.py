import os
import numpy as np
import tensorflow as tf

from safelife.gym_env import SafeLifeEnv
from . import sac_baseline
from .wrappers import SafeLifeWrapper


class SafeLifeBaseSAC(sac_baseline.StableSAC):
    """
    Minimal extension to PPO to load the environment and record video.

    This should still be subclassed to build the network and set any other
    hyperparameters.
    """
    video_freq = 100
    video_counter = None
    video_name = "episode-{episode}-{steps}"

    environment_params = {}
    board_gen_params = {}

    def __init__(self, logdir=sac_baseline.DEFAULT_LOGDIR, **kwargs):
        self.logdir = logdir
        print (self.num_env)
        envs = [
            #SafeLifeWrapper(
            SafeLifeEnv(**self.environment_params)
                #self.update_environment) 
            for _ in range(self.num_env)
        ]
        super().__init__(envs, logdir=logdir, **kwargs)

    def update_environment(self, env_wrapper):
        # Called just before an environment resets
        if self.video_counter is None:
            self.video_counter = self.num_episodes
        if self.video_freq > 0 and self.video_counter % self.video_freq == 0:
            base_name = self.video_name.format(
                episode=self.video_counter, steps=self.num_steps)
            env_wrapper.video_name = os.path.join(self.logdir, base_name)
        else:
            env_wrapper.video_name = None
        self.video_counter += 1
        # If the board_gen_params are implemented as a property, then they
        # could easily be changed with every update to do some sort of
        # curriculum learning.
        env_wrapper.unwrapped.board_gen_params = self.board_gen_params


class SafeLifeSAC(SafeLifeBaseSAC):
    """
    Defines the network architecture and parameters for agent training.

    Note that this subclass is essentially designed to be a rich parameter
    file. By changing some parameters to properties (or descriptors) one
    can easily make the parameters a function of e.g. the total number of
    training steps.

    This class will generally change between training runs. Of course, copies
    can be made to support different architectures, etc., and then those can
    all be run together or in sequence.
    """

    # Training batch params
    num_env = 1
    steps_per_env = 20
    envs_per_minibatch = 4
    epochs_per_batch = 3
    total_steps = 5e6
    report_every = 5000
    save_every = 10000

    # Training network params
    gamma = np.array([0.9, 0.99], dtype=np.float32)
    policy_discount_weights = np.array([0.5, 0.5], dtype=np.float32)
    value_discount_weights = np.array([0.5, 0.5], dtype=np.float32)
    lmda = 0.9
    learning_rate = 3e-4
    entropy_reg = 1e-2
    vf_coef = 1.0
    max_gradient_norm = 1.0
    eps_clip = 0.1
    reward_clip = 10.0
    policy_rectifier = 'elu'
    scale_prob_clipping = True

    # Environment params
    environment_params = {
        'max_steps': 1200,
        'no_movement_penalty': 0.02,
        'remove_white_goals': True,
        'view_shape': (15, 15),
        'output_channels': tuple(range(15)),
    }
    board_gen_params = {
        'board_shape': (25, 25),
        'difficulty': 3,
        'max_regions': 4,
    }
