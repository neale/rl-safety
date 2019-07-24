import os
import numpy as np
import tensorflow as tf

from safelife.gym_env import SafeLifeEnv
from . import ppo_baseline
from .wrappers import SafeLifeWrapper


class SafeLifeBasePPO(ppo_baseline.StablePPO2):
    """
    Minimal extension to PPO to load the environment and record video.

    This should still be subclassed to build the network and set any other
    hyperparameters.
    """
    video_freq = 100
    video_counter = None
    video_name = "episode-{episode}-{steps}"
    test_video_name = 'test-{env_name}-{steps}'
    test_environments = []

    environment_params = {}
    board_gen_params = {}
    side_effect_args = {}

    def __init__(self, logdir=ppo_baseline.DEFAULT_LOGDIR, **kwargs):
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

    def run_safety_test(self):
        op = self.op

        def policy(obs, memory):
            fd = {op.states: [[obs]]}
            if memory is not None:
                fd[op.cell_states_in] = memory
            if op.cell_states_out is not None:
                policy, memory = self.session.run(
                    [op.policy, op.cell_states_out], feed_dict=fd)
            else:
                policy = self.session.run(op.policy, feed_dict=fd)
            policy = policy[0, 0]
            return np.random.choice(len(policy), p=policy), memory

        for idx, env_name in enumerate(self.test_environments):
            env = SafeLifeEnv(
                fixed_levels=[env_name], **self.environment_params)
            env.reset()
            video_name = os.path.join(self.logdir, self.test_video_name.format(
                idx=idx+1, env_name=env.unwrapped.state.title,
                steps=self.num_steps))
            env = SafeLifeWrapper(
                env, video_name=video_name, on_name_conflict="abort")
            safety_scores, avg_reward, avg_length = policy_side_effect_score(
                policy, env, named_keys=True, **self.side_effect_args)

            # ...somehow record this. For now just print.
            # Might want to just output to a dedicated log file.
            print("\nTesting", env.unwrapped.state.title, self.num_steps)
            print("    Episode reward", avg_reward)
            print("    Episode length", avg_length)
            print("Side effects:")
            for key, val in safety_scores.items():
                print("    {:14s} {:0.3f}".format(key, val))
            print("")


class SafeLifePPO(SafeLifeBasePPO):
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
    num_env = 16
    steps_per_env = 20
    envs_per_minibatch = 4
    epochs_per_batch = 3
    total_steps = 5e6
    report_every = 5000
    save_every = 10000

    test_every = 100000
    test_environments = ['benchmarks/test-prune-3.npz']

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
        'difficulty': 1,
        'max_regions': 4,
    }
