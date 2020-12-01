# ##################### #
# Simon Fraser University
# CMPT726 project - deep reinforcement learning for a robotic arm
#
# Trains a model using TRPO in the FetchReach environment. The FetchReach
# environment has been modified externally to enable the gripper fingers;
# the goal is to improve arm stability when the fingers are changing
# velocity.
# ##################### #
import gym
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from spinup import trpo_tf1 as trpo
from spinup.utils.test_policy import load_policy_and_env, run_policy

# FetchPickAndPlace-v1 wrappers

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.full(13, -np.inf)
        high = np.full(13, np.inf)
        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, obs):
        obs = np.concatenate(
            (obs['observation'], obs['desired_goal'])
        )
        return obs

class TrainRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.init_pos = None

    def reset(self):
        obs = self.env.reset()
        # set initial gripper position
        self.init_pos = obs[0:3]
        return obs

    def step(self, action):
        # modify action - randomly move grip fingers to add instability
        x = np.random.rand()
        if x >= 0 and x < 0.475: # 47.5% chance
            action[3] = 0.1
        elif x >= 0.475 and x < 0.95: # 47.5% chance
            action[3] = -0.1
        else: # 5% chance
            action[3] = np.clip(np.random.normal(loc=0.0, scale=0.5), -1.0, 1.0)

        # apply env step
        next_state, reward, done, info = self.env.step(action)

        # === modify reward ===
        #
        # stabilize the gripper while moving to the target location
        #
        # [0:3]   -> gripper xyz position
        # [3:5]   -> finger position
        # [10:13] -> desired goal position
        #
        grip_pos = next_state[0:3]
        finger_pos = next_state[3:5]
        goal_pos = next_state[10:13]
        
        dist_to_goal = norm(goal_pos - grip_pos, ord=2)
        
        #u = grip_pos - self.init_pos
        #v = goal_pos - self.init_pos
        #proj_u_on_v = (np.dot(u, v)/np.dot(v, v)) * v
        #z = proj_u_on_v - u
        #proj_err = norm(z, ord=2)

        #reward = -(dist_to_goal + 0.25*proj_err)

        reward = -dist_to_goal*(1 + 0.01*norm(action[0:3]))
        
        return next_state, reward, done, info


class TestWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        action[3] = 0.1 # force open fingers
        return self.env.step(action)


env_f = lambda : TrainRewardWrapper(
    ObsWrapper(gym.make('FetchReach-v1'))
)

######################################
# ======= SCRIPT STARTS HERE ======= #
######################################

skip_train = input("Skip training?\n")

if skip_train in ('n', 'N'):
    output_dir = input("Enter a output directory path to store the result to:\n")
    exp_name = input("Enter an experiment name:\n")
    
    logger_kwargs = {
        'output_dir' : output_dir, 
        'exp_name' : exp_name
    }

    ac_kwargs = {
        'hidden_sizes' : [128,128],
        'output_activation' : tf.tanh
    }

    # https://spinningup.openai.com/en/latest/algorithms/trpo.html
    trpo(
        env_f,
        ac_kwargs=ac_kwargs,
        steps_per_epoch=12000,
        epochs=100,
        gamma=0.99,
        max_ep_len=1000,
        logger_kwargs=logger_kwargs
    )

else:
    print("Skipping training...")
    output_dir = input("Enter an output directory to load a model from:\n")

_, get_action = load_policy_and_env(output_dir)
env = TestWrapper(ObsWrapper(gym.make('FetchReach-v1')))
run_policy(env, get_action)
