# ##################### #
# Simon Fraser University
# CMPT726 project - deep reinforcement learning for a robotic arm
# ##################### #
import gym
import numpy as np
from numpy.linalg import norm
from spinup import trpo_tf1 as trpo

# FetchPickAndPlace-v1 wrappers

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.full(28, -np.inf)
        high = np.full(28, np.inf)
        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, obs):
        obs = np.concatenate(
            (obs['observation'], obs['desired_goal'])
        )
        return obs

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # === modify reward ===
        #
        # encourage gripper to move towards object, then to move
        # it to the target location
        #
        # [0:3]   -> gripper xyz position
        # [3:6]  -> object position
        # [6:9] -> relative object position to gripper
        # [25:28] -> desired goal position (of object)
        #
        grip_pos = next_state[0:3]
        obj_pos = next_state[3:6]
        rel_obj_pos = next_state[6:9]
        goal_pos = next_state[25:28]
        
        rel_dist = norm(rel_obj_pos)
        rel_goal_dist = norm(goal_pos - obj_pos) / norm(goal_pos)

        reward = -(rel_dist + rel_goal_dist)
        
        return next_state, reward, done, info

env_f = lambda : RewardWrapper(
    ObsWrapper(gym.make('FetchPickAndPlace-v1'))
)

output_dir = input("Enter a output directory path to store the result to:\n")
exp_name = input("Enter an experiment name:\n")

logger_kwargs = {
    'output_dir' : output_dir, 
    'exp_name' : exp_name
}

# https://spinningup.openai.com/en/latest/algorithms/trpo.html
trpo(
    env_f,
    steps_per_epoch=5000,
    epochs=200,
    gamma=0.999,
#    max_ep_len=2500,
    logger_kwargs=logger_kwargs
)
