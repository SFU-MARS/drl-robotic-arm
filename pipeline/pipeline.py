import gym
import numpy as np
from spinup.utils.test_policy import load_policy_and_env, run_pipeline

class ObsWrapper(gym.ObservationWrapper):
    """
    Wrapper to modify observations returned by Fetch environments.
    The observation state defined here is meant comply with the
    modified PPO algorithm in the 'fetchReachV1Code' folder.
    """
    def __init__(self, env):
        super().__init__(env)
        low = np.full(31, -np.inf)
        high = np.full(31, np.inf)
        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, obs):
        obs = np.concatenate(
            (obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        )
        return obs

env = ObsWrapper(gym.make('FetchPickAndPlace-v1'))

# load models
_, get_reach_action = load_policy_and_env('./pretrainedNetwork/gripperEnabled-v02')
# _, dummy_action = load_policy_and_env('../../reach_block_output_model')

policy_dict = {
    'reach' : get_reach_action,
    # 'dummy' : dummy_action
}

run_pipeline(env, policy_dict)
