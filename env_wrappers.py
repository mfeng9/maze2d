from numpy.lib.function_base import kaiser
from maze.maze_env import MazeEnv, compute_dense_reward
import maze
import gym
import numpy as np
import pdb


class MazeEnvDict(MazeEnv):
    def __init__(self, **kwargs):
        super(MazeEnvDict, self).__init__(**kwargs)
        # pdb.set_trace()
        # super().__init__(**kwargs)
        maze_obs_space = gym.spaces.Box(
        low=np.array([0, 0]),
        high=np.array(self._walls.shape),
        dtype=np.float32,
        )
        
        self.observation_space = gym.spaces.Dict(
        {
            'observation': maze_obs_space,
            'achieved_goal': maze_obs_space,
            'desired_goal': maze_obs_space,
        }
        )
        
    def step(self, *args, **kwargs):
        o2, r, done, info = super().step(*args, **kwargs)
        ret = {
            'observation': o2,
            'achieved_goal': o2,
            'desired_goal': self.goal
        }
        return ret, r, done, info
    
    def reset(self, *args, **kwargs):
        o2 = super().reset(*args, **kwargs)
        ret = {
            'observation': o2,
            'achieved_goal': o2,
            'desired_goal': self.goal
        }
        return ret
    
    def compute_reward(self, achieved, desired, info=None):
        return compute_dense_reward(achieved, desired)
    

if __name__ == '__main__':
    env_name = 'OneObstacle2'  # Choose one of the environments shown above.
    resize_factor = 1  # Inflate the environment to increase the difficulty.
    if env_name == 'OneObstacle2':
        start = np.array([3, 5], dtype=np.float32)
        goal = np.array([8, 5], dtype=np.float32)
        padd_walls = True
    if env_name == 'middle':
        start = np.array([3, 5], dtype=np.float32)
        goal = np.array([33, 33], dtype=np.float32)
        resize_factor = 5
        padd_walls = True
    elif env_name == 'FourRooms':
        start = np.array([2, 4], dtype=np.float32)
        goal = np.array([10, 8], dtype=np.float32)
        padd_walls = True
    elif env_name == 'Maze11x11':
        start = np.array([2, 1.5], dtype=np.float32)
        goal = np.array([2, 11.5], dtype=np.float32)
        padd_walls = True
    elif env_name == 'Spiral11x11':
        start = np.array([5.6, 5.5])
        goal = np.array([1.5, 6.])
        padd_walls = False

    env = MazeEnvDict(walls=env_name,
                resize_factor=resize_factor,
                # dynamics_noise=np.array([0.5, 0.5]),
                start=start,
                goal=goal,
                # noise_sample_size=1000,
                random_seed=0,
                action_noise=0.0,
                # risk_bound=variant['delta'],
                padd_walls = padd_walls,
                )