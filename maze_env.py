#@title Implement the 2D navigation environment and helper functions.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gym
import gym.spaces
import matplotlib
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from copy import deepcopy
import pdb
from goal_conditioned_point_wrapper import GoalConditionedPointWrapper
from utils import plot_all_environments, plot_walls, plot_problem, plot_problem_path

from wall_templates import WALLS

def generate_highway(num_lanes, 
                     num_cars=2,
                     lane_width=2.0,
                      car_width=2.0,
                      min_car_distance=2.0,
                      max_car_distance=6.0,
                      max_distance=10000,
                      seed=None,
                     ):
  if seed is not None:
      np.random.seed(seed)
  walls = np.zeros([int(num_lanes*lane_width), int(max_distance)])
  # randomly generate obstacle with at least one solution
  inc_y_idx = np.random.randint(low=min_car_distance, 
                              high=max_car_distance,
                              size=num_cars, 
                              )
  car_x_idx = np.random.randint(low=0,
                                high=num_lanes,
                                size=num_cars)
  car_width = int(car_width)
  # generate wall
  y_idx = 1
  for x_idx, dy in zip(car_x_idx, inc_y_idx):
    dy = int(dy)
    y_idx += dy
    if y_idx*car_width + car_width < max_distance:
      walls[int(x_idx*lane_width):int(x_idx*lane_width)+car_width,
            int(y_idx*car_width):int(y_idx*car_width)+car_width] = 1
  
  used_distance = y_idx*car_width + car_width
  walls = walls[:, :used_distance+car_width+1]
  padded_walls = np.pad(walls, pad_width=1, mode='constant', constant_values=1)
  # remove wall on destination
  padded_walls[:, -1] = 0
  padded_walls[:, 0] = 0
  return padded_walls, inc_y_idx

def resize_walls(walls, factor):
  """Increase the environment by rescaling.

  Args:
    walls: 0/1 array indicating obstacle locations.
    factor: (int) factor by which to rescale the environment."""
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])
  col_indices = np.array([i for i in range(width) for _ in range(factor)])
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls

def compute_dense_reward(state, goal):
  """negative euclidean distance"""
  batch_mode = True
  if len(state.shape) == 1:
      state = state[None]
      batch_mode = False
  if len(goal.shape) == 1:
      goal = goal[None]
  dist = np.linalg.norm(state - goal, axis=1)
  if batch_mode:
      return -dist[:,np.newaxis] # adhere to imagination format
  return -dist[0]


class MazeEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self,
               walls=None,
               resize_factor=1,
               action_noise=1.0,
               percent_random_change=0.0,
               start=None,
               goal=None,
               noise_sample_size=None,
               random_seed=0,
               dense_reward=True,
               padd_walls=True,
               record_paths=False,
               ):
    """Initialize the point environment.

    Args:
      walls: (str) name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
      action_noise: (float) Standard deviation of noise to add to actions. Use 0
        to add no noise.
      percent_random_change: (float) percentage of map pixels to be flipped (free -> obstacle, obstacle -> free).
      start: start location of the agent.
      goal: goal location of the agent.
      noise_sample_size: number of noisy maps to simulate uncertainty in perception.
      random_seed: random seed to support consistency in tests.
      dense_reward: whether to compute dense reward.
    """
    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    elif walls == 'Highway':
      self._walls, _ = generate_highway(
                 num_lanes=4,
                 car_width=2,
                 num_cars=5,
                 min_car_distance=3,
                 max_car_distance=6,
                 seed=0,
                 )
    else:
      self._walls = WALLS[walls]
    if padd_walls:
      self._walls = np.pad(self._walls, pad_width=1, mode='constant', constant_values=1)
    self._apsp = self._compute_apsp(self._walls)
    self._original_walls = deepcopy(self._walls)
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._action_noise = action_noise
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.array([0.0, 0.0]),
        high=np.array([self._height, self._width]),
        dtype=np.float32)
    self.start = start
    self.goal = goal
    self.percent_random_change = percent_random_change
    self.dense_reward = dense_reward
    self.env_name = walls
    self.padd_walls = padd_walls
    self.paths = []
    self.record_paths = record_paths

    # Use random seed for test purposes.
    np.random.seed(random_seed)
    self.reset()
  
  def create_noisy_environments(self, noise_sample_size, percent_random_change):
    """Create randomly created environments from the existing one"""
    self.noise_sample_size = noise_sample_size
    if self.noise_sample_size is not None:
      self._walls_samples = []
      self._apsp_samples = []
      for _ in range(noise_sample_size):
        self.randomly_change_walls(percent_random_change)
        self._walls_samples.append(self._walls)
        self._apsp_samples.append(self._apsp)

      # Shape [num_samples, wall_size_x, wall_size_y].
      self._walls_samples = np.stack(self._walls_samples)
      # Compute probability of being occupied.
      self._walls_samples_p = np.sum(self._walls_samples, axis=0) / self.noise_sample_size
      # Reset self._walls
      self._walls = deepcopy(self._original_walls)

  def randomly_change_walls(self, percent_random_change=None, verbose=False):
    """randomly change environment for meta learning"""
    if percent_random_change is None:
      percent_random_change = self.percent_random_change
    num_change = np.floor(self._height * self._width * percent_random_change)
    num_change = int(num_change)
    if verbose:
      print('changing amount: {}'.format(num_change))
    self._walls = deepcopy(self._original_walls)
    pos_x = np.random.randint(low=0, high=self._walls.shape[0], size=num_change)
    pos_y = np.random.randint(low=0, high=self._walls.shape[1], size=num_change)
    for x, y in zip(pos_x, pos_y):
      self._walls[x,y] = 1 - self._walls[x,y]
    self._apsp = self._compute_apsp(self._walls)

  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index]],
                     dtype=np.float)
    state += np.random.uniform(size=2)
    assert not self._is_blocked(state)
    return state

  def reset(self, start=None, risk_bound=None):
    if start is not None:
      self.state = start
    elif self.start is not None:
      self.state = self.start.copy()
    else:
      self.state = self._sample_empty_state()
      
    if self.record_paths:
      # pdb.set_trace()
      self.paths.append([self.state.copy()])
    return self.state.copy()

  def _get_distance(self, obs, goal):
    """Compute the shortest path distance.

    Note: This distance is *not* used for training."""
    (i1, j1) = self._discretize_state(obs)
    (i2, j2) = self._discretize_state(goal)
    return self._apsp[i1, j1, i2, j2]
  
  def _is_done(self, obs, goal, threshold_distance=0.5):
    """Determines whether observation equals goal."""
    return np.linalg.norm(obs - goal) < threshold_distance

  def _discretize_state(self, state, resolution=1.0):
    (i, j) = np.floor(resolution * state).astype(np.int)
    # Round down to the nearest cell if at the boundary.
    if i == self._height:
      i -= 1
    if j == self._width:
      j -= 1
    return (i, j)

  def _is_blocked(self, state):
    if not self.observation_space.contains(state):
      return True
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  # def _is_blocked_noise(self, state):
  #   """ Compute probability of being blocked in a state given noisy maps."""
  #   if not self.observation_space.contains(state):
  #     return 1.0
  #   (i, j) = self._discretize_state(state)
  #   return (self._walls_samples_p[i, j])

  def render(self):
      pass

  def step(self, action):
    """action is the desired state change (movement)
    of the point agent, but it may be blocked by a wall"""
    old_state = self.state
    if np.sum(np.abs(self._action_noise)) > 0:
      # print('noise')
      # print(self._action_noise)
      action += np.random.normal(0, self._action_noise)
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state
    
    done = False
    if self.goal is not None:
      done = self._is_done(self.state, self.goal, 0.5)
    ##NOTE: reward is overriden in the goal condition wrapper
    # rew = -1.0 * np.linalg.norm(self.state)
    rew = -1.0
    if done:
      rew = 0.0
    if self.goal is not None and self.dense_reward:
      rew = compute_dense_reward(self.state.copy(), self.goal.copy())
    # Compute travelled distance at the new state.
    env_info = {}
    env_info['Step distance'] = np.sqrt(np.sum((self.state - old_state)**2, -1))
    
    if self.record_paths:
      self.paths[-1].append(self.state.copy())
    return self.state.copy(), rew, done, env_info

  @property
  def walls(self):
    return self._walls

  def _compute_apsp(self, walls):
    (height, width) = walls.shape
    g = nx.Graph()
    # Add all the nodes
    for i in range(height):
      for j in range(width):
        if walls[i, j] == 0:
          g.add_node((i, j))

    # Add all the edges
    for i in range(height):
      for j in range(width):
        for di in [-1, 0, 1]:
          for dj in [-1, 0, 1]:
            if di == dj == 0: continue  # Don't add self loops
            if i + di < 0 or i + di > height - 1: continue  # No cell here
            if j + dj < 0 or j + dj > width - 1: continue  # No cell here
            if walls[i, j] == 1: continue  # Don't add edges to walls
            if walls[i + di, j + dj] == 1: continue  # Don't add edges to walls
            g.add_edge((i, j), (i + di, j + dj))

    # dist[i, j, k, l] is path from (i, j) -> (k, l)
    dist = np.full((height, width, height, width), np.float('inf'))
    for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
      for ((i2, j2), d) in dist_dict.items():
        dist[i1, j1, i2, j2] = d
    return dist


class DubinsMaze(MazeEnv):
  def __init__(self,
               walls=None,
               resize_factor=1,
               action_noise=1.0,
               percent_random_change=0.0,
               start=None,
               goal=None,
               noise_sample_size=None,
               random_seed=0,
               dense_reward=True,
               ):
    super().__init__(walls=walls,
                     resize_factor=resize_factor,
                     action_noise=action_noise,
                     percent_random_change=percent_random_change,
                     start=start,
                     goal=goal,
                     noise_sample_size=noise_sample_size,
                     random_seed=random_seed,
                     dense_reward=dense_reward, 
    )
    ## heading angular vel, acceleration
    self.action_space = gym.spaces.Box(
      low= np.array([-np.pi/4, -1.]),
      high=np.array([np.pi/4, 1.]),
      dtype=np.float32
    )
    ## heading angle, scalar speed
    self.dubins_state = np.array([0., 1.0])
    self.speed_ub = 2.0
    self.speed_lb = 0.0
    self.theta_ub = np.pi/4.
    self.theta_lb = -np.pi/4.

  def reset(self):
    self.dubins_state = np.array([0., 1.0])
    if self.start is not None:
      self.state = self.start.copy()
      return self.state.copy()
    self.state = self._sample_empty_state()
    return self.state.copy()
  
  def step(self, action):
    """action: dubins path action"""
    old_state = np.copy(self.state)
    if np.sum(np.abs(self._action_noise)) > 0:
      # print('noise')
      # print(self._action_noise)
      action += np.random.normal(0, self._action_noise)
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    
    dubin_action = np.copy(action)
    # dubin_action[0] = np.clip(dubin_action[0], self.theta_lb, self.theta_ub)
    self.dubins_state[0] = self.dubins_state[0] + dubin_action[0]
    self.dubins_state[1] = np.clip(self.dubins_state[1] + dubin_action[1],
                                   a_min=self.speed_lb, a_max=self.speed_ub)
    theta, v = self.dubins_state
    action_xy = np.array([v*np.cos(theta), v*np.sin(theta)])
    
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action_xy[axis]
        if not self._is_blocked(new_state):
          self.state = new_state

    done = self._is_done(self.state, self.goal, 0.5)
    ##NOTE: reward is overriden in the goal condition wrapper
    rew = -1.0 * np.linalg.norm(self.state)
    if self.dense_reward:
      rew = compute_dense_reward(self.state.copy(), self.goal.copy())
    # Compute collisions at the new state.
    env_info = {}
    env_info['Step distance'] = np.sqrt(np.sum((self.state - old_state)**2, -1))
    return self.state.copy(), rew, done, env_info
  
class RiskAwareMaze(MazeEnv):
  def __init__(self,
               walls=None,
               resize_factor=1,
               dynamics_noise=[1.0, 1.0],
               percent_random_change=0.0,
               start=None,
               goal=None,
               noise_sample_size:int=None,
               random_seed=None,
               dense_reward=True,
               ):
    """
    dynamics_noise an numpy array or list of the same length as state
    """
    super().__init__(walls=walls,
                     resize_factor=resize_factor,
                     action_noise=0.0,
                     percent_random_change=percent_random_change,
                     start=start,
                     goal=goal,
                     noise_sample_size=noise_sample_size,
                     random_seed=random_seed,
                     dense_reward=dense_reward,
                     )
    self._dynamics_noise = dynamics_noise
    self._noise_sample_size = noise_sample_size
    
  def step(self, action):
    sn, reward, done, info = super().step(action)
    
    ## Monte-Carlo compute step-wise risk
    samples = np.random.normal(loc=sn, 
                                scale=self._dynamics_noise, 
                                size=[self._noise_sample_size, len(sn)])
    count_blocked_samples = 0
    for sn_sample in samples:
      if self._is_blocked(sn_sample):
        count_blocked_samples += 1
    
    stepwise_risk = count_blocked_samples*1.0/self._noise_sample_size
    info['collision'] = stepwise_risk
    info['risk'] = stepwise_risk
    return sn, reward, done, info


class RiskConditionedMaze(MazeEnv):
  def __init__(self,
               walls=None,
               resize_factor=1,
               dynamics_noise=[1.0, 1.0],
               percent_random_change=0.0,
               start=None,
               goal=None,
               noise_sample_size:int=None,
               random_seed=None,
               dense_reward=True,
               risk_bound=0.5,
               ):
    """
    dynamics_noise an numpy array or list of the same length as state
    """
    self._dynamics_noise = dynamics_noise
    self._noise_sample_size = noise_sample_size
    self._risk_bound = risk_bound
    self._allocated_risk = 0.0
    super().__init__(walls=walls,
                     resize_factor=resize_factor,
                     action_noise=0.0,
                     percent_random_change=percent_random_change,
                     start=start,
                     goal=goal,
                     noise_sample_size=noise_sample_size,
                     random_seed=random_seed,
                     dense_reward=dense_reward,
                     )
    maze_obs_space = gym.spaces.Box(
      low=np.array([0, 0]),
      high=np.array(self._walls.shape),
      dtype=np.float32,
    )
    budget_space = gym.spaces.Box(
      low=np.array([0.0]),
      high=np.array([1.0]),
    )
    self.observation_space = gym.spaces.Dict(
      {
        'observation': maze_obs_space,
        'risk_budget': budget_space,
        'risk_bound': budget_space,
      }
    )
    self.observation_space = maze_obs_space

  def step(self, action, train=True):
    sn, reward, done, info = super().step(action)
    if train:
        ## Monte-Carlo compute step-wise risk
        samples = np.random.normal(loc=sn,
                                    scale=self._dynamics_noise,
                                    size=[self._noise_sample_size, len(sn)])
        count_blocked_samples = 0
        for sn_sample in samples:
          if self._is_blocked(sn_sample):
            count_blocked_samples += 1

        stepwise_risk = count_blocked_samples*1.0/self._noise_sample_size
    else:
        # Skip expensive risk computing at test time.
        stepwise_risk = 0
    # Compute accumulated risk.
    self._allocated_risk = self._allocated_risk + (1.0 - self._allocated_risk) * stepwise_risk

    # TODO: set allocated risk to none 0.
    info['collision'] = stepwise_risk
    info['risk'] = stepwise_risk
    info['risk_bound'] = self._risk_bound
    info['allocated_risk'] = 0.0

    sn_dict = {
      'observation': sn,
      'risk_bound': self._risk_bound,
      'allocated_risk': self._allocated_risk,
    }
    return sn_dict, reward, done, info

  def set_risk_bound(self, risk_bound):
    self._risk_bound = risk_bound

  def compute_collision(self, state):
      samples = np.random.normal(loc=state,
                                 scale=self._dynamics_noise,
                                 size=[self._noise_sample_size, len(state)])
      count_blocked_samples = 0
      for sn_sample in samples:
        if self._is_blocked(sn_sample):
          count_blocked_samples += 1

      stepwise_risk = count_blocked_samples*1.0/self._noise_sample_size
      return stepwise_risk

  def reset(self, risk_bound=None):
    self._allocated_risk = 0.0
    if risk_bound is None:
        # Reset risk bound randomly.
        self._risk_bound = np.random.uniform(0.05, 0.33)
        # pass
    else:
        self._risk_bound = risk_bound
    sn_dict = {
      'observation': super().reset(),
      'risk_bound': self._risk_bound,
      'allocated_risk': self._allocated_risk,
    }
    return sn_dict


class RiskAwareDubinsMaze(DubinsMaze):
  def __init__(self,
               walls=None,
               resize_factor=1,
               dynamics_noise=[1.0, 1.0],
               percent_random_change=0.0,
               start=None,
               goal=None,
               noise_sample_size:int=None,
               random_seed=None,
               dense_reward=True,
               risk_bound=0.5,
               ):
    """
    dynamics_noise an numpy array or list of the same length as state
    """
    self._dynamics_noise = dynamics_noise
    self._noise_sample_size = noise_sample_size
    self._risk_bound = risk_bound
    self._allocated_risk = 0.0
    super().__init__(walls=walls,
                     resize_factor=resize_factor,
                     action_noise=0.0,
                     percent_random_change=percent_random_change,
                     start=start,
                     goal=goal,
                     noise_sample_size=noise_sample_size,
                     random_seed=random_seed,
                     dense_reward=dense_reward,
                     )    
    maze_obs_space = gym.spaces.Box(
      low=np.array([0, 0]),
      high=np.array(self._walls.shape),
      dtype=np.float32,
    )
    budget_space = gym.spaces.Box(
      low=np.array([0.0]),
      high=np.array([1.0]),
    )
    self.observation_space = gym.spaces.Dict(
      {
        'observation': maze_obs_space,
        'risk_budget': budget_space,
        'risk_bound': budget_space,
      }
    )
    self.observation_space = maze_obs_space
    
  def step(self, action, train=True):
    sn, reward, done, info = super().step(action)
    if train:
        ## Monte-Carlo compute step-wise risk
        samples = np.random.normal(loc=sn,
                                    scale=self._dynamics_noise,
                                    size=[self._noise_sample_size, len(sn)])
        count_blocked_samples = 0
        for sn_sample in samples:
          if self._is_blocked(sn_sample):
            count_blocked_samples += 1

        stepwise_risk = count_blocked_samples*1.0/self._noise_sample_size
    else:
        # Skip expensive risk computing at test time.
        stepwise_risk = 0
    # Compute accumulated risk.
    self._allocated_risk = self._allocated_risk + (1.0 - self._allocated_risk) * stepwise_risk
    
    info['collision'] = stepwise_risk
    info['risk'] = stepwise_risk
    info['risk_bound'] = self._risk_bound
    info['allocated_risk'] = 0.0
    
    sn_dict = {
      'observation': sn,
      'risk_bound': self._risk_bound,
      'allocated_risk': self._allocated_risk,
    }
    return sn_dict, reward, done, info
  
  def set_risk_bound(self, risk_bound):
    self._risk_bound = risk_bound
  
  def reset(self, risk_bound=None):
    self._allocated_risk = 0.0
    if risk_bound is None:
        # Reset risk bound randomly.
        self._risk_bound = np.random.uniform(0.05, 0.33)
        # pass
    else:
        self._risk_bound = risk_bound
    sn_dict = {
      'observation': super().reset(),
      'risk_bound': self._risk_bound,
      'allocated_risk': self._allocated_risk,
    }
    return sn_dict
    
  # def reset(self):
  #   o0 = super().reset()
  #   info = {
  #     'collision': 0.0,
  #     'risk': 0.0,
  #     'risk_bound': 0.0,
  #   }
  #   return o0, info
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-v', '--visualize', action='store_true', help='Visualize all environments.')
  parser.add_argument('-e', '--env-name', default='TwoRooms', help='Select an environment.')
  args = parser.parse_args()

  if args.visualize:
    plot_all_environments(WALLS)

  max_episode_steps = 20
  env_name = args.env_name
  env_name = 'middle_block'  # Choose one of the environments shown above.
  resize_factor = 1  # Inflate the environment to increase the difficulty.

  start = np.array([2, 5], dtype=np.float32)
  goal = np.array([8, 5], dtype=np.float32)

  env = MazeEnv(env_name,
          resize_factor,
          start=start,
          goal=goal)
  env.reset()
  
  path = [env.start]
  for i in range(100):
    at = env.action_space.sample()
    st = env.step(at)
    path.append(st[0])
  
  plot_problem_path(env, path, show_start_goal=True, filepath='path.png')
  genv = GoalConditionedPointWrapper(env)
  genv.reset()
  print('DONE')
