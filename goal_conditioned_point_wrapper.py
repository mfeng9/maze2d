import gym
import numpy as np

class GoalConditionedPointWrapper(gym.Wrapper):
  """Wrapper that appends goal to state produced by environment."""


  def __init__(self,
               env,
               prob_constraint=0.8,
               min_dist=0,
               max_dist=4,
               threshold_distance=0.5):
    """Initialize the environment.
    sets the environment with a sampled goal
    Args:
      env: an environment.
      prob_constraint: (float) Probability that the distance constraint is
        followed after resetting.
      min_dist: (float) When the constraint is enforced, ensure the goal is at
        least this far from the initial state.
      max_dist: (float) When the constraint is enforced, ensure the goal is at
        most this far from the initial state.
      threshold_distance: (float) States are considered equivalent if they are
        at most this far away from one another.
    """
    self._threshold_distance = threshold_distance
    self._prob_constraint = prob_constraint
    self._min_dist = min_dist
    self._max_dist = max_dist
    super(GoalConditionedPointWrapper, self).__init__(env)
    self.observation_space = gym.spaces.Dict({
        'observation': env.observation_space,
        'goal': env.observation_space,
    })

  def _normalize_obs(self, obs):
    return np.array([
        obs[0] / float(self.env._height),
        obs[1] / float(self.env._width)
    ])

  def reset(self):
    goal = None
    obs = None
    count = 0
    while goal is None:
      obs = self.env.reset()
      if self.env.goal is None:
        (obs, goal) = self._sample_goal(obs)
        count += 1
        if count > 1000:
          print('WARNING: Unable to find goal within constraints.')
      else:
        goal = self.env.goal.copy()
    self._goal = goal
    return {'observation': self._normalize_obs(obs),
            'goal': self._normalize_obs(self._goal)}

  def step(self, action):
    obs, _, _, _ = self.env.step(action)
    rew = -1.0
    done = self._is_done(obs, self._goal)
    return {'observation': self._normalize_obs(obs),
            'goal': self._normalize_obs(self._goal)}, rew, done, {}

  def set_sample_goal_args(self, prob_constraint=None,
                           min_dist=None, max_dist=None):
    assert prob_constraint is not None
    assert min_dist is not None
    assert max_dist is not None
    assert min_dist >= 0
    assert max_dist >= min_dist
    self._prob_constraint = prob_constraint
    self._min_dist = min_dist
    self._max_dist = max_dist

  def _is_done(self, obs, goal):
    """Determines whether observation equals goal."""
    return np.linalg.norm(obs - goal) < self._threshold_distance

  def _sample_goal(self, obs):
    """Sampled a goal state."""
    if np.random.random() < self._prob_constraint:
      return self._sample_goal_constrained(obs, self._min_dist, self._max_dist)
    else:
      return self._sample_goal_unconstrained(obs)

  def _sample_goal_constrained(self, obs, min_dist, max_dist):
    """Samples a goal with dist min_dist <= d(obs, goal) <= max_dist.

    Args:
      obs: observation (without goal).
      min_dist: (int) minimum distance to goal.
      max_dist: (int) maximum distance to goal.
    Returns:
      obs: observation (without goal).
      goal: a goal state.
    """
    (i, j) = self.env._discretize_state(obs)
    mask = np.logical_and(self.env._apsp[i, j] >= min_dist,
                          self.env._apsp[i, j] <= max_dist)
    mask = np.logical_and(mask, self.env._walls == 0)
    candidate_states = np.where(mask)
    num_candidate_states = len(candidate_states[0])
    if num_candidate_states == 0:
      return (obs, None)
    goal_index = np.random.choice(num_candidate_states)
    goal = np.array([candidate_states[0][goal_index],
                     candidate_states[1][goal_index]],
                    dtype=np.float)
    goal += np.random.uniform(size=2)
    dist_to_goal = self.env._get_distance(obs, goal)
    assert min_dist <= dist_to_goal <= max_dist
    assert not self.env._is_blocked(goal)
    return (obs, goal)

  def _sample_goal_unconstrained(self, obs):
    """Samples a goal without any constraints.

    Args:
      obs: observation (without goal).
    Returns:
      obs: observation (without goal).
      goal: a goal state.
    """
    return (obs, self.env._sample_empty_state())

  @property
  def max_goal_dist(self):
    apsp = self.env._apsp
    return np.max(apsp[np.isfinite(apsp)])
