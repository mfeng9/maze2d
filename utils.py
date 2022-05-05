import matplotlib
from matplotlib import markers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from copy import deepcopy
import torch
import pdb


def compute_risk(collisions):
  overall_risk = 0.0
  for rb in collisions[::-1]:
    overall_risk = rb + (1 - rb) * overall_risk
  return overall_risk

def plot_walls(walls, fig=None, ax=None):
  walls = deepcopy(walls).T
  (height, width) = walls.shape
  if fig is None:
    fig, ax = plt.subplots()
  scaling = np.max([height, width])
  for (i, j) in zip(*np.where(walls)):
    x = np.array([j, j+1]) / float(scaling)
    y0 = np.array([i, i]) / float(scaling)
    y1 = np.array([i+1, i+1]) / float(scaling)
    ax.fill_between(x, y0, y1, color='grey')
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  ax.axis('equal')
  return scaling

def plot_all_environments(walls_all):
  plt.figure(figsize=(16, 7))
  for index, (name, walls) in enumerate(walls_all.items()):
    plt.subplot(3, 7, index + 1)
    plt.title(name)
    plot_walls(walls)
  plt.subplots_adjust(wspace=0.1, hspace=0.2)
  plt.suptitle('Navigation Environments', fontsize=20)
  plt.savefig('all_environments.png', dpi=300)
  print('generated fig: all_environments.png')
  # plt.show()

def plot_problem(env, show_start_goal=True):
  goal = deepcopy(env.goal)
  start = deepcopy(env.start)
  walls = deepcopy(env._walls)
  scaling = plot_walls(env._walls)
  (height, width) = walls.shape
  if show_start_goal:
    if start is not None:
      start[0] = start[0]/scaling
      start[1] = start[1]/scaling
      plt.scatter([start[0]], [start[1]], marker='+',
                  color='red', s=200, label='start')
    if goal is not None:
      goal[0] = goal[0]/scaling
      goal[1] = goal[1]/scaling
      plt.scatter([goal[0]], [goal[1]], marker='*',
                  color='green', s=200, label='goal')

    plt.legend()
    plt.title(env.env_name)
  return scaling
  
def plot_problem_path(env, path, 
                      show_start_goal=True,
                      subgoals=[], 
                      filepath=None, 
                      log=True,
                      axis_off=False,
                      pass_through=False,
                      alpha=0.4,
                      fill_color=None,
                      **kwargs):
  goal = deepcopy(env.goal)
  start = deepcopy(env.start)
  walls = deepcopy(env._walls)
  # scaling = plot_walls(env._walls)
  #########################################
  walls = walls.T
  (height, width) = walls.shape
  scaling = np.max([height, width])
  for (i, j) in zip(*np.where(walls)):
    x = np.array([j, j+1]) / float(scaling)
    y0 = np.array([i, i]) / float(scaling)
    y1 = np.array([i+1, i+1]) / float(scaling)
    plt.fill_between(x, y0, y1, color='grey')
  ############################################
  (height, width) = walls.shape
  ## TODO: not sure 0-->width and 1-->height
  start[0] = start[0]/scaling
  start[1] = start[1]/scaling
  goal[0] = goal[0]/scaling
  goal[1] = goal[1]/scaling
  
  if isinstance(path, list):
    path = np.vstack(path)
  
  if len(subgoals) > 0:
    subgoals = np.asarray(subgoals)/scaling
    plt.plot(subgoals[:,0], 
            subgoals[:,1], 
            marker='D', 
            color='b',
            linestyle='-',
            # markersize=100, 
            label='subgoals', 
            alpha=alpha)  
  
  path_x = path[:,0]/scaling
  path_y = path[:,1]/scaling
  
  if 'color_path_by_attribute' in kwargs.keys():
    v_attribute = kwargs['color_path_by_attribute']
    v_function = np.array(kwargs[v_attribute])
    cmap = cm.get_cmap('rainbow')
    v_map = np.array(v_function) - np.min(v_function)
    v_map = v_map / np.max(v_map)
    for i in range(len(v_function)):
        plt.plot(path_x[i:i+2], path_y[i:i+2], color=cmap(v_map[i]), marker='o', linestyle='-',alpha=0.5)
        # plt.plot(path_x[i], path_y[i], '-', color=cmap(v_map[i]), alpha=0.5)
  else:
    # plt.plot(path_x, path_y, 'c-o', alpha=0.3)
    plt.plot(path_x, path_y, 
          color='blue',
          marker='o',
          linestyle='-',
          alpha=0.1)
  
  if fill_color is not None:
    empty_states = []
    empty_states.append(np.array([0,0])/scaling)
    empty_states.append(np.array([0,env._walls.shape[1]])/scaling)
    empty_states.append(np.array([env._walls.shape[0],env._walls.shape[1]])/scaling)
    empty_states.append(np.array([env._walls.shape[0],0])/scaling)
    empty_states = np.asarray(empty_states)  
    plt.fill(empty_states[:,0], empty_states[:,1], color=fill_color, alpha=0.6)

  if show_start_goal:
    plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')
    plt.scatter([start[0]], [start[1]], marker='+',
                color='red', s=200, label='start')
    plt.legend()
    plt.title(env.env_name)
    
    
  if not pass_through:  
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('equal')
    
    if axis_off:
      plt.axis('off')
      # plt.tight_layout()
      # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
      plt.gca().set_axis_off()
      plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                  hspace = 0, wspace = 0)
      plt.margins(0,0)
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    if filepath is not None:
      plt.savefig(filepath, dpi=320, bbox_inches='tight')
      plt.close()  
      if log:
        print('generated figure: {}'.format(filepath))  

def plot_polo_value_function(env, model, attribute='mean',
                             scaling=1.0, plot_edge=True, 
                             show_start_goal=True,
                             axis_off=False,
                             filepath=None, verbose=True):
    """goal state has 0 value"""
    # fig, ax = plt.subplots()
    # pdb.set_trace()
    empty_states = []
    ## add four corners
    # pdb.set_trace()
    if env.padd_walls:
      empty_states.append(np.array([1,1]))
      empty_states.append(np.array([1,env._walls.shape[1]-1]))
      empty_states.append(np.array([env._walls.shape[0]-1,1]))
      empty_states.append(np.array([env._walls.shape[0]-1,env._walls.shape[1]-1]))
    else:
      empty_states.append(np.array([0,0]))
      empty_states.append(np.array([0,env._walls.shape[1]-1]))
      empty_states.append(np.array([env._walls.shape[0]-1,0]))
      empty_states.append(np.array([env._walls.shape[0]-1,env._walls.shape[1]-1]))
    for _ in range(1000):
        empty_states.append(env._sample_empty_state())
    empty_states = np.asarray(empty_states)
    empty_states_t = torch.from_numpy(empty_states).cuda().float()

    val_mean, val_var = model( empty_states_t )
    vis_value = None
    if attribute == 'mean':
        vis_value = val_mean
    elif attribute == 'var':
        vis_value = val_var
    else:
        print('attribute must be either (1) mean, (2) var')
        return
    if torch.is_tensor(vis_value):
        vis_value = vis_value.detach().cpu().numpy()
    cmap = cm.get_cmap('rainbow')
    # vis_value = vis_value - np.min(vis_value)
    # vis_value = vis_value / np.max(vis_value)
    vis_value = vis_value.squeeze()
    scaling = plot_problem(env, show_start_goal=show_start_goal)
    # q_color = cmap(val_var) 

    levels = np.linspace(np.min(vis_value), np.max(vis_value), 100)
    if len(np.unique(vis_value)) <= 1:
        levels = None
    try:
        tcf = plt.tricontourf(empty_states[:,0]/scaling,
                    empty_states[:,1]/scaling,
                    vis_value,
                    cmap='rainbow',
                    # N=1,
                    levels=levels,
                    alpha=0.7,
                    )
    except:
        pdb.set_trace()
    plt.colorbar(tcf)
    if axis_off:
      plt.axis('off')
      # plt.tight_layout()
      # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
      plt.gca().set_axis_off()
      plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                  hspace = 0, wspace = 0)
      plt.margins(0,0)
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if filepath is not None:
        plt.savefig(filepath, dpi=320)
        plt.close()  
        if verbose:
            print('generated figure: {}'.format(filepath))  
            
def plot_polo_function(env, function,
                             scaling=1.0, 
                             filepath=None, 
                             verbose=True):
    """goal state has 0 value"""
    # fig, ax = plt.subplots()
    empty_states = []
    for _ in range(1000):
        empty_states.append(env._sample_empty_state())
    empty_states = np.asarray(empty_states)
    empty_states_t = torch.from_numpy(empty_states).cuda().float()

    vis_value = function( empty_states_t )
    cmap = cm.get_cmap('rainbow')
    
    vis_value = vis_value.squeeze()
    scaling = plot_problem(env)
    # q_color = cmap(val_var) 
    tcf = plt.tricontourf(empty_states[:,0]/scaling,
                empty_states[:,1]/scaling,
                vis_value,
                cmap='rainbow',
                # N=1,
                alpha=0.7,
                )
    plt.colorbar(tcf)
    if filepath is not None:
        plt.savefig(filepath, dpi=320)
        plt.close()  
        if verbose:
            print('generated figure: {}'.format(filepath))  

def compute_execution_rsk(collisions):
  overall_risk = 0
  for rb in collisions[::-1]:
      overall_risk = rb + (1 - rb) * overall_risk
  return overall_risk

def plot_problem_paths(env, paths, risk_bounds, fig_dir, show_fig=True):
  goal = env.goal
  start = env.start
  walls = env._walls
  (height, width) = walls.shape
  ## TODO: not sure 0-->width and 1-->height
  start[0] = start[0]/width
  start[1] = start[1]/height
  goal[0] = goal[0]/width
  goal[1] = goal[1]/height

  fig, ax = plt.subplots()
  plot_walls(env._walls, fig, ax)
  ax.scatter([start[0]], [start[1]], marker='*',
            color='green', s=200, label='start')
  ax.scatter([goal[0]], [goal[1]], marker='+',
            color='red', s=200, label='goal')

  # Obtain psulu paths.
  psulu_paths = [[[2.0, 5.0], [2.969107, 4.0], [3.969107, 3.020893], [4.969107, 3.020893], [5.969107, 3.020893], [6.969107, 3.020893], [7.6608406, 4.020893], [8.6608406, 4.6608406]],
                 [[2.0, 5.0], [2.8406971, 4.0], [3.8406971, 3.1493029], [4.8406971, 3.1493029], [5.8406971, 3.1493029], [6.8406971, 3.1493029], [7.6608406, 4.1493029], [8.6608406, 4.6608406]],
                 [[2.0, 5.0], [2.7585149, 4.0], [3.7585149, 3.2314851], [4.7585149, 3.2314851], [5.7585149, 3.2314851], [6.7585149, 3.2314851], [7.7585149, 4.2314851], [8.6608406, 4.6608406]],
                 ]
  psulu_paths = np.array(psulu_paths)
  # Compute risk for psulu paths.
  for psulu_path in psulu_paths:
    psulu_path_risk = [env.compute_collision(state) for state in psulu_path]
    psulu_er = compute_execution_rsk(psulu_path_risk)
    print('Psulu execution risk', psulu_er)

  colors = ['cyan', 'magenta', 'orange']
  for i, path_dict in enumerate(paths):
    path = path_dict['observations']
    next_path = path_dict['next_observations']
    collisions = [info['collision'] for info in path_dict['env_infos']]
    er = compute_execution_rsk(collisions)
    if isinstance(path, list):
      path = np.vstack(path)
      next_path = np.vstack(next_path)
    path = np.vstack((path, next_path[-1:]))
    # import IPython; IPython.embed()
    dist = np.sum(np.sqrt(np.sum((path[1:] - path[:-1]) ** 2, -1)))
    path_x = path[:,0]/width
    path_y = path[:,1]/height

    path_x_psulu = psulu_paths[i,:,0] / width
    path_y_psulu = psulu_paths[i,:,1] / height
    print('Execution Risk', er)
    ax.plot(path_x, path_y, 'o-', color=colors[i], alpha=0.3, label='Delta: {}, Ours.'.format(risk_bounds[i]))
    ax.plot(path_x_psulu, path_y_psulu, 's-', color=colors[i], alpha=0.3, label='Delta: {}, IRA.'.format(risk_bounds[i]))

  ax.legend()
  ax.set_title(env.env_name)
  fig.savefig(fig_dir+'/policy_vis.png', dpi=320)
  if show_fig:
    plt.show()
  
def plot_value_function(agent, normalized=False, scaling=1.0, plot_edge=True):
    """goal state has 0 value"""
    # visualize value functions
    v_function = []
    coords = deepcopy(agent.model.states.get_data())
    for i in range(len(coords)):
        v_function.append(agent.get_value(i))
    
    if normalized:
        scaling_factor = np.max(coords)
    # else scaling != 1.0:
    #     scaling_factor = scaling
    
    coords = coords/scaling_factor
    cmap = cm.get_cmap('rainbow')
    v_map = np.array(v_function) / np.max(v_function)
    for i in range(len(v_function)):
        plt.plot(coords[i,0], coords[i,1], 'o', color=cmap(v_map[i]), alpha=0.5)

    if plot_edge:
        for eg in agent.model.DG.edges:
            edge_coords = np.array([coords[eg[0]], coords[eg[1]]])
            plt.plot(edge_coords[:,0], edge_coords[:,1], color='b', alpha=0.5, linestyle='-')
    return v_function

def plot_maze_value_function(agent, 
    env, 
    plot_edge=True, 
    filepath=None, 
    show_start_goal=True,
    log=True, 
    annotate_node_id=True,
    axis_off=False,):
    scaling_factor = plot_problem(env, show_start_goal=show_start_goal)
    
    v_function = []
    coords = deepcopy(agent.model.states.get_data())
    for i in range(len(coords)):
        v_function.append(agent.get_value(i))
    coords = coords/scaling_factor
    
    cmap = cm.get_cmap('rainbow')
    v_map = np.array(v_function) - np.min(v_function)
    v_map = v_map / np.max(v_map)
    for i in range(len(v_function)):
        if annotate_node_id:
            plt.text(coords[i,0], coords[i,1]+0.05, '{}'.format(i))
            # plt.annotate(text='{}'.format(i), xy=(coords[i,0], coords[i,1]),
            #              xycoords='data',
            #             xytext=(coords[i,0], coords[i,1]+0.05))
        plt.plot(coords[i,0], coords[i,1], 'o', color=cmap(v_map[i]), alpha=0.5)
    
    if plot_edge:
        for eg in agent.model.DG.edges:
            edge_coords = np.array([coords[eg[0]], coords[eg[1]]])
            plt.plot(edge_coords[:,0], edge_coords[:,1], color='b', alpha=0.5, linestyle='-')
            
    if axis_off:
      plt.axis('off')
      # plt.tight_layout()
      # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
      plt.gca().set_axis_off()
      plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                  hspace = 0, wspace = 0)
      plt.margins(0,0)
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if filepath is not None:
        plt.savefig(filepath, dpi=320)
        plt.close()  
        if log:
            print('generated figure: {}'.format(filepath))  
    return v_function
  
  
def plot_maze_meta_querying_function(agent, 
    env, 
    plot_edge=True, 
    filepath=None, 
    log=True, annotate_node_id=True):
    scaling_factor = plot_problem(env)
    
    v_function = []
    v_count = []
    coords = deepcopy(agent.model.states.get_data())
    for i in range(len(coords)):
        v_function.append(agent.get_value(i))
        v_count.append(agent.model.DG.nodes[i]['visit_count'])
    coords = coords/scaling_factor
    
    
    cmap = cm.get_cmap('rainbow')
    v_count = np.array(v_count)
    v_function = np.array(v_function)
    v_function = v_function + float(agent.count_based_bonus) / np.sqrt(v_count)
    v_map = np.array(v_function) - np.min(v_function)
    v_map = v_map / np.max(v_map)
    for i in range(len(v_function)):
        if annotate_node_id:
            plt.text(coords[i,0], coords[i,1]+0.05, '{}'.format(i))
            # plt.annotate(text='{}'.format(i), xy=(coords[i,0], coords[i,1]),
            #              xycoords='data',
            #             xytext=(coords[i,0], coords[i,1]+0.05))
        plt.plot(coords[i,0], coords[i,1], 'o', color=cmap(v_map[i]), alpha=0.5)
    
    if plot_edge:
        for eg in agent.model.DG.edges:
            edge_coords = np.array([coords[eg[0]], coords[eg[1]]])
            plt.plot(edge_coords[:,0], edge_coords[:,1], color='b', alpha=0.5, linestyle='-')
    if filepath is not None:
        plt.savefig(filepath, dpi=320)
        plt.close()  
        if log:
            print('generated figure: {}'.format(filepath))  
    return v_function