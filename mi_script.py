from datetime import datetime
import functools
import math
import os
import pprint
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brax.io import html
from brax.experimental.composer import composer
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import evaluators
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.envs.obs_indices import OBS_INDICES
from brax.experimental.braxlines.training import ppo
from brax.experimental.braxlines.vgcrl import evaluators as vgcrl_evaluators
from brax.experimental.braxlines.vgcrl import utils as vgcrl_utils

import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions


'''
首先要定义task and experiment parameters

'''
#env_name = 'ant'  # @param ['ant', 'humanoid', 'halfcheetah', 'uni_ant', 'bi_ant']
for env_name in ['ant', 'humanoid', 'uni_ant', 'bi_ant']:
  print(f'----------env_name:{env_name}-------------')
  obs_indices = 'vel'  # @param ['vel']
  obs_scale = 10.0 #@param{'type': 'number'}
  obs_indices_str = obs_indices
  obs_indices = OBS_INDICES[obs_indices][env_name]

  #@markdown **Experiment Parameters**
  #@markdown See [vgcrl/utils.py](https://github.com/google/brax/blob/main/brax/experimental/braxlines/vgcrl/utils.py)
  evaluate_mi = False # @param{'type': 'boolean'}
  evaluate_lgr = False # @param{'type': 'boolean'}
  #for algo_name in ['gcrl', 'cdiayn', 'diayn', 'diayn_full', 'fixed_gcrl']:
  for algo_name in ['diayn']:
  #algo_name = 'diayn'  # @param ['gcrl', 'cdiayn', 'diayn', 'diayn_full', 'fixed_gcrl']
    print(f'----------algo_name:{algo_name}-------------')
    env_reward_multiplier =   0# @param{'type': 'number'}
    obs_norm_reward_multiplier =   0# @param{'type': 'number'}
    normalize_obs_for_disc = False  # @param {'type': 'boolean'}
    seed =   0# @param {type: 'integer'}
    diayn_num_skills = 8  # @param {type: 'integer'}
    spectral_norm = True  # @param {'type': 'boolean'}
    output_path = 'mi_checkpoint' # @param {'type': 'string'}
    task_name = "" # @param {'type': 'string'}
    exp_name = '' # @param {'type': 'string'}
    if output_path:
      output_path = output_path.format(
        date=datetime.now().strftime('%Y%m%d'))
      task_name = task_name or f'{env_name}_{obs_indices_str}_{obs_scale}'
      exp_name = exp_name or algo_name 
      output_path = f'{output_path}/{task_name}/{exp_name}'
      exist = os.path.exists(output_path)
      if not exist:
        os.makedirs(output_path, exist_ok=True)
    print(f'output_path={output_path}')


    # @title Initialize Brax environment
    visualize = False # @param{'type': 'boolean'}

    # Create baseline environment to get observation specs
    base_env_fn = composer.create_fn(env_name=env_name)
    base_env = base_env_fn()
    #使用composer来创造环境


    #创建判别器 这个应该很重要
    # Create discriminator-parameterized environment
    disc = vgcrl_utils.create_disc_fn(algo_name=algo_name,
                      #dianyn       
                      observation_size=base_env.observation_size,
                      #87
                      obs_indices=obs_indices,
                      #(13,14) 专门对应于ant的观测 暂时不清楚意义
                      scale=obs_scale,
                      #10
                      diayn_num_skills = diayn_num_skills,
                      #8
                      spectral_norm=spectral_norm,
                      #True
                      env=base_env,
                      #ant
                      normalize_obs=normalize_obs_for_disc)()

    '''
    深入到上面那个函数里面去看一下：
    'diayn':
    functools.partial(
        Discriminator,
        q_fn='indexing_mlp',
        z_size=diayn_num_skills,
        obs_indices=obs_indices,
        q_fn_params=dict(output_size=diayn_num_skills,),
        dist_p='UniformCategorial',
        dist_q='Categorial',
        logits_clip_range=logits_clip_range,
        spectral_norm=spectral_norm,
    ),

    主要就是utils.py里面的判别器函数Discriminator
    # define dist_params_to_dist for q_z_o
    # define dist for p_z
    '''

    extra_params = disc.init_model(rng=jax.random.PRNGKey(seed=seed))
    # define observation_to_dist_params mapping for q_z_o


    env_fn = vgcrl_utils.create_fn(env_name=env_name, wrapper_params=dict(
        disc=disc, env_reward_multiplier=env_reward_multiplier,#0
        obs_norm_reward_multiplier=obs_norm_reward_multiplier, #0
        ))

    #最关键的部分应该都在utils.py里面 包括扩充obs 以及重新计算reward
    #后面如果要扩成多智能体 也是要修改那个部分
    #还有一个关键的部分是在ppo.py里面 对额外的差异最大化loss进行了实现


    eval_env_fn = functools.partial(env_fn, auto_reset=False)

    # make inference functions and goals for LGR metric
    core_env = env_fn()
    params, inference_fn = ppo.make_params_and_inference_fn(
          core_env.observation_size, core_env.action_size,
          normalize_observations=True, extra_params=extra_params)
    inference_fn = jax.jit(inference_fn)
    goals = tfd.Uniform(low=-disc.obs_scale, high=disc.obs_scale).sample(
        seed=jax.random.PRNGKey(0), sample_shape=(10,))

    # Visualize
    if visualize:
      env = env_fn()
      jit_env_reset = jax.jit(env.reset)
      state = jit_env_reset(rng=jax.random.PRNGKey(seed=seed))
      #HTML(html.render(env.sys, [state.qp]))
      html.save_html(
          f'{output_path}/render.html',
          env.sys,
          [state.qp]
      )    



    #@title Training
    num_timesteps_multiplier =   6# @param {type: 'number'}
    ncols = 5 # @param{type: 'integer'}

    tab = logger_utils.Tabulator(
        output_path=f'{output_path}/training_curves.csv',
        append=False)

    # We determined some reasonable hyperparameters offline and share them here.
    n = num_timesteps_multiplier
    ppo_params = experiments.defaults.get_ppo_params(
        env_name, num_timesteps_multiplier, default='ant')
    train_fn = functools.partial(ppo.train, **ppo_params)

    times = [datetime.now()]
    plotpatterns = ['eval/episode_reward', 'losses/disc_loss', 'metrics/lgr',
                'metrics/entropy_all_', 'metrics/entropy_z_', 'metrics/mi_']

    def update_metrics_fn(num_steps, metrics, params):
      if evaluate_mi:
        metrics.update(vgcrl_evaluators.estimate_empowerment_metric(
          env_fn=env_fn, disc=disc,
          inference_fn=inference_fn, params=params,
          # custom_obs_indices = list(range(core_env.observation_size))[:30],
          # custom_obs_scale = obs_scale,
        ))
      if evaluate_lgr:
        metrics.update(vgcrl_evaluators.estimate_latent_goal_reaching_metric( 
          params=params, env_fn=env_fn, disc=disc, inference_fn=inference_fn,
          goals=goals))
      
    progress, plot, _, _ = experiments.get_progress_fn(
        plotpatterns, times, tab=tab, max_ncols=5,
        xlim=[0, train_fn.keywords['num_timesteps']],
        update_metrics_fn = update_metrics_fn,
        #pre_plot_fn = lambda : clear_output(wait=True),
        #post_plot_fn = plt.show
        )

    extra_loss_fns = dict(disc_loss=disc.disc_loss_fn) if extra_params else None
    _, params, _ = train_fn(
        environment_fn=env_fn, progress_fn=progress, extra_params=extra_params,
        extra_loss_fns=extra_loss_fns, seed=seed)
    #clear_output(wait=True)
    #plot(output_path=output_path)
    process_id = jax.process_index()
    if process_id == 0:
      print(f'time to jit: {times[1] - times[0]}')
      print(f'time to train: {times[-1] - times[1]}')

    #save video
    for z_value in range(diayn_num_skills):
    #z_value =   0# @param {'type': 'raw'}
      eval_seed = 0  # @param {'type': 'integer'}

      z = {
          'fixed_gcrl': jnp.ones(disc.z_size) * z_value,
          'gcrl': jnp.ones(disc.z_size) * z_value,
          'cdiayn': jnp.ones(disc.z_size) * z_value,
          'diayn': jax.nn.one_hot(jnp.array(int(z_value)), disc.z_size),
          'diayn_full': jax.nn.one_hot(jnp.array(int(z_value)), disc.z_size),
      }[algo_name] if z_value is not None else None

      env, states = evaluators.visualize_env(
          env_fn=eval_env_fn,
          inference_fn=inference_fn,
          params=params,
          batch_size=0,
          seed = eval_seed,
          reset_args = (z,),
          step_args = (params['normalizer'], params['extra']),
          output_path=output_path,
          output_name=f'video_z_{z_value}',
      )