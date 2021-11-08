from datetime import datetime
import functools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

import brax
from brax import envs
from brax.training import ppo
#from brax.training import ppo,sac,ppo_sweep
#from brax.training import sumo_sp_debug
#from brax.training import sp_decouple
from brax.io import html
import importlib
import os
import pprint
from brax.experimental.braxlines.common import config_utils
from brax.io import model
from brax.io import html



if __name__ == '__main__':
    env_name = "humanoid_mujoco"  
    # param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'ur5e', 'reacher']
    env_fn = envs.create_fn(env_name=env_name)
    env = env_fn()
    
    train_fn = {
    'humanoid_mujoco': functools.partial(
      ppo.train, num_timesteps = 50000000, log_frequency = 20,
      reward_scaling = 0.1, episode_length = 1000, normalize_observations = True,
      action_repeat = 1, unroll_length = 10, num_minibatches = 32,
      num_update_epochs = 8, discounting = 0.97, learning_rate = 3e-4,
      entropy_cost = 1e-3, num_envs = 2048, batch_size = 1024, seed=1
    )}[env_name]

    
    output_path = './checkpoint'
    
    

    xdata = []
    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
       
        print('eval/episode_reward:',metrics['eval/episode_reward'])
        #print('eval/episode_y_vs_x_reward:',metrics['eval/episode_y_vs_x_reward'])
    process_id = jax.process_index()
    if process_id == 0:
        print('init success')
        print('local device count:',jax.local_device_count())
        print('total device count:',jax.device_count())

    for i in range(1):
       
        times = [datetime.now()]
        inference_fn, params, _ = train_fn(environment_fn=env_fn, progress_fn=progress)
        if process_id == 0:
            print(f'time to jit: {times[1] - times[0]}')
            print(f'time to train: {times[-1] - times[1]}')


        
        #save video
        for ii in range(5):
            jit_env_reset = jax.jit(env.reset)
            jit_env_step = jax.jit(env.step)
            jit_inference_fn = jax.jit(inference_fn)
            rng = jax.random.PRNGKey(seed=ii)
            reset_key, rng = jax.random.split(rng)
            state = jit_env_reset(rng=reset_key)
            qps = []
            num_count = 0
            while not state.done:
                num_count += 1
                qps.append(state.qp)
                tmp_key, rng = jax.random.split(rng)
                act = jit_inference_fn(params, state.obs, tmp_key)
                flag = jnp.array([1])
                state = jit_env_step(state, act)
            html.save_html(
                f'{output_path}/_sweep.html',
                env.sys,
                qps
            )    


