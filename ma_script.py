from datetime import datetime
import functools
import os
import pprint
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from brax.io import html
from brax.experimental.composer import composer
from brax.experimental.composer.training import mappo
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import evaluators
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.training import ppo
from brax.experimental.braxlines import experiments

output_path = 'checkpoint'


def show_env(env, mode):
  if mode == 'print_obs':
    pprint.pprint(composer.get_env_obs_dict_shape(env.unwrapped))
  elif mode == 'print_sys':
    pprint.pprint(env.unwrapped.composer.metadata.config_json)
  elif mode == 'print_step':
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    state0 = jit_env_reset(rng=jax.random.PRNGKey(seed=0))
    state1 = jit_env_step(state0, jnp.zeros((env.action_size,)))
    print(f'obs0={state0.obs.shape}') 
    print(f'obs1={state1.obs.shape}') 
    print(f'rew0={state0.reward}') 
    print(f'rew1={state1.reward}')
    print(f'action0={(env.action_size,)}') 
  else:
    jit_env_reset = jax.jit(env.reset)
    state = jit_env_reset(rng=jax.random.PRNGKey(seed=0))
    html.save_html(
                f'{output_path}/render.html',
                env.sys,
                [state.qp]
            )    

if __name__ == '__main__':


    env_list = composer.list_env()
    print(f'{len(env_list)} registered envs, e.g. {env_list[:]}...')

    env_name = 'sumo' # @param ['squidgame', 'sumo', 'follow', 'chase', 'pro_ant_run', 'ant_run', 'ant_chase', 'ant_push']
    env_params = None # @param{'type': 'raw'}
    mode = 'viewer'# @param ['print_step', 'print_obs', 'print_sys', 'viewer']
    if output_path:
        output_path = f'{output_path}/{datetime.now().strftime("%Y%m%d")}' 
        output_path = f'{output_path}/{env_name}'
        exist = os.path.exists(output_path)
        if not exist:
            os.makedirs(output_path, exist_ok=True)
        print(f'Saving outputs to {output_path}')

    # check supported params
    env_params = env_params or {}
    supported_params, support_kwargs = composer.inspect_env(env_name=env_name)
    assert support_kwargs or all(
        k in supported_params for k in env_params
    ), f'invalid {env_params} for {supported_params}' 

    times = [datetime.now()]
    # create env
    env_fn = composer.create_fn(env_name=env_name,
    **(env_params or {}))
    env = env_fn()
    process_id = jax.process_index()
    if process_id == 0:
        print('init success')
        print('local device count:',jax.local_device_count())
        print('total device count:',jax.device_count())
        print('env observation:',env.observation_size)
        
    show_env(env, mode)
    times.append(datetime.now())
    print(f'time to init: {times[-1] - times[-2]}')

