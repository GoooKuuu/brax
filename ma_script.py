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
    envs = ['follow','chase','sumo','squidgame']
    for env_name in envs:
        print(f'---------------{env_name}---------------')
        env_list = composer.list_env()
        print(f'{len(env_list)} registered envs, e.g. {env_list[:]}...')

        #env_name = 'sumo' # @param ['squidgame', 'sumo', 'follow', 'chase', 'pro_ant_run', 'ant_run', 'ant_chase', 'ant_push']
        env_params = None # @param{'type': 'raw'}
        mode = 'viewer'# @param ['print_step', 'print_obs', 'print_sys', 'viewer']
        output_path = f'checkpoint'
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


        num_timesteps_multiplier =   3# @param {type: 'number'}
        seed = 0 # @param{type: 'integer'}
        skip_training = False # @param {type: 'boolean'}

        log_path = output_path
        if log_path:
            log_path = f'{log_path}/training_curves.csv'
        tab = logger_utils.Tabulator(output_path=log_path,
            append=False)

        ppo_lib = mappo if env.is_multiagent else ppo
        ppo_params = experiments.defaults.get_ppo_params(
            'ant', num_timesteps_multiplier)
        train_fn = functools.partial(ppo_lib.train, **ppo_params)

        times = [datetime.now()]
        plotpatterns = ['eval/episode_reward', 'eval/episode_score']

        progress, _, _, _ = experiments.get_progress_fn(
            plotpatterns, times, tab=tab, max_ncols=5,
            xlim=[0, train_fn.keywords['num_timesteps']],
            #pre_plot_fn = lambda : clear_output(wait=True),
            post_plot_fn = plt.savefig(f'{output_path}/train.png'))
        if skip_training:
            action_size = (env.group_action_shapes if 
                env.is_multiagent else env.action_size)
            params, inference_fn = ppo_lib.make_params_and_inference_fn(
                env.observation_size, action_size,
                normalize_observations=True)
            inference_fn = jax.jit(inference_fn)
        else:
            inference_fn, params, _ = train_fn(
                environment_fn=env_fn, seed=seed,
                extra_step_kwargs=False, progress_fn=progress)

        times.append(datetime.now())

        print(f'time to jit: {times[1] - times[0]}')
        print(f'time to train: {times[-1] - times[1]}')

        eval_seed = 0  # @param {'type': 'integer'}
        batch_size =  0# @param {type: 'integer'}

        env, states = evaluators.visualize_env(
            env_fn=env_fn, inference_fn=inference_fn,
            params=params, batch_size=batch_size,
            seed = eval_seed, output_path=output_path,
            verbose=True,
        )
        html.save_html(
                    f'{output_path}/episode.html',
                    env.sys,
                    [state.qp for state in states]
                )    

        #print(f'time to init: {times[-1] - times[-2]}')

