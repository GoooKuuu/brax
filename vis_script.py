from datetime import datetime
import functools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from brax import jumpy as jp
import brax
from brax import envs
from brax.training import ppo
from brax.io import html
import importlib
import os
import pprint
from brax.experimental.braxlines.common import config_utils
from brax.io import model
from brax.io import html
from brax.experimental.braxlines.common import logger_utils

if __name__ == '__main__':
    env_name = "humanoid_mujoco"  
    # param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'ur5e', 'reacher']
    env_fn = envs.create_fn(env_name=env_name)
    env = env_fn()
    print('observation size:',env.observation_size)
    print('action size:',env.action_size)
    state = env.reset(rng=jp.random_prngkey(seed=0))
    output_dir = './vis'
    html.save_html(
            f'{output_dir}/render.html',
            env.sys,
            [state.qp]
        )    