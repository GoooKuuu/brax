# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a humanoid to run in the +x direction."""

import brax
from brax import jumpy as jp
from brax.envs import env
from brax.physics import bodies


class Humanoid_mujoco(env.Env):
  """Trains a humanoid to run in the +x direction."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    body = bodies.Body(self.sys.config)
    body = jp.take(body, body.idx[:-1])  # skip the floor body
    self.mass = body.mass.reshape(-1, 1)
    self.inertia = body.inertia

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.01, .01)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.01, .01)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info, jp.zeros(self.action_size))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'reward_impact': zero
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info, action)

    pos_before = state.qp.pos[:-1]  # ignore floor at last index
    pos_after = qp.pos[:-1]  # ignore floor at last index
    com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
    com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
    lin_vel_cost = 12.5 * (com_after[0] - com_before[0]) / self.sys.config.dt
    quad_ctrl_cost = .01 * jp.sum(jp.square(action))
    # can ignore contact cost, see: https://github.com/openai/gym/issues/1541
    quad_impact_cost = jp.float32(0)
    alive_bonus = jp.float32(5)
    reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

    done = jp.where(qp.pos[0, 2] < 0.65, jp.float32(1), jp.float32(0))
    done = jp.where(qp.pos[0, 2] > 2.1, jp.float32(1), done)
    state.metrics.update(
        reward_linvel=lin_vel_cost,
        reward_quadctrl=quad_ctrl_cost,
        reward_alive=alive_bonus,
        reward_impact=quad_impact_cost)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info,
               action: jp.ndarray) -> jp.ndarray:
    """Observe humanoid body position, velocities, and angles."""
    # some pre-processing to pull joint angles and velocities
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    # qpos:
    # Z of the torso (1,)
    # orientation of the torso as quaternion (4,)
    # joint angles (8,)
    qpos = [
        qp.pos[0, 2:], qp.rot[0], joint_angle
    ]

    # qvel:
    # velocity of the torso (3,)
    # angular velocity of the torso (3,)
    # joint angle velocities (8,)
    qvel = [
        qp.vel[0], qp.ang[0], joint_vel
    ]

    # actuator forces
    qfrc_actuator = []
    for act in self.sys.actuators:
      torque = jp.take(action, act.act_index)
      torque = torque.reshape(torque.shape[:-2] + (-1,))
      torque *= jp.repeat(act.strength, act.act_index.shape[-1])
      qfrc_actuator.append(torque)

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * num bodies in the system
    cfrc_ext = [info.contact.vel, info.contact.ang]
    # flatten bottom dimension
    cfrc_ext = [x.reshape(x.shape[:-2] + (-1,)) for x in cfrc_ext]

    # center of mass obs:
    body_pos = qp.pos[:-1]  # ignore floor at last index
    body_vel = qp.vel[:-1]  # ignore floor at last index

    com_vec = jp.sum(body_pos * self.mass, axis=0) / jp.sum(self.mass)
    com_vel = body_vel * self.mass / jp.sum(self.mass)

    v_outer = jp.vmap(lambda a: jp.outer(a, a))
    v_cross = jp.vmap(jp.cross)

    disp_vec = body_pos - com_vec
    com_inert = self.inertia + self.mass.reshape(
        (15, 1, 1)) * ((jp.norm(disp_vec, axis=1)**2.).reshape(
            (15, 1, 1)) * jp.stack([jp.eye(3)] * 15) - v_outer(disp_vec))

    cinert = [com_inert.reshape(-1)]

    square_disp = (1e-7 + (jp.norm(disp_vec, axis=1)**2.)).reshape((15, 1))
    com_angular_vel = (v_cross(disp_vec, body_vel) / square_disp)
    cvel = [com_vel.reshape(-1), com_angular_vel.reshape(-1)]
    return jp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator +
                          cfrc_ext)


_SYSTEM_CONFIG = """
bodies {
  name: "root"
  colliders {
    position {
      z: 0.07
    }
    sphere {
      radius: 0.09
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.0
}
bodies {
  name: "chest"
  colliders {
    position {
      z: 0.12
    }
    sphere {
      radius: 0.11
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 14.0
}
bodies {
  name: "neck"
  colliders {
    position {
      z: 0.175
    }
    sphere {
      radius: 0.1025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.0
}
bodies {
  name: "right_shoulder"
  colliders {
    position {
      z: -0.14
    }
    rotation {
    }
    capsule {
      radius: 0.045
      length: 0.27
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.5
}
bodies {
  name: "right_elbow"
  colliders {
    position {
      z: -0.12
    }
    rotation {
    }
    capsule {
      radius: 0.04
      length: 0.215
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "right_wrist"
  colliders {
    position {
      z: -0.258947
    }
    sphere {
      radius: 0.04
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.5
}
bodies {
  name: "left_shoulder"
  colliders {
    position {
      z: -0.14
    }
    rotation {
    }
    capsule {
      radius: 0.045
      length: 0.27
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.5
}
bodies {
  name: "left_elbow"
  colliders {
    position {
      z: -0.12
    }
    rotation {
    }
    capsule {
      radius: 0.04
      length: 0.215
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "left_wrist"
  colliders {
    position {
      z: -0.258947
    }
    sphere {
      radius: 0.04
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.5
}
bodies {
  name: "right_hip"
  colliders {
    position {
      z: -0.21
    }
    rotation {
    }
    capsule {
      radius: 0.055
      length: 0.41
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5
}
bodies {
  name: "right_knee"
  colliders {
    position {
      z: -0.2
    }
    rotation {
    }
    capsule {
      radius: 0.05
      length: 0.41
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.0
}
bodies {
  name: "right_ankle"
  colliders {
    position {
      x: 0.045
      z: -0.0225
    }
    rotation {
    }
    box {
      halfsize {
        x: 0.0885
        y: 0.045
        z: 0.0275
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "left_hip"
  colliders {
    position {
      z: -0.21
    }
    rotation {
    }
    capsule {
      radius: 0.055
      length: 0.41
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5
}
bodies {
  name: "left_knee"
  colliders {
    position {
      z: -0.2
    }
    rotation {
    }
    capsule {
      radius: 0.05
      length: 0.41
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.0
}
bodies {
  name: "left_ankle"
  colliders {
    position {
      x: 0.045
      z: -0.0225
    }
    rotation {
    }
    box {
      halfsize {
        x: 0.0885
        y: 0.045
        z: 0.0275
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen { all: true }
}
joints {
  name: "chest_x"
  stiffness: 15000.0
  parent: "root"
  child: "chest"
  parent_offset {
    z: 0.236151
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.2
    max: 1.2
  }
  reference_rotation {
  }
}
joints {
  name: "chest_y"
  stiffness: 15000.0
  parent: "root"
  child: "chest"
  parent_offset {
    z: 0.236151
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.2
    max: 1.2
  }
  reference_rotation {
  }
}
joints {
  name: "chest_z"
  stiffness: 15000.0
  parent: "root"
  child: "chest"
  parent_offset {
    z: 0.236151
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.2
    max: 1.2
  }
  reference_rotation {
  }
}
joints {
  name: "neck_x"
  stiffness: 15000.0
  parent: "chest"
  child: "neck"
  parent_offset {
    z: 0.223894
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
joints {
  name: "neck_y"
  stiffness: 15000.0
  parent: "chest"
  child: "neck"
  parent_offset {
    z: 0.223894
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
joints {
  name: "neck_z"
  stiffness: 15000.0
  parent: "chest"
  child: "neck"
  parent_offset {
    z: 0.223894
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
joints {
  name: "right_shoulder_x"
  stiffness: 15000.0
  parent: "chest"
  child: "right_shoulder"
  parent_offset {
    x: -0.02405
    y: -0.18311
    z: 0.2435
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -3.14
    max: 0.5
  }
  reference_rotation {
  }
}
joints {
  name: "right_shoulder_y"
  stiffness: 15000.0
  parent: "chest"
  child: "right_shoulder"
  parent_offset {
    x: -0.02405
    y: -0.18311
    z: 0.2435
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -3.14
    max: 0.7
  }
  reference_rotation {
  }
}
joints {
  name: "right_shoulder_z"
  stiffness: 15000.0
  parent: "chest"
  child: "right_shoulder"
  parent_offset {
    x: -0.02405
    y: -0.18311
    z: 0.2435
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.5
    max: 1.5
  }
  reference_rotation {
  }
}
joints {
  name: "right_wrist"
  stiffness: 15000.0
  parent: "right_elbow"
  child: "right_wrist"
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
  }
  reference_rotation {
  }
}
joints {
  name: "right_elbow"
  stiffness: 15000.0
  parent: "right_shoulder"
  child: "right_elbow"
  parent_offset {
    z: -0.274788
  }
  child_offset {
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    max: 2.8
  }
  reference_rotation {
  }
}
joints {
  name: "left_shoulder_x"
  stiffness: 15000.0
  parent: "chest"
  child: "left_shoulder"
  parent_offset {
    x: -0.02405
    y: 0.18311
    z: 0.2435
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -0.5
    max: 3.14
  }
  reference_rotation {
  }
}
joints {
  name: "left_shoulder_y"
  stiffness: 15000.0
  parent: "chest"
  child: "left_shoulder"
  parent_offset {
    x: -0.02405
    y: 0.18311
    z: 0.2435
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -3.14
    max: 0.7
  }
  reference_rotation {
  }
}
joints {
  name: "left_shoulder_z"
  stiffness: 15000.0
  parent: "chest"
  child: "left_shoulder"
  parent_offset {
    x: -0.02405
    y: 0.18311
    z: 0.2435
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.5
    max: 1.5
  }
  reference_rotation {
  }
}
joints {
  name: "left_wrist"
  stiffness: 15000.0
  parent: "left_elbow"
  child: "left_wrist"
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
  }
  reference_rotation {
  }
}
joints {
  name: "left_elbow"
  stiffness: 15000.0
  parent: "left_shoulder"
  child: "left_elbow"
  parent_offset {
    z: -0.274788
  }
  child_offset {
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    max: 2.8
  }
  reference_rotation {
  }
}
joints {
  name: "right_hip_x"
  stiffness: 15000.0
  parent: "root"
  child: "right_hip"
  parent_offset {
    y: -0.084887
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -1.2
    max: 1.2
  }
  reference_rotation {
  }
}
joints {
  name: "right_hip_y"
  stiffness: 15000.0
  parent: "root"
  child: "right_hip"
  parent_offset {
    y: -0.084887
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -2.57
    max: 1.57
  }
  reference_rotation {
  }
}
joints {
  name: "right_hip_z"
  stiffness: 15000.0
  parent: "root"
  child: "right_hip"
  parent_offset {
    y: -0.084887
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
joints {
  name: "right_knee"
  stiffness: 15000.0
  parent: "right_hip"
  child: "right_knee"
  parent_offset {
    z: -0.421546
  }
  child_offset {
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -2.7
  }
  reference_rotation {
  }
}
joints {
  name: "right_ankle_x"
  stiffness: 15000.0
  parent: "right_knee"
  child: "right_ankle"
  parent_offset {
    z: -0.40987
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
joints {
  name: "right_ankle_y"
  stiffness: 15000.0
  parent: "right_knee"
  child: "right_ankle"
  parent_offset {
    z: -0.40987
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.57
  }
  reference_rotation {
  }
}
joints {
  name: "right_ankle_z"
  stiffness: 15000.0
  parent: "right_knee"
  child: "right_ankle"
  parent_offset {
    z: -0.40987
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
joints {
  name: "left_hip_x"
  stiffness: 15000.0
  parent: "root"
  child: "left_hip"
  parent_offset {
    y: 0.084887
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -1.2
    max: 1.2
  }
  reference_rotation {
  }
}
joints {
  name: "left_hip_y"
  stiffness: 15000.0
  parent: "root"
  child: "left_hip"
  parent_offset {
    y: 0.084887
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -2.57
    max: 1.57
  }
  reference_rotation {
  }
}
joints {
  name: "left_hip_z"
  stiffness: 15000.0
  parent: "root"
  child: "left_hip"
  parent_offset {
    y: 0.084887
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
joints {
  name: "left_knee"
  stiffness: 15000.0
  parent: "left_hip"
  child: "left_knee"
  parent_offset {
    z: -0.421546
  }
  child_offset {
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -2.7
  }
  reference_rotation {
  }
}
joints {
  name: "left_ankle_x"
  stiffness: 15000.0
  parent: "left_knee"
  child: "left_ankle"
  parent_offset {
    z: -0.40987
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
joints {
  name: "left_ankle_y"
  stiffness: 15000.0
  parent: "left_knee"
  child: "left_ankle"
  parent_offset {
    z: -0.40987
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.57
  }
  reference_rotation {
  }
}
joints {
  name: "left_ankle_z"
  stiffness: 15000.0
  parent: "left_knee"
  child: "left_ankle"
  parent_offset {
    z: -0.40987
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -1.0
    max: 1.0
  }
  reference_rotation {
  }
}
actuators {
  name: "chest_x"
  joint: "chest_x"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "chest_y"
  joint: "chest_y"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "chest_z"
  joint: "chest_z"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "neck_x"
  joint: "neck_x"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "neck_y"
  joint: "neck_y"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "neck_z"
  joint: "neck_z"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "right_shoulder_x"
  joint: "right_shoulder_x"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "right_shoulder_y"
  joint: "right_shoulder_y"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "right_shoulder_z"
  joint: "right_shoulder_z"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "right_elbow"
  joint: "right_elbow"
  strength: 60.0
  angle {
  }
}
actuators {
  name: "left_shoulder_x"
  joint: "left_shoulder_x"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "left_shoulder_y"
  joint: "left_shoulder_y"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "left_shoulder_z"
  joint: "left_shoulder_z"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "left_elbow"
  joint: "left_elbow"
  strength: 60.0
  angle {
  }
}
actuators {
  name: "right_hip_x"
  joint: "right_hip_x"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "right_hip_y"
  joint: "right_hip_y"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "right_hip_z"
  joint: "right_hip_z"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "right_knee"
  joint: "right_knee"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "right_ankle_x"
  joint: "right_ankle_x"
  strength: 90.0
  angle {
  }
}
actuators {
  name: "right_ankle_y"
  joint: "right_ankle_y"
  strength: 90.0
  angle {
  }
}
actuators {
  name: "right_ankle_z"
  joint: "right_ankle_z"
  strength: 90.0
  angle {
  }
}
actuators {
  name: "left_hip_x"
  joint: "left_hip_x"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "left_hip_y"
  joint: "left_hip_y"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "left_hip_z"
  joint: "left_hip_z"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "left_knee"
  joint: "left_knee"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "left_ankle_x"
  joint: "left_ankle_x"
  strength: 90.0
  angle {
  }
}
actuators {
  name: "left_ankle_y"
  joint: "left_ankle_y"
  strength: 90.0
  angle {
  }
}
actuators {
  name: "left_ankle_z"
  joint: "left_ankle_z"
  strength: 90.0
  angle {
  }
}
collide_include {
  first: "floor"
  second: "root"
}
collide_include {
  first: "floor"
  second: "chest"
}
collide_include {
  first: "floor"
  second: "neck"
}
collide_include {
  first: "floor"
  second: "right_shoulder"
}
collide_include {
  first: "floor"
  second: "right_elbow"
}
collide_include {
  first: "floor"
  second: "right_wrist"
}
collide_include {
  first: "floor"
  second: "left_shoulder"
}
collide_include {
  first: "floor"
  second: "left_elbow"
}
collide_include {
  first: "floor"
  second: "left_wrist"
}
collide_include {
  first: "floor"
  second: "right_hip"
}
collide_include {
  first: "floor"
  second: "right_knee"
}
collide_include {
  first: "floor"
  second: "right_ankle"
}
collide_include {
  first: "floor"
  second: "left_hip"
}
collide_include {
  first: "floor"
  second: "left_knee"
}
collide_include {
  first: "floor"
  second: "left_ankle"
}
friction: 1.0
gravity {
  z: -9.81
}
angular_damping: -0.05
baumgarte_erp: 0.1
dt: 0.015
substeps: 8
"""
