"""Spinkick environment configuration for Unitree G1."""

import math

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

_MAX_ANG_VEL = 500 * math.pi / 180.0  # [rad/s]


def base_ang_vel_exceed(
  env: ManagerBasedRlEnv,
  threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  ang_vel = asset.data.root_link_ang_vel_b
  return torch.any(ang_vel.abs() > threshold, dim=-1)


def unitree_g1_spinkick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 spinkick environment configuration."""
  # Start with the base tracking environment without state estimation.
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  # Add custom spinkick termination.
  cfg.terminations["base_ang_vel_exceed"] = TerminationTermCfg(
    func=base_ang_vel_exceed, params={"threshold": _MAX_ANG_VEL}
  )

  # Customize rewards.
  cfg.rewards["joint_limit"].weight = -100.0

  return cfg


def unitree_g1_spinkick_runner_cfg():
  """Create RL runner configuration for Unitree G1 spinkick task."""
  from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_spinkick"
  return cfg
