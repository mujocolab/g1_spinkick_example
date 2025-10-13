import math
from dataclasses import dataclass, field

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.tasks.tracking.config.g1.flat_env_cfg import G1FlatNoStateEstimationEnvCfg
from mjlab.tasks.tracking.tracking_env_cfg import TerminationsCfg

_MAX_ANG_VEL = 500 * math.pi / 180.0  # [rad/s]


def base_ang_vel_exceed(
  env: ManagerBasedRlEnv,
  threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  ang_vel = asset.data.root_link_ang_vel_b
  return torch.any(ang_vel.abs() > threshold, dim=-1)


@dataclass
class SpinkickTerminationsCfg(TerminationsCfg):
  base_ang_vel_exceed: DoneTerm = term(
    DoneTerm,
    func=base_ang_vel_exceed,
    params={"threshold": _MAX_ANG_VEL},
  )


@dataclass
class G1SpinkickCfg(G1FlatNoStateEstimationEnvCfg):
  terminations: SpinkickTerminationsCfg = field(default_factory=SpinkickTerminationsCfg)


@dataclass
class G1SpinkickCfg_PLAY(G1SpinkickCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
