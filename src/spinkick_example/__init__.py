from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .spinkick_env_cfg import (
  unitree_g1_spinkick_env_cfg,
  unitree_g1_spinkick_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Spinkick-Unitree-G1",
  env_cfg=unitree_g1_spinkick_env_cfg(),
  play_env_cfg=unitree_g1_spinkick_env_cfg(play=True),
  rl_cfg=unitree_g1_spinkick_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
