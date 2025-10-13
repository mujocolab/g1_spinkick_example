import gymnasium as gym

gym.register(
  id="Mjlab-Spinkick-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1SpinkickCfg",
    "rl_cfg_entry_point": "mjlab.tasks.tracking.config.g1.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Spinkick-Unitree-G1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1SpinkickCfg_PLAY",
    "rl_cfg_entry_point": "mjlab.tasks.tracking.config.g1.rl_cfg:G1FlatPPORunnerCfg",
  },
)
