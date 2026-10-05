import sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from hydra import compose, initialize
from omegaconf import OmegaConf
with initialize(version_base="1.3", config_path="conf/ppo"):
    cfg = compose(config_name="config", overrides=[
        "task=real68_balance/mujoco", "training.play_only=true", "training.no_play=false",
        "env.cold_start.fraction=1.0",
        "+env.cold_start.ladder=[0.192,0.202,0.212,0.223,0.234,0.245,0.257]",
        "play_profile.env.terrain_curriculum.initial_type_cols=[0,0,0,0,0,0,0]",
        "play_profile.env.terrain_curriculum.initial_levels=[0,0,0,0,0,0,0]",
        "play_profile.env.commands.rel_standing_envs=1.0",
    ])
from train_rsl_rl import _resolve_ppo_wrapper_cls, _resolve_ppo_runner_cls, build_ppo_play_env_cfg_override
from unilab.training import create_env
from unilab.utils.device import get_default_device
from unilab.base.registry import ensure_registries
from unilab.visualization.render_many import render_states_tracking_to_video
ensure_registries()
device = str(get_default_device()) or "cuda"
rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
wrapper_cls = _resolve_ppo_wrapper_cls(rl_cfg)
runner_cls, is_him = _resolve_ppo_runner_cls(rl_cfg)
env = create_env(cfg, num_envs=cfg.training.play_env_num, env_cfg_override=build_ppo_play_env_cfg_override(cfg))
wrapped_env = wrapper_cls(env, device=device)
train_cfg = dict(rl_cfg); train_cfg.setdefault("runner", {})["logger"] = "none"
runner = runner_cls(wrapped_env, train_cfg, log_dir=None, device=device)
runner.load("logs/rsl_rl_ppo/Real68Balance/2026-08-08_00-15-51_mujoco/model_400.pt", map_location=device, restore_training_state=False)
policy = runner.get_inference_policy(device=device)

wrapped_env.reset()
terrain_z = np.asarray(env._spawn.origins_for(np.arange(env.num_envs, dtype=np.int32)))[:, 2]
snapshots = []
hz = 50
N = int(6.0 * hz)  # 6 s
for step in range(N):
    wrapped_env.step(policy(wrapped_env.get_observations()))
    if step % 2 == 0:  # 25 fps sampling
        snapshots.append(np.asarray(env.get_physics_state_snapshot(), dtype=np.float32).copy())
print("收集状态帧:", len(snapshots), "| 每帧形状:", snapshots[0].shape)
print("terrain_z:", terrain_z.round(3))
# 减去地形偏移，使 qpos z 对 scene_flat 平地正确
for s in snapshots:
    s[:, 3] -= terrain_z
print("初始各 env base_z(减地形后):", snapshots[0][:, 3].round(3))
out = "logs/rsl_rl_ppo/Real68Balance/2026-08-08_00-15-51_mujoco/ladder_grid.mp4"
ok = render_states_tracking_to_video(
    snapshots,
    "src/unilab/assets/robots/real68/scene_flat.xml",
    out,
    fps=25,
    tracking_env_idx=0,
    max_extra_envs=6,
    cam_distance=12,
    cam_elevation=-20,
    cam_azimuth=120,
    render_spacing=2.0,
)
print("渲染完成:", ok, "->", out)
env.close()
