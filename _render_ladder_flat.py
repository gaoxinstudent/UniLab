import sys, numpy as np
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
from hydra import compose, initialize
from omegaconf import OmegaConf
with initialize(version_base='1.3', config_path='conf/ppo'):
    cfg = compose(config_name='config', overrides=[
        'task=real68_balance/mujoco','training.play_only=true','training.no_play=false',
    ])
from train_rsl_rl import _resolve_ppo_wrapper_cls,_resolve_ppo_runner_cls
from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.utils.device import get_default_device
from unilab.visualization.render_many import render_states_tracking_to_video
ensure_registries()
device=str(get_default_device()) or 'cuda'
rl_cfg=OmegaConf.to_container(cfg.algo,resolve=True)
wrapper_cls=_resolve_ppo_wrapper_cls(rl_cfg); runner_cls,is_him=_resolve_ppo_runner_cls(rl_cfg)
env = registry.make("Real68BalanceFlat", num_envs=7, sim_backend="mujoco", env_cfg_override={
    "cold_start": {"enabled": True, "fraction": 1.0, "ladder": [0.192,0.202,0.212,0.223,0.234,0.245,0.257]},
    "commands": {"rel_standing_envs": 1.0},
    "command_curriculum": {"enabled": False},
    "recovery": {"enabled": False},
    "reward_config": {"scales": {"alive": 1.0}, "tracking_sigma": 0.25},
})
wrapped_env=wrapper_cls(env,device=device)
train_cfg=dict(rl_cfg); train_cfg.setdefault('runner',{})['logger']='none'
runner=runner_cls(wrapped_env,train_cfg,log_dir=None,device=device)
runner.load('logs/rsl_rl_ppo/Real68Balance/2026-08-08_00-15-51_mujoco/model_400.pt',map_location=device,restore_training_state=False)
policy=runner.get_inference_policy(device=device)
wrapped_env.reset()
print("初始 base_z:", np.asarray(env._backend.get_base_pos())[:,2].round(3))
snapshots=[]
hz=50
for step in range(int(6.0*hz)):
    wrapped_env.step(policy(wrapped_env.get_observations()))
    if step%2==0:
        snapshots.append(np.asarray(env.get_physics_state_snapshot(),dtype=np.float32).copy())
    if step in (0,25,50,99,150,199):
        print(f"t={step*0.02:.1f}s base_z={np.asarray(env._backend.get_base_pos())[:,2].round(3)}")
out="logs/rsl_rl_ppo/Real68Balance/2026-08-08_00-15-51_mujoco/ladder_flat.mp4"
ok=render_states_tracking_to_video(snapshots,"src/unilab/assets/robots/real68/scene_flat.xml",out,
    fps=25,tracking_env_idx=0,max_extra_envs=6,cam_distance=12,cam_elevation=-18,cam_azimuth=120,render_spacing=2.5)
print("渲染:",ok,"->",out)
env.close()
