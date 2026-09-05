# CLI Reference

UniLab exposes package commands for common training and playback routes, and
keeps the lower-level scripts available for debugging Hydra composition.

## Unified Commands

| Goal | Command Shape | Routed Script |
| --- | --- | --- |
| PPO | `uv run train --algo ppo --task <task> --sim <backend>` | `scripts/train_rsl_rl.py` |
| APPO | `uv run train --algo appo --task <task> --sim <backend>` | `scripts/train_appo.py` |
| SAC | `uv run train --algo sac --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |
| TD3 | `uv run train --algo td3 --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |
| FlashSAC | `uv run train --algo flashsac --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |
| HIM-PPO (WheelBipe custom) | `uv run train --algo him_ppo --task wheelbipe_v14_flat --sim <backend>` | `scripts/train_custom_ppo.py` |
| DreamWaQ (WheelBipe custom) | `uv run train --algo dreamwaq --task wheelbipe_v14_flat --sim <backend>` | `scripts/train_custom_ppo.py` |
| NP3O + Barlow (WheelBipe custom) | `uv run train --algo np3o --task wheelbipe_v14_flat --sim <backend>` | `scripts/train_custom_ppo.py` |

Examples:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim motrix training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco training.no_play=true
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco
uv run train --algo him_ppo --task wheelbipe_v14_flat --sim mujoco training.no_play=true
```

For the migrated WheelBipe history policies, `--profile` selects the named
Hydra profile while retaining the backend owner YAML.  The source profiles set
long-run budgets and source-like network hyperparameters; they do not claim
Isaac dynamics or asset parity:

```bash
uv run train --algo him_ppo --task wheelbipe_v14_flat --sim mujoco \
  --profile source_him_long training.device=cpu
uv run train --algo dreamwaq --task wheelbipe_v14_flat --sim motrix \
  --profile source_dreamwaq_long training.device=cpu
uv run train --algo np3o --task wheelbipe_v14_flat --sim mujoco \
  --profile source_np3o_barlow_long training.device=cpu
```

The CLI builds the owner YAML path from `--algo`, `--task`, `--sim`, and
optional `--profile`. Route-defining values must use the CLI flags; Hydra
overrides after the command are for fields such as `algo.max_iterations`,
`algo.num_envs`, `training.device`, and `training.no_play`. For generic
algorithms `--profile` remains an owner YAML suffix (for example
`sharpa_inhand/mujoco_hora`); the three custom WheelBipe names above are Hydra
profile groups.

### Per-Environment Invocation

The CLI prefix depends on the install flavor:

- ROCm: run `make sync-rocm` once, then use `uv run ...`.
- Intel XPU: use `uv run --no-sync ...`.

## Tab Completion

Shell completion is optional. It only completes the `uv run train` / `uv run eval`
entrypoints, their flags, and some choices; it never changes command behavior.
On a fresh checkout, one setup command syncs the environment and installs the
completion:

```bash
make setup

# When you need Motrix:
make setup-motrix
```

`make setup` runs `uv sync` followed by `uv run --no-sync unilab-complete install`;
`make setup-motrix` runs `uv sync --extra motrix` followed by the same completion
install. The install command picks Bash or Zsh from `$SHELL` / platform and only
writes user-level rc files. The current shell is not auto-activated; reopen the
terminal or source the rc file to apply.

If `make` is unavailable, run the steps directly:

```bash
uv sync && uv run --no-sync unilab-complete install
```

Bash users (Linux / WSL) can instead add this to `~/.bashrc`:

```bash
source scripts/completions/unilab.bash
```

Zsh users (default on macOS) can add this to `~/.zshrc`:

```zsh
autoload -Uz compinit
compinit
source scripts/completions/unilab.zsh
```

After reopening the terminal or sourcing the rc file, candidates appear for
`uv run <TAB>`, `uv run train --algo <TAB>`, and `uv run train --sim <TAB>`.

## Evaluation

`uv run eval` sets `training.play_only=true` and optionally maps `--load-run` to
`algo.load_run`. The value can be `-1`, a run-directory name, or an absolute
run/checkpoint path.

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1 \
  --render-mode record
```

Supported render modes are `auto`, `interactive`, `record`, and `none`.

The custom WheelBipe routes use a dedicated history-aware runner and are
currently flat-only. Their `eval` path accepts the same `--load-run`/Hydra
checkpoint selectors as PPO and validates the algorithm/variant contract
before playback; use `--render-mode none` for headless numerical evaluation.

## Demo

```bash
uv run demo dance
uv run demo wallflip
uv run demo boxtracking
uv run demo locomani
uv run demo inhandgrasp
uv run demo dance --refresh --device cpu
```

Available demos: `teaser`, `dance`, `wallflip`, `boxtracking`, `locomani`, `inhandgrasp`.
Each demo fetches a pre-trained checkpoint from the
`unilabsim/unilab-checkpoints` Hugging Face dataset on first run and caches it
under `src/unilab/assets/checkpoints/<demo>/model_0.pt`. Pass `--refresh` to
re-download.

Mainland China users: when `huggingface.co` is unreachable, switch to the
community mirror before running the demo:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

The demo entrypoint is implemented by `src/unilab/demo.py` and routed from
`src/unilab/cli.py`.

## Low-Level Scripts

The lower-level scripts remain available when you need to inspect Hydra config
groups or reproduce a script-level issue. For normal usage, keep
route-defining values in the unified CLI flags above.

For off-policy routes, keep `--algo` aligned with the owner tree under
`conf/offpolicy/task/<algo>/`; do not include the algorithm name in `--task`.

## Common Overrides

```bash
training.no_play=true
algo.max_iterations=10
algo.num_envs=128
algo.load_run=-1
training.logger=wandb
```

Use `uv run eval ... --load-run ...` for playback and `--render-mode record`
or `--render-mode none` for render behavior.

Backend selection belongs to the CLI `--sim` choice and the resulting owner
YAML. Do not use `training.sim_backend=<backend>` as a standalone switch.
