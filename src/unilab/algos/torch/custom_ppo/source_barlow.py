"""Explicit source-compatible NP3O/Barlow-Twins policy components.

The compact :class:`~unilab.algos.torch.custom_ppo.models.NP3OActorCritic`
owner is intentionally kept as the default WheelBipe route.  The upstream
``wheeled-legged_RL`` project, however, has a materially different NP3O
policy: it consumes a 312-dimensional ``on_constraint`` stream, normalizes
the complete stream for the critic, and trains a fixed five-frame teacher
backbone with a Barlow-Twins objective.  This module contains that graph as a
separate, opt-in implementation.  Keeping it separate is important: a
compact checkpoint must never be presented as a source checkpoint merely
because both classes are called NP3O.

The module names and layer containers intentionally follow the upstream
implementation (``obs_normalize``, ``priv_encoder``, ``scan_encoder``,
``history_encoder``, ``actor_teacher_backbone``, ``critic``, ``cost``).  This
gives source checkpoints a strict, inspectable loading path while still
rejecting shape or architecture mismatches at the runner boundary.

Source attribution: ``third-party/wheeled-legged_RL/source/agent_rl``.  The
vendored source is MIT/BSD-derived; see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SourceEmpiricalNormalization(nn.Module):
    """Faithful source normalizer (``count`` is a Python integer).

    UniLab's shared normalizer registers ``count`` as a tensor so distributed
    learners can checkpoint it.  The upstream WheelBipe/rsl_rl helper keeps
    ``count`` outside ``state_dict``; source checkpoints therefore contain
    only ``_mean``, ``_var`` and ``_std``.  A dedicated class avoids an
    otherwise invisible strict-load mismatch.
    """

    def __init__(
        self,
        shape: int | Sequence[int],
        eps: float = 1.0e-2,
        until: int | None = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        if until is not None and (isinstance(until, bool) or int(until) < 0):
            raise ValueError(f"source normalizer until must be non-negative or None, got {until!r}")
        # Keep this Python-side just like the upstream helper.  ``until`` is
        # a learning cap, not a tensor/buffer, so it intentionally does not
        # become part of source checkpoint state.
        self.until = None if until is None else int(until)
        shape_tuple = (int(shape),) if isinstance(shape, int) else tuple(int(x) for x in shape)
        if not shape_tuple or any(x <= 0 for x in shape_tuple):
            raise ValueError("source normalizer shape must contain positive dimensions")
        self.register_buffer("_mean", torch.zeros(shape_tuple).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape_tuple).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape_tuple).unsqueeze(0))
        self.count = 0

    @property
    def mean(self) -> torch.Tensor:
        mean = cast(torch.Tensor, self._mean)
        return mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        std = cast(torch.Tensor, self._std)
        return std.squeeze(0).clone()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(value)
        mean = cast(torch.Tensor, self._mean)
        std = cast(torch.Tensor, self._std)
        return (value - mean) / (std + self.eps)

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        count_x = int(value.shape[0])
        if count_x <= 0:
            return
        if self.until is not None and self.count >= self.until:
            return
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(value, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(value, dim=0, keepdim=True)
        mean = cast(torch.Tensor, self._mean)
        var = cast(torch.Tensor, self._var)
        std = cast(torch.Tensor, self._std)
        delta_mean = mean_x - mean
        mean.add_(rate * delta_mean)
        var.add_(rate * (var_x - var + delta_mean * (mean_x - mean)))
        std.copy_(var.clamp_min(0.0).sqrt())

    @torch.jit.unused
    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Map normalized values back to the source feature scale."""

        mean = cast(torch.Tensor, self._mean)
        std = cast(torch.Tensor, self._std)
        return value * (std + self.eps) + mean


def _activation(name: str) -> nn.Module:
    """Resolve the activation names used by the source ``rsl_rl`` helpers."""

    normalized = str(name).strip().lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "crelu":
        # ``rsl_rl.utils.resolve_nn_activation`` names CELU "crelu";
        # keeping this mapping exact matters for non-default source profiles.
        return nn.CELU()
    if normalized == "selu":
        return nn.SELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"lrelu", "leaky_relu"}:
        return nn.LeakyReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "swish":
        return nn.SiLU()
    if normalized == "softplus":
        return nn.Softplus()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "mish":
        return nn.Mish()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported source Barlow activation={name!r}")


class SourceMLP(nn.Module):
    """The source ``MLP`` container, including its ``.model`` key path."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_dims: Sequence[int] = (256, 128),
        activation: str = "elu",
        output_activation: str = "identity",
    ) -> None:
        super().__init__()
        hidden = tuple(int(width) for width in hidden_dims)
        if not hidden or any(width <= 0 for width in hidden):
            raise ValueError("source Barlow hidden_dims must contain positive widths")
        layers: list[nn.Module] = [nn.Linear(int(input_dim), hidden[0]), _activation(activation)]
        for index in range(len(hidden) - 1):
            layers.extend((nn.Linear(hidden[index], hidden[index + 1]), _activation(activation)))
        if output_dim is not None:
            layers.extend((nn.Linear(hidden[-1], int(output_dim)), _activation(output_activation)))
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)

    def __getitem__(self, index: int) -> nn.Module:
        return self.model[index]


class SourceBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm with a singleton-batch inference fallback.

    The subclass intentionally keeps PyTorch's parameter/buffer names and
    serialization format unchanged, so source checkpoints remain strict-load
    compatible.  Only the otherwise-invalid ``N=1`` training call is routed
    through running statistics.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.shape[0] < 2:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(input)


class SourceMLPBatchNorm(SourceMLP):
    """Source ``MLPBatchNorm`` (Linear -> BatchNorm -> activation)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_dims: Sequence[int] = (256, 128),
        activation: str = "elu",
        *,
        last_act: bool = False,
        bias: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        hidden = tuple(int(width) for width in hidden_dims)
        if not hidden or any(width <= 0 for width in hidden):
            raise ValueError("source Barlow hidden_dims must contain positive widths")
        layers: list[nn.Module] = [
            nn.Linear(int(input_dim), hidden[0], bias=bias),
            SourceBatchNorm1d(hidden[0]),
            _activation(activation),
        ]
        for index in range(len(hidden) - 1):
            layers.extend(
                (
                    nn.Linear(hidden[index], hidden[index + 1], bias=bias),
                    SourceBatchNorm1d(hidden[index + 1]),
                    _activation(activation),
                )
            )
        if output_dim is not None:
            layers.append(nn.Linear(hidden[-1], int(output_dim), bias=bias))
        if last_act:
            layers.append(_activation(activation))
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the source stack with a deterministic singleton-batch fallback.

        The Isaac/RSL-RL runs use large minibatches, so ``BatchNorm1d`` sees
        more than one sample and this is exactly the ordinary source path.
        Tiny UniLab smoke runs (and a final partial minibatch) can legitimately
        contain one row, for which PyTorch raises while updating batch
        statistics.  In that degenerate case use the frozen running statistics
        for that layer; this keeps the graph finite without changing the
        multi-sample source behavior or its state-dict keys.
        """

        output = value
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d) and self.training and output.shape[0] < 2:
                output = F.batch_norm(
                    output,
                    layer.running_mean,
                    layer.running_var,
                    layer.weight,
                    layer.bias,
                    training=False,
                    momentum=0.0,
                    eps=layer.eps,
                )
            else:
                output = layer(output)
        return output


class SourceStateHistoryEncoder(nn.Module):
    """Convolutional history encoder used by source NP3O's optional critic.

    The upstream implementation intentionally supports only 10, 20, or 50
    frames.  Keeping that closed set catches a source configuration typo at
    construction instead of silently changing the receptive field.
    """

    def __init__(
        self,
        num_his_frame: int,
        state_dim: int,
        encoding_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        frames = int(num_his_frame)
        if frames not in {10, 20, 50}:
            raise ValueError("source StateHistoryEncoder num_his_frame must be 10, 20, or 50")
        channel_size = 10
        self.encoder = nn.Sequential(
            nn.Linear(int(state_dim), 3 * channel_size),
            _activation(activation),
        )
        if frames == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=8, stride=4),
                _activation(activation),
                nn.Conv1d(2 * channel_size, channel_size, kernel_size=5, stride=1),
                _activation(activation),
                nn.Conv1d(channel_size, channel_size, kernel_size=5, stride=1),
                _activation(activation),
                nn.Flatten(),
            )
        elif frames == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=6, stride=2),
                _activation(activation),
                nn.Conv1d(2 * channel_size, channel_size, kernel_size=4, stride=2),
                _activation(activation),
                nn.Flatten(),
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=4, stride=2),
                _activation(activation),
                nn.Conv1d(2 * channel_size, channel_size, kernel_size=2, stride=1),
                _activation(activation),
                nn.Flatten(),
            )
        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, int(encoding_dim)),
            _activation(activation),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projection = self.encoder(value)
        output = self.conv_layers(projection.permute(0, 2, 1))
        return self.linear_output(output)


def source_off_diagonal(value: torch.Tensor) -> torch.Tensor:
    """Return the off-diagonal entries exactly as the source helper does."""

    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"Barlow cross-correlation must be square, got {tuple(value.shape)}")
    n = int(value.shape[0])
    return value.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SourceMlpBarlowTwinsActor(nn.Module):
    """Source teacher/student actor backbone.

    The upstream class receives a nominal ``num_hist=5`` but normalizes a
    ten-frame history and selects its final five frames.  This seemingly odd
    behavior is part of the checkpoint graph, so it is represented explicitly
    by ``source_history_frames`` and ``actor_history_frames``.
    """

    def __init__(
        self,
        num_prop: int,
        num_hist: int,
        num_state_est: int,
        obs_encoder_dims: Sequence[int],
        mlp_encoder_dims: Sequence[int],
        actor_dims: Sequence[int],
        latent_dim: int,
        num_actions: int,
        activation: str,
        *,
        source_history_frames: int = 10,
        actor_history_frames: int = 5,
    ) -> None:
        super().__init__()
        if int(num_hist) != int(actor_history_frames):
            raise ValueError(
                "source MlpBarlowTwinsActor expects the source nominal five-frame actor window; "
                f"got num_hist={num_hist}"
            )
        if int(source_history_frames) < int(actor_history_frames):
            raise ValueError("source history must contain at least the actor window")
        self.num_prop = int(num_prop)
        self.num_hist = int(num_hist)
        self.num_state_est = int(num_state_est)
        self.source_history_frames = int(source_history_frames)
        self.actor_history_frames = int(actor_history_frames)
        self.obs_normalizer = SourceEmpiricalNormalization(shape=self.num_prop)
        # ``obs_encoder_dims`` is accepted for source constructor parity.  The
        # shipped source graph does not instantiate an obs_encoder; retaining
        # no unused module is necessary for strict source state-dict loading.
        del obs_encoder_dims
        self.mlp_encoder = SourceMLPBatchNorm(
            self.num_prop * self.num_hist,
            None,
            mlp_encoder_dims,
            activation,
            last_act=False,
        )
        encoder_width = int(tuple(mlp_encoder_dims)[-1])
        self.latent_layer = nn.Sequential(
            nn.Linear(encoder_width, 32),
            SourceBatchNorm1d(32),
            nn.ELU(),
            nn.Linear(32, int(latent_dim)),
        )
        self.vel_layer = nn.Linear(encoder_width, int(num_state_est))
        self.actor = SourceMLP(
            self.num_prop + int(num_state_est) + int(latent_dim),
            int(num_actions),
            actor_dims,
            activation,
        )
        self.projector = SourceMLPBatchNorm(
            int(latent_dim),
            64,
            [64],
            activation,
            last_act=False,
            bias=False,
        )
        # Same state-dict layout as ``nn.BatchNorm1d``; the small subclass
        # only avoids a PyTorch singleton-batch exception in tiny smoke runs.
        self.bn = SourceBatchNorm1d(64, affine=False)

    def _normalize_history(self, obs_hist: torch.Tensor) -> torch.Tensor:
        if obs_hist.ndim != 3:
            raise ValueError(
                "source Barlow history must have shape (batch, frames, num_prop), "
                f"got {tuple(obs_hist.shape)}"
            )
        if int(obs_hist.shape[1]) != self.source_history_frames:
            raise ValueError(
                "source Barlow history frame count must be "
                f"{self.source_history_frames}, got {obs_hist.shape[1]}"
            )
        if int(obs_hist.shape[2]) != self.num_prop:
            raise ValueError(
                f"source Barlow history feature width must be {self.num_prop}, "
                f"got {obs_hist.shape[2]}"
            )
        # The source normalizer is stateful in train mode.  Calling it on the
        # flattened frames preserves the source update order.
        return self.obs_normalizer(obs_hist.reshape(-1, self.num_prop)).reshape(
            -1, self.source_history_frames, self.num_prop
        )

    def _normalize_pair(
        self, obs: torch.Tensor, obs_hist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2 or int(obs.shape[-1]) != self.num_prop:
            raise ValueError(
                f"source Barlow current observation must have width {self.num_prop}, "
                f"got {tuple(obs.shape)}"
            )
        # Source calls the same normalizer first for the current frame and then
        # for the history.  Keep that order (and running-stat updates) exact.
        obs_norm = self.obs_normalizer(obs)
        hist_norm = self._normalize_history(obs_hist)
        return obs_norm, hist_norm

    def _tail_features(
        self, obs: torch.Tensor, obs_hist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_norm, hist_norm = self._normalize_pair(obs, obs_hist)
        history_full = torch.cat((hist_norm[:, 1:, :], obs_norm.unsqueeze(1)), dim=1)
        batch = int(history_full.shape[0])
        tail = history_full[:, -self.actor_history_frames :, :].reshape(
            batch, self.num_prop * self.actor_history_frames
        )
        return obs_norm, tail

    def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        obs_norm, tail = self._tail_features(obs, obs_hist)
        # Source deliberately freezes the encoder during the actor forward;
        # representation gradients enter through BarlowTwinsLoss instead.
        with torch.no_grad():
            latent = self.mlp_encoder(tail)
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
        actor_input = torch.cat((vel.detach(), z.detach(), obs_norm.detach()), dim=-1)
        return self.actor(actor_input)

    def BarlowTwinsLoss(
        self,
        obs: torch.Tensor,
        obs_hist: torch.Tensor,
        priv: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        """Compute the source Barlow-Twins + velocity regression objective."""

        obs_norm, hist_norm = self._normalize_pair(obs, obs_hist)
        obs_norm = obs_norm.detach()
        hist_norm = hist_norm.detach()
        history_full = torch.cat((hist_norm[:, 1:, :], obs_norm.unsqueeze(1)), dim=1)
        batch = int(obs_norm.shape[0])
        # The source implementation deliberately uses two temporally
        # distinct views for the Barlow objective: ``z1`` sees the shifted
        # history with the current frame appended, while ``z2`` sees the
        # unshifted history tail.  This is easy to miss because the actor
        # inference path only ever consumes the shifted tail; collapsing the
        # views to one tensor changes the representation loss (and gradients)
        # for every non-stationary rollout.
        tail = history_full[:, -self.actor_history_frames :, :].reshape(
            batch, self.num_prop * self.actor_history_frames
        )
        source_tail = hist_norm[:, -self.actor_history_frames :, :].reshape(
            batch, self.num_prop * self.actor_history_frames
        )
        z1 = self.mlp_encoder(tail)
        z2 = self.mlp_encoder(source_tail)
        z1_latent = self.latent_layer(z1)
        z1_vel = self.vel_layer(z1)
        z2_latent = self.latent_layer(z2)
        z1_projected = self.projector(z1_latent)
        z2_projected = self.projector(z2_latent)
        c = self.bn(z1_projected).T @ self.bn(z2_projected)
        c = c / max(batch, 1)
        on_diag = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag = source_off_diagonal(c).pow(2).sum()
        if priv.ndim != 2 or int(priv.shape[-1]) != self.num_state_est:
            raise ValueError(
                f"source Barlow privileged estimate must have width {self.num_state_est}, "
                f"got {tuple(priv.shape)}"
            )
        priv_loss = F.mse_loss(z1_vel, priv)
        return on_diag + float(weight) * off_diag + priv_loss


def _source_flat_obs(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for key in ("on_constraint", "state", "policy"):
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
    for key in ("on_constraint", "state", "policy"):
        try:
            candidate = value[key]
        except (AttributeError, KeyError, TypeError, IndexError):
            continue
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError("source NP3O observation must be a tensor or contain on_constraint")


def resolve_source_state_dict(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a source/RSL-RL model mapping without partial loading.

    Upstream runners use ``model_state_dict``; UniLab emits additive
    ``policy_state_dict``/``actor_state_dict`` aliases.  DDP may prepend one
    of a small, known set of module prefixes.  We strip a prefix only when it
    applies to every key, then let strict loading report any real mismatch.
    """

    state: Any = None
    for key in ("policy_state_dict", "actor_state_dict", "model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if candidate is not None:
            state = candidate
            break
    # A few lightweight exporters write the state mapping itself rather than
    # wrapping it under ``model_state_dict``.  Accept that form only when the
    # *entire* payload is a non-empty tensor mapping; a normal runner payload
    # also contains metadata/optimizer entries and must continue to fail
    # closed when its model key is absent.
    if (
        state is None
        and payload
        and all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in payload.items()
        )
    ):
        state = payload
    if not isinstance(state, Mapping):
        raise KeyError(
            "source checkpoint must contain policy_state_dict, actor_state_dict, "
            "model_state_dict, or state_dict (or be a direct tensor mapping)"
        )
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("source checkpoint state_dict must map string keys to tensors")
        result[key] = value
    # DDP/checkpoint wrappers occasionally compose prefixes (for example
    # ``module.policy.``).  Strip only a uniform, known prefix and repeat; a
    # partial strip would turn a valid source checkpoint into a misleading
    # shape error while an arbitrary key rewrite would weaken strict loading.
    prefixes = ("module.", "policy.", "actor_critic.", "model.")
    changed = True
    while result and changed:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in result):
                result = {key[len(prefix) :]: value for key, value in result.items()}
                changed = True
                break
    if not result:
        raise ValueError("source checkpoint state_dict cannot be empty")
    return result


def load_source_state_dict(module: nn.Module, payload: Mapping[str, Any]) -> None:
    """Strictly load a source policy and expose an actionable mismatch."""

    state = resolve_source_state_dict(payload)
    try:
        module.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "source Barlow checkpoint tensors do not match the explicit source graph; "
            "compact NP3O checkpoints cannot be loaded here: "
            f"{exc}"
        ) from exc


class SourceBarlowTwinsActorCritic(nn.Module):
    """The upstream WheelBipe ``ActorCriticBarlowTwins`` graph.

    This class is deliberately opt-in.  For the canonical V14 values
    ``(num_prop, num_scan, num_state_est, num_priv_latent, num_hist)`` are
    ``(28, 0, 4, 4, 10)`` and the required ``on_constraint`` width is 312.
    ``num_critic_obs`` is therefore the same 312-dimensional stream at the
    public model boundary, while the internal value MLP receives the source
    32-dimensional ``[policy, priv_latent]`` feature vector.
    """

    is_recurrent = False
    source_architecture = True

    def __init__(
        self,
        num_prop: int,
        num_scan: int,
        num_state_est: int,
        num_priv_latent: int,
        num_hist: int,
        num_actions: int,
        scan_encoder_dims: Sequence[int] | None = (256, 256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        hist_encoder: bool = False,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        fixed_std: bool = False,
        action_mean_clip: float | None = 20.0,
        *,
        priv_encoder_dims: Sequence[int] = (),
        num_costs: int = 5,
        teacher_act: bool = True,
        imi_flag: bool = True,
        latent_dim: int = 16,
        continue_from_last_std: bool = True,
        tanh_encoder_output: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        ints = {
            "num_prop": num_prop,
            "num_state_est": num_state_est,
            "num_priv_latent": num_priv_latent,
            "num_hist": num_hist,
            "num_actions": num_actions,
            "num_costs": num_costs,
        }
        if any(isinstance(value, bool) or int(value) <= 0 for value in ints.values()):
            raise ValueError("source Barlow dimensions must be positive")
        if isinstance(num_scan, bool) or int(num_scan) < 0:
            raise ValueError("source Barlow num_scan must be non-negative")
        if int(num_costs) != 5:
            raise ValueError(f"source NP3O requires exactly five costs, got {num_costs}")
        if int(num_hist) != 10:
            raise ValueError(
                f"source WheelBipe NP3O Barlow graph requires num_hist=10; got {num_hist}"
            )
        if not torch.isfinite(torch.tensor(float(init_noise_std))) or float(init_noise_std) <= 0:
            raise ValueError("source Barlow init_noise_std must be finite and positive")
        self.num_prop = int(num_prop)
        self.num_scan = int(num_scan)
        self.num_hist = int(num_hist)
        self.num_actions = int(num_actions)
        self.num_state_est = int(num_state_est)
        self.num_priv_latent = int(num_priv_latent)
        self.num_costs = int(num_costs)
        self.num_obs = (
            self.num_prop + self.num_scan + self.num_priv_latent + self.num_hist * self.num_prop
        )
        self.num_actor_obs = self.num_obs
        self.num_critic_obs = self.num_obs
        self.action_mean_clip = None if action_mean_clip is None else float(action_mean_clip)
        self.if_scan_encode = scan_encoder_dims is not None and self.num_scan > 0
        self.hist_encoder = bool(hist_encoder)
        self.teacher_act = bool(teacher_act)
        self.imi_flag = bool(imi_flag)
        # These two fields are present in the upstream WheelBipe policy
        # config.  The checked-in source ActorCriticBarlowTwins constructor
        # accepts them through ``**kwargs`` but does not branch on either
        # value; retain them as explicit metadata so source YAML composes
        # without silently dropping a knob while keeping the graph/state
        # dict byte-for-byte compatible.
        self.continue_from_last_std = bool(continue_from_last_std)
        self.tanh_encoder_output = bool(tanh_encoder_output)
        self.obs_normalize = SourceEmpiricalNormalization(shape=self.num_obs)

        priv_dims = tuple(int(width) for width in priv_encoder_dims)
        if priv_dims:
            self.priv_encoder = SourceMLP(self.num_priv_latent, None, priv_dims, activation)
            priv_output_dim = priv_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_output_dim = self.num_priv_latent

        if self.if_scan_encode:
            scan_dims = tuple(int(width) for width in (scan_encoder_dims or ()))
            if not scan_dims:
                raise ValueError("scan_encoder_dims must be non-empty when num_scan > 0")
            self.scan_encoder = SourceMLP(
                self.num_scan,
                scan_dims[-1],
                scan_dims[:-1] or (scan_dims[-1],),
                activation,
            )
            # The source call uses hidden_dims[:-1] and an output layer.  For
            # a one-width tuple, preserve the same effective linear shape.
            if len(scan_dims) == 1:
                self.scan_encoder = SourceMLP(
                    self.num_scan, scan_dims[-1], (scan_dims[-1],), activation
                )
            self.scan_encoder_output_dim = scan_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = self.num_scan

        self.history_encoder = SourceStateHistoryEncoder(
            self.num_hist, self.num_prop, 16, activation
        )
        # The source outer graph fixes this backbone's actor history to five
        # frames even though the on-constraint stream carries ten.
        self.actor_teacher_backbone = SourceMlpBarlowTwinsActor(
            num_prop=self.num_prop,
            num_hist=5,
            num_state_est=self.num_state_est,
            actor_dims=(512, 256, 128),
            mlp_encoder_dims=(128, 64),
            activation=activation,
            latent_dim=int(latent_dim),
            num_actions=self.num_actions,
            obs_encoder_dims=(128, 64),
        )
        critic_input_dim = self.num_prop + self.scan_encoder_output_dim + priv_output_dim
        if self.hist_encoder:
            critic_input_dim += 16
        self.critic = SourceMLP(critic_input_dim, 1, critic_hidden_dims, activation)
        cost_backbone = SourceMLP(
            critic_input_dim,
            self.num_costs,
            critic_hidden_dims,
            activation,
            output_activation="identity",
        )
        self.cost = nn.Sequential(cost_backbone, nn.Softplus())

        self.fixed_std = bool(fixed_std)
        std = torch.full((self.num_actions,), float(init_noise_std))
        self.std: torch.Tensor | nn.Parameter
        if self.fixed_std:
            # Match source behavior: fixed standard deviation is not a trainable
            # parameter.  Registering a buffer keeps device moves/checkpoints
            # deterministic without changing the trainable graph.
            # ``fixed_std`` is a plain tensor in the source class and is not
            # serialized.  A non-persistent buffer preserves that checkpoint
            # behavior while still following ``.to(device)`` correctly.
            self.register_buffer("std", std, persistent=False)
        else:
            self.std = nn.Parameter(std)
        self.distribution: Normal | None = None

    @staticmethod
    def init_weights(sequential: nn.Sequential, scales: Sequence[float]) -> None:
        for index, module in enumerate(
            layer for layer in sequential if isinstance(layer, nn.Linear)
        ):
            if index < len(scales):
                nn.init.orthogonal_(module.weight, gain=cast(Any, float(scales[index])))

    def _sanitize_tensor(self, value: torch.Tensor, clip: float | None = None) -> torch.Tensor:
        if clip is None:
            return torch.nan_to_num(value, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return torch.nan_to_num(value, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)

    def _obs(self, value: Any) -> torch.Tensor:
        obs = _source_flat_obs(value)
        if obs.ndim != 2 or int(obs.shape[-1]) != self.num_obs:
            raise ValueError(
                "source NP3O on_constraint width must be "
                f"{self.num_obs}, got shape={tuple(obs.shape)}"
            )
        return self._sanitize_tensor(obs)

    def get_std(self) -> torch.Tensor:
        return self.std

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("source Barlow act() must be called before action_mean")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("source Barlow act() must be called before action_std")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("source Barlow act() must be called before entropy")
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def set_teacher_act(self, flag: bool) -> None:
        if not isinstance(flag, bool):
            raise ValueError("source Barlow teacher_act must be boolean")
        self.teacher_act = flag

    def update_distribution(self, obs: Any) -> None:
        mean = self._sanitize_tensor(self.act_teacher(obs), self.action_mean_clip)
        std = torch.nan_to_num(self.get_std().expand_as(mean), nan=1.0, posinf=10.0, neginf=1.0e-6)
        self.distribution = Normal(mean, std.clamp_min(1.0e-6))

    def act(self, obs: Any, **kwargs: Any) -> torch.Tensor:
        del kwargs
        self.update_distribution(obs)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, obs: Any) -> torch.Tensor:
        return self._sanitize_tensor(self.act_teacher(obs), self.action_mean_clip)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("source Barlow act() must be called before log probability")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def _split(
        self, value: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self._obs(value)
        prop = obs[:, : self.num_prop]
        scan_start = self.num_prop
        scan = obs[:, scan_start : scan_start + self.num_scan]
        priv_start = scan_start + self.num_scan
        priv = obs[:, priv_start : priv_start + self.num_priv_latent]
        hist = obs[:, -self.num_hist * self.num_prop :].reshape(-1, self.num_hist, self.num_prop)
        return obs, prop, scan, priv, hist

    def act_teacher(self, obs: Any, **kwargs: Any) -> torch.Tensor:
        del kwargs
        _full, prop, _scan, _priv, hist = self._split(obs)
        return self.actor_teacher_backbone(prop, hist)

    def _critic_features(self, obs: Any) -> torch.Tensor:
        _full, prop, scan, priv, hist = self._split(obs)
        scan_latent = self.scan_encoder(scan)
        priv_latent = self.priv_encoder(priv)
        parts = [prop, priv_latent, scan_latent]
        if self.hist_encoder:
            parts.append(self.history_encoder(hist))
        return torch.cat(parts, dim=1)

    def evaluate(self, obs: Any, **kwargs: Any) -> torch.Tensor:
        del kwargs
        full = self.obs_normalize(self._obs(obs))
        return self.critic(self._critic_features(full))

    def evaluate_cost(self, obs: Any, **kwargs: Any) -> torch.Tensor:
        del kwargs
        full = self.obs_normalize(self._obs(obs))
        return self.cost(self._critic_features(full))

    def infer_priv_latent(self, obs: Any) -> torch.Tensor:
        _full, _prop, _scan, priv, _hist = self._split(obs)
        return self.priv_encoder(priv)

    def infer_scandots_latent(self, obs: Any) -> torch.Tensor:
        _full, _prop, scan, _priv, _hist = self._split(obs)
        return self.scan_encoder(scan)

    def infer_hist_latent(self, obs: Any) -> torch.Tensor:
        _full, _prop, _scan, _priv, hist = self._split(obs)
        return self.history_encoder(hist)

    def imitation_learning_loss(self, obs: Any, imi_weight: float = 1.0) -> torch.Tensor:
        _full, prop, _scan, _priv, hist = self._split(obs)
        # Source uses the first ``num_state_est`` entries of the privileged
        # latent segment as the velocity/estimation target.
        priv = _full[
            :, self.num_prop + self.num_scan : self.num_prop + self.num_scan + self.num_state_est
        ]
        return float(imi_weight) * self.actor_teacher_backbone.BarlowTwinsLoss(
            prop, hist, priv, 5.0e-3
        )

    def imitation_mode(self) -> None:
        # Kept as a source API hook.  The source implementation is a no-op.
        return None

    def representation_parameters(self):
        yield from self.actor_teacher_backbone.mlp_encoder.parameters()
        yield from self.actor_teacher_backbone.latent_layer.parameters()
        yield from self.actor_teacher_backbone.vel_layer.parameters()
        yield from self.actor_teacher_backbone.projector.parameters()
        yield from self.actor_teacher_backbone.bn.parameters()


__all__ = [
    "SourceEmpiricalNormalization",
    "SourceBatchNorm1d",
    "SourceMLP",
    "SourceMLPBatchNorm",
    "SourceStateHistoryEncoder",
    "SourceMlpBarlowTwinsActor",
    "SourceBarlowTwinsActorCritic",
    "source_off_diagonal",
    "resolve_source_state_dict",
    "load_source_state_dict",
]
