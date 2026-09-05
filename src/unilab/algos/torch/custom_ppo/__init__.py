"""Algorithm-specific on-policy runtimes used by migrated WheelBipe tasks.

The upstream WheelBipe project ships three non-standard PPO variants.  This
package keeps their representation and constraint losses explicit instead of
silently routing them through the vanilla RSL-RL PPO runner.
"""

from unilab.algos.torch.custom_ppo.algorithm import NP3O, DreamWaQPPO, PPODreamWaq
from unilab.algos.torch.custom_ppo.models import (
    ADABOOT_MODES,
    ActorCriticBarlowTwins,
    ActorCriticBarlowTwinsSource,
    ActorCriticDreamWaq,
    DreamWaQActorCritic,
    NP3OActorCritic,
    SourceActorCriticBarlowTwins,
    SourceBarlowTwinsActorCritic,
)
from unilab.algos.torch.custom_ppo.runner import (
    CUSTOM_ALGORITHM_ALIASES,
    CUSTOM_ALGORITHM_CONTRACTS,
    CUSTOM_POLICY_ARCHITECTURES,
    CustomOnPolicyRunner,
    canonical_custom_algorithm,
    validate_custom_runner_contract,
)
from unilab.algos.torch.custom_ppo.source_barlow import (
    SourceBatchNorm1d,
    SourceEmpiricalNormalization,
    SourceMLP,
    SourceMlpBarlowTwinsActor,
    SourceMLPBatchNorm,
    SourceStateHistoryEncoder,
    load_source_state_dict,
    resolve_source_state_dict,
    source_off_diagonal,
)

__all__ = [
    "CustomOnPolicyRunner",
    "CUSTOM_ALGORITHM_ALIASES",
    "CUSTOM_ALGORITHM_CONTRACTS",
    "CUSTOM_POLICY_ARCHITECTURES",
    "ADABOOT_MODES",
    "DreamWaQActorCritic",
    "ActorCriticDreamWaq",
    "NP3OActorCritic",
    "ActorCriticBarlowTwins",
    "SourceBarlowTwinsActorCritic",
    "ActorCriticBarlowTwinsSource",
    "SourceActorCriticBarlowTwins",
    "DreamWaQPPO",
    "PPODreamWaq",
    "NP3O",
    "SourceEmpiricalNormalization",
    "SourceBatchNorm1d",
    "SourceMLP",
    "SourceMLPBatchNorm",
    "SourceMlpBarlowTwinsActor",
    "SourceStateHistoryEncoder",
    "load_source_state_dict",
    "resolve_source_state_dict",
    "source_off_diagonal",
    "canonical_custom_algorithm",
    "validate_custom_runner_contract",
]
