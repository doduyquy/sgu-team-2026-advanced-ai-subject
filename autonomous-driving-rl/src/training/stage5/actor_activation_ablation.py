"""Versioned SAC policy for an actor-only LeakyReLU ablation.

The treatment changes only the two hidden activations in SAC's actor.  The
policy-wide activation remains ReLU so SB3 constructs the critic and target
critics with the control activation.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch
from stable_baselines3.sac.policies import Actor, SACPolicy


ACTOR_RELU_V1 = "actor_relu_v1"
ACTOR_LEAKY_RELU_001_V1 = "actor_leaky_relu_001_v1"


class ActorOnlyLeakyReLUPolicy(SACPolicy):
    """SB3 SAC policy whose actor is LeakyReLU(0.01), critic remains ReLU."""

    def make_actor(self, features_extractor=None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["activation_fn"] = torch.nn.LeakyReLU
        return Actor(**actor_kwargs).to(self.device)


def actor_activation_id(policy: SACPolicy) -> str:
    activation = getattr(policy.actor, "activation_fn", None)
    if activation is torch.nn.ReLU:
        return ACTOR_RELU_V1
    if activation is torch.nn.LeakyReLU:
        modules = [m for m in policy.actor.latent_pi.modules() if isinstance(m, torch.nn.LeakyReLU)]
        if modules and all(abs(float(m.negative_slope) - 0.01) < 1e-12 for m in modules):
            return ACTOR_LEAKY_RELU_001_V1
    return "unknown"


def activation_inventory(policy: SACPolicy) -> dict[str, Any]:
    actor_types = sorted({type(m).__name__ for m in policy.actor.latent_pi.modules()
                          if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU))})
    critic_types = sorted({type(m).__name__ for q in policy.critic.q_networks
                           for m in q.modules()
                           if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU))})
    slopes = sorted({float(m.negative_slope) for m in policy.actor.latent_pi.modules()
                     if isinstance(m, torch.nn.LeakyReLU)})
    return {
        "actor_id": actor_activation_id(policy), "actor_module_types": actor_types,
        "actor_leaky_slopes": slopes, "critic_module_types": critic_types,
        "critic_is_relu_only": critic_types == ["ReLU"],
        "critic_target_module_types": sorted({type(m).__name__ for q in policy.critic_target.q_networks
                                               for m in q.modules()
                                               if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU))}),
    }


def tensor_state_fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()

