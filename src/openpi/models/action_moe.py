"""Observation-routed residual FFN experts for the JAX action transformer."""

import dataclasses

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ActionMoEConfig:
    num_experts: int = 4
    rank: int = 16
    router_hidden_dim: int = 128
    # Zero means soft routing over all experts, in both training and inference.
    top_k: int = 0
    temperature: float = 1.0
    phase_loss_weight: float = 0.02
    balance_loss_weight: float = 0.001
    phase_warmup_steps: int = 1000
    phase_ramp_steps: int = 1000

    def __post_init__(self):
        if self.num_experts != 4:
            raise ValueError("The phase-supervised recipe requires four experts / phase classes.")
        if min(self.rank, self.router_hidden_dim) <= 0 or self.temperature <= 0:
            raise ValueError("MoE dimensions and temperature must be positive.")
        if self.top_k not in (0, 2, 3, 4):
            raise ValueError("Use soft routing (0) or top_k >= 2; hard top-1 has no routing gradient.")
        if min(self.phase_loss_weight, self.balance_loss_weight, self.phase_warmup_steps) < 0:
            raise ValueError("MoE loss weights and warmup must be nonnegative.")
        if self.phase_ramp_steps <= 0:
            raise ValueError("phase_ramp_steps must be positive.")


class ActionRouter(nnx.Module):
    """One route per observation, independent of target actions and flow time.

    Uses masked pooled image/language prefix embeddings (before the LLM) and
    normalized proprioception. No phase labels are accepted by this module.
    """

    def __init__(self, context_dim: int, state_dim: int, config: ActionMoEConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.context_norm = nnx.LayerNorm(context_dim, rngs=rngs)
        self.state_norm = nnx.LayerNorm(state_dim, rngs=rngs)
        self.hidden = nnx.Linear(context_dim + state_dim, config.router_hidden_dim, rngs=rngs)
        self.out = nnx.Linear(config.router_hidden_dim, config.num_experts, rngs=rngs)

    def __call__(self, prefix_tokens, prefix_mask, state):
        weights = prefix_mask.astype(jnp.float32)
        context = jnp.sum(prefix_tokens.astype(jnp.float32) * weights[..., None], axis=1)
        context /= jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1)
        features = jnp.concatenate([self.context_norm(context), self.state_norm(state)], axis=-1)
        logits = self.out(nnx.swish(self.hidden(features))) / self.config.temperature
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        gates = probs
        if 0 < self.config.top_k < self.config.num_experts:
            _, indices = jax.lax.top_k(logits, self.config.top_k)
            mask = jnp.sum(jax.nn.one_hot(indices, self.config.num_experts), axis=-2)
            gates = probs * mask
            gates /= jnp.maximum(jnp.sum(gates, axis=-1, keepdims=True), 1e-8)
        return gates, probs


class ResidualExperts(nn.Module):
    config: ActionMoEConfig
    width: int

    @nn.compact
    def __call__(self, x, gates):
        # Keep original FFN names/weights intact. Zero output projections make
        # this addition a no-op when loading a dense pretrained checkpoint.
        down = self.param(
            "down",
            nn.initializers.normal(stddev=self.width**-0.5),
            (self.config.num_experts, self.width, self.config.rank),
        )
        up = self.param(
            "up",
            nn.initializers.zeros_init(),
            (self.config.num_experts, self.config.rank, self.width),
        )
        hidden = nn.silu(jnp.einsum("btd,edr->bter", x, down.astype(x.dtype)))
        # This small reference implementation evaluates all adapters, including
        # with top-k gates. Top-k is semantic sparsity, not a speedup claim.
        return jnp.einsum("bter,erd,be->btd", hidden, up.astype(x.dtype), gates.astype(x.dtype))


def routing_loss(probs, phase_targets, phase_confidence, config: ActionMoEConfig, step):
    """Weak-label CE plus a small anti-starvation penalty, not equal-use forcing."""
    log_probs = jnp.log(jnp.maximum(probs, 1e-8))
    confidence = phase_confidence.astype(jnp.float32)
    ce = -jnp.sum(phase_targets * log_probs, axis=-1)
    phase_loss = jnp.sum(ce * confidence) / jnp.maximum(jnp.sum(confidence), 1.0)
    usage = jnp.mean(probs, axis=0)
    # Only penalize near-unused experts; unequal phase durations are expected.
    floor = 0.1 / config.num_experts
    balance_loss = jnp.mean(jnp.square(jax.nn.relu(floor - usage) / floor))
    ramp = jnp.clip((jnp.asarray(step) - config.phase_warmup_steps) / config.phase_ramp_steps, 0.0, 1.0)
    aux_loss = config.phase_loss_weight * ramp * phase_loss + config.balance_loss_weight * balance_loss
    accuracy = jnp.sum((jnp.argmax(probs, -1) == jnp.argmax(phase_targets, -1)) * confidence)
    accuracy /= jnp.maximum(jnp.sum(confidence), 1.0)
    metrics = {
        "moe/phase_loss": phase_loss,
        "moe/phase_accuracy": accuracy,
        "moe/label_confidence": jnp.mean(confidence),
        "moe/phase_ramp": ramp,
        "moe/balance_loss": balance_loss,
        "moe/router_entropy": -jnp.mean(jnp.sum(probs * log_probs, axis=-1)),
        **{f"moe/expert_{i}_prob": usage[i] for i in range(config.num_experts)},
    }
    return aux_loss, metrics
