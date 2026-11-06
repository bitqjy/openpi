"""CPU tests for dense compatibility, real routing gradients and inference paths."""

import dataclasses
import functools
import importlib.util
from pathlib import Path

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import action_moe
from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.training import checkpoints
from openpi.training import config as training_config
from openpi.training import utils as training_utils
from openpi.training import weight_loaders


class TinyVision(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, *, train=False):
        del train
        return nn.Dense(self.num_classes)(jnp.mean(x, axis=(1, 2)))[:, None, :], None


@pytest.fixture(params=["float32", "bfloat16"])
def tiny_models(monkeypatch, request):
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    monkeypatch.setattr(siglip, "Module", lambda num_classes, **_: TinyVision(num_classes))
    base = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype=request.param,
        action_dim=8,
        action_horizon=10,
        max_token_len=4,
    )
    moe_config = dataclasses.replace(base, action_moe=action_moe.ActionMoEConfig(rank=4, router_hidden_dim=8))
    dense = base.create(jax.random.key(0))
    # A pretrained adaRMS gate is nonzero. Random Pi05 initializes these gates
    # to zero, which would mask every FFN and make the gradient test vacuous.
    state = nnx.state(dense)
    for path, variable in state.flat_state().items():
        if "pre_ffw_norm_1" in path and path[-1] == "bias":
            variable.value = variable.value.at[..., -64:].set(1.0)
    nnx.update(dense, state)
    moe = moe_config.create(jax.random.key(1))
    loaded = weight_loaders._merge_params(  # noqa: SLF001 -- exercise the real checkpoint merge
        nnx.state(dense).to_pure_dict(),
        nnx.state(moe).to_pure_dict(),
        missing_regex=r"(?:action_moe_router/.*|PaliGemma/llm/layers/action_moe/.*)",
    )
    state = nnx.state(moe)
    state.replace_by_pure_dict(loaded)
    nnx.update(moe, state)
    obs = base.fake_obs(batch_size=2)
    obs = dataclasses.replace(
        obs,
        state=jax.random.normal(jax.random.key(3), (2, 8)),
        phase_targets=jnp.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=jnp.float32),
        phase_confidence=jnp.ones(2),
    )
    return dense, moe, obs, base.fake_act(batch_size=2), moe_config


def test_zero_init_preserves_dense_loss_and_actions(tiny_models):
    dense, moe, obs, actions, _ = tiny_models
    rng = jax.random.key(7)
    dense_loss = dense.compute_loss(rng, obs, actions)
    _, metrics = moe.compute_loss_with_metrics(rng, obs, actions, step=2000)
    np.testing.assert_allclose(metrics["flow_loss"], dense_loss.mean(), rtol=1e-6, atol=1e-6)
    noise = jax.random.normal(rng, actions.shape)
    np.testing.assert_allclose(
        dense.sample_actions(rng, obs, noise=noise, num_steps=2),
        moe.sample_actions(rng, obs, noise=noise, num_steps=2),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gradients_update_router_experts_and_original_weights(tiny_models):
    _, model, obs, actions, _ = tiny_models

    def objective(m):
        loss, metrics = m.compute_loss_with_metrics(jax.random.key(5), obs, actions, step=2000)
        return loss.mean(), metrics

    (loss, _), grads = nnx.value_and_grad(objective, has_aux=True)(model)
    assert np.isfinite(loss)
    flat = grads.flat_state()
    for fragment in ("action_moe_router", "action_moe", "action_out_proj", "mlp_1"):
        norm = sum(float(jnp.square(v.value).sum()) for p, v in flat.items() if fragment in p)
        assert norm > 0, fragment
    # B=0 implies A has zero flow gradient on the first step; B must update first.
    up_grad = next(v.value for p, v in flat.items() if "action_moe" in p and p[-1] == "up")
    assert np.all(np.sum(np.square(np.asarray(up_grad)), axis=(0, 2, 3)) > 0)
    nnx.update(model, optax.apply_updates(nnx.state(model), jax.tree.map(lambda g: -1e-3 * g, grads)))
    _, next_grads = nnx.value_and_grad(lambda m: objective(m)[0])(model)
    down_grad = next(v.value for p, v in next_grads.flat_state().items() if "action_moe" in p and p[-1] == "down")
    assert float(jnp.square(down_grad).sum()) > 0


def test_inference_ignores_labels_and_all_entrypoints_use_experts(tiny_models):
    _, model, obs, actions, config = tiny_models
    state = nnx.state(model)
    for path, variable in state.flat_state().items():
        if "action_moe" in path and path[-1] == "up":
            variable.value = jax.random.normal(jax.random.key(1), variable.value.shape) * 0.03
    nnx.update(model, state)
    rng = jax.random.key(8)
    noise = jax.random.normal(rng, actions.shape)
    no_labels = dataclasses.replace(obs, phase_targets=None, phase_confidence=None)
    output = model.sample_actions(rng, no_labels, noise=noise, num_steps=2)
    changed = dataclasses.replace(obs, phase_targets=jnp.roll(obs.phase_targets, 1, axis=-1))
    np.testing.assert_array_equal(output, model.sample_actions(rng, changed, noise=noise, num_steps=2))
    for method in (
        model.sample_actions_with_action_latent,
        model.sample_actions_with_action_latents,
        model.sample_actions_with_safe_pre_velocity,
    ):
        np.testing.assert_allclose(method(rng, no_labels, noise=noise, num_steps=2)[0], output, rtol=1e-6, atol=1e-6)
    restored = config.load(nnx.state(model).to_pure_dict())
    np.testing.assert_array_equal(restored.sample_actions(rng, no_labels, noise=noise, num_steps=2), output)
    # Turning off trained adapters must actually change actions, not just logging.
    for path, variable in state.flat_state().items():
        if "action_moe" in path and path[-1] == "up":
            variable.value = jnp.zeros_like(variable.value)
    nnx.update(model, state)
    assert not np.allclose(model.sample_actions(rng, no_labels, noise=noise, num_steps=2), output)


def test_router_masking_and_top_k():
    config = action_moe.ActionMoEConfig(top_k=2)
    router = action_moe.ActionRouter(8, 4, config, rngs=nnx.Rngs(0))
    tokens = jax.random.normal(jax.random.key(0), (2, 3, 8))
    mask = jnp.array([[True, True, False], [True, False, False]])
    state = jnp.ones((2, 4))
    gates, probs = router(tokens, mask, state)
    masked = jnp.where(mask[..., None], tokens, 1e4)
    np.testing.assert_array_equal(router(masked, mask, state)[0], gates)
    np.testing.assert_allclose(gates.sum(-1), 1, atol=1e-6)
    np.testing.assert_equal((np.asarray(gates) > 0).sum(-1), 2)
    assert np.all(np.asarray(probs) > 0)


def test_phase_loss_masks_unknown_labels_and_warms_up():
    config = action_moe.ActionMoEConfig(balance_loss_weight=0)
    probs = jnp.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
    targets = jnp.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.float32)
    confidence = jnp.array([1.0, 0.0])
    early, _ = action_moe.routing_loss(probs, targets, confidence, config, 0)
    late, metrics = action_moe.routing_loss(probs, targets, confidence, config, 2000)
    assert float(early) == 0
    np.testing.assert_allclose(late, -np.log(0.7) * config.phase_loss_weight, rtol=1e-6)
    assert float(metrics["moe/phase_accuracy"]) == 1


def test_full_config_and_strict_dense_weight_loading(monkeypatch):
    config = training_config.get_config("pi05_maniskill_phase_moe_full")
    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    params = nnx.state(model)
    assert not params.filter(config.freeze_filter).flat_state()
    assert len(params.filter(config.trainable_filter).flat_state()) == len(params.flat_state())
    ref = {"original": {"kernel": np.ones((2, 2))}, "action_moe_router": {"kernel": np.zeros((2, 4))}}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(model_lib, "restore_params", lambda *_, **__: {})
    loaded = config.weight_loader.load(ref)
    assert "action_moe_router" in loaded
    assert "original" not in loaded  # original missing keys must remain missing, so train validation fails


def test_jitted_training_checkpoint_and_resume(tiny_models, tmp_path):
    _, model, obs, actions, model_config = tiny_models
    spec = importlib.util.spec_from_file_location(
        "openpi_train_moe_test", Path(__file__).parents[3] / "scripts/train.py"
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    config = dataclasses.replace(training_config.get_config("pi05_maniskill_phase_moe_full"), model=model_config)
    params = nnx.state(model)
    optimizer = optax.adam(1e-3)
    state = training_utils.TrainState(
        step=jnp.array(2000),
        params=params,
        model_def=nnx.graphdef(model),
        tx=optimizer,
        opt_state=optimizer.init(params),
        ema_decay=0.99,
        ema_params=params,
    )
    step = jax.jit(functools.partial(train.train_step, config))
    rng = jax.random.key(4)
    updated, metrics = step(rng, state, (obs, actions))
    assert int(updated.step) == 2001
    assert float(metrics["grad_norm"]) > 0
    assert float(metrics["moe/phase_ramp"]) == 1
    assert np.isfinite(float(metrics["loss"]))

    class Loader:
        def data_config(self):
            return training_config.DataConfig()

        def __iter__(self):
            return iter(())

    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints", keep_period=None, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, updated, Loader(), step=2001)
        manager.wait_until_finished()
        restored = checkpoints.restore_state(manager, updated, Loader(), step=2001)
        for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(updated), strict=True):
            np.testing.assert_array_equal(actual, expected)
        expected_next, _ = step(rng, updated, (obs, actions))
        resumed_next, resumed_metrics = step(rng, restored, (obs, actions))
        for actual, expected in zip(jax.tree.leaves(resumed_next), jax.tree.leaves(expected_next), strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert int(resumed_next.step) == 2002
        assert float(resumed_metrics["moe/phase_ramp"]) == 1
        # The same saved EMA weights are loadable for inference without weak labels.
        restored_params = model_lib.restore_params(tmp_path / "checkpoints/2001/params", dtype=jnp.float32)
        policy_model = model_config.load(restored_params)
        no_labels = dataclasses.replace(obs, phase_targets=None, phase_confidence=None)
        prediction = policy_model.sample_actions(rng, no_labels, num_steps=2)
        assert prediction.shape == actions.shape
        assert np.isfinite(prediction).all()
    finally:
        manager.close()
