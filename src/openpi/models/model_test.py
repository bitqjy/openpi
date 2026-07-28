from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_frozen_action_latent_preserves_actions():
    key = jax.random.key(11)
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=10,
        max_token_len=8,
    )
    model = config.create(key)
    obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(12), (1, 10, 8))

    base_actions = model.sample_actions(key, obs, num_steps=2, noise=noise)
    latent_actions, latent = model.sample_actions_with_action_latent(key, obs, num_steps=2, noise=noise)
    _, other_latent = model.sample_actions_with_action_latent(
        jax.random.key(13),
        obs,
        num_steps=2,
        noise=jax.random.normal(jax.random.key(14), (1, 10, 8)),
    )
    temporal_actions, temporal_mean, temporal_latent = model.sample_actions_with_action_latents(
        key, obs, num_steps=2, noise=noise
    )
    jitted_temporal_actions, jitted_temporal_mean, jitted_temporal_latent = nnx_utils.module_jit(
        model.sample_actions_with_action_latents
    )(key, obs, num_steps=2, noise=noise)
    np.testing.assert_array_equal(np.asarray(latent_actions), np.asarray(base_actions))
    np.testing.assert_array_equal(np.asarray(temporal_actions), np.asarray(base_actions))
    np.testing.assert_array_equal(np.asarray(jitted_temporal_actions), np.asarray(base_actions))
    np.testing.assert_array_equal(np.asarray(temporal_mean), np.asarray(latent))
    np.testing.assert_array_equal(np.asarray(jitted_temporal_mean), np.asarray(temporal_mean))
    np.testing.assert_array_equal(np.asarray(jitted_temporal_latent), np.asarray(temporal_latent))
    assert latent.shape == (1, 64)
    assert temporal_latent.shape == (1, 5, 64)
    np.testing.assert_allclose(
        np.asarray(temporal_latent).mean(axis=1),
        np.asarray(latent),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.all(np.isfinite(np.asarray(latent)))
    assert np.all(np.isfinite(np.asarray(temporal_latent)))
    assert np.max(np.abs(np.asarray(latent) - np.asarray(other_latent))) > 1e-5


def test_pi0_temporal_action_latent_rejects_nondivisible_bins():
    key = jax.random.key(21)
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=8,
    )
    model = config.create(key)
    with pytest.raises(ValueError, match="must be divisible"):
        model.sample_actions_with_action_latents(
            key,
            config.fake_obs(batch_size=1),
            num_steps=1,
        )


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
