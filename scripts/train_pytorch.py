"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def _logit_normal_sample(batch_size, mean, std, device):
    return torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32) * std + mean)


def _mean_action_mse(pred, target):
    return torch.mean((pred.float() - target.float()) ** 2, dim=tuple(range(1, pred.ndim)))


def _sum_action_mse(pred, target):
    return torch.sum((pred.float() - target.float()) ** 2, dim=tuple(range(1, pred.ndim)))


def _adaptive_weighted_loss(per_sample_loss, normalizer_loss, eps, power, min_denom):
    weights = torch.clamp(normalizer_loss.float() + eps, min=eps).pow(power)
    if min_denom is not None:
        weights = torch.clamp(weights, min=min_denom)
    weights = weights.detach()
    return per_sample_loss / weights


def _assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        finite = torch.isfinite(tensor)
        num_bad = tensor.numel() - finite.sum().item()
        raise FloatingPointError(f"{name} contains {num_bad} non-finite values out of {tensor.numel()}.")


def _linear_warmup_weight(max_weight, warmup_steps, step):
    if max_weight <= 0:
        return 0.0
    if warmup_steps <= 0:
        return float(max_weight)
    return float(max_weight) * min(1.0, float(step + 1) / float(warmup_steps))


def compute_dmf_mse_loss(model, observation, actions, config, global_step):
    """DMF objective adapted to pi0 action chunks.

    The FM anchor stays aligned with the official pi0.5 loss. The auxiliary
    branch is configurable: "mf" keeps the previous MeanFlow target, while
    "imf" uses the iMeanFlow-C consistency target plus an instant-velocity loss.
    """
    model_to_call = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    device = actions.device
    batch_size = actions.shape[0]

    images, img_masks, lang_tokens, lang_masks, state = model_to_call.preprocess_observation_for_velocity(
        observation, train=True
    )

    # Auxiliary flow-matching loss. Use r=t so this path is identical to the
    # original pi0 conditioning under the parameter-free t/r fusion.
    noise_fm = torch.randn_like(actions)
    t_fm = model_to_call.sample_time(batch_size, device)
    x_t_fm = t_fm[:, None, None] * noise_fm + (1 - t_fm[:, None, None]) * actions
    v_tgt_fm = noise_fm - actions
    v_fm = model_to_call.predict_velocity_from_preprocessed(
        images, img_masks, lang_tokens, lang_masks, state, x_t_fm, t_fm, t_fm
    )
    _assert_finite_tensor("v_fm", v_fm)
    fm_loss = _mean_action_mse(v_fm, v_tgt_fm)
    _assert_finite_tensor("fm_loss", fm_loss)

    # MeanFlow loss. t >= r, matching DMF's interval convention. The interval
    # length is capped because long intervals make the JVP target high variance
    # early in fine-tuning.
    noise_mf = torch.randn_like(actions)
    ln_1 = _logit_normal_sample(batch_size, mean=-0.2, std=1.0, device=device)
    ln_2 = _logit_normal_sample(batch_size, mean=0.2, std=1.0, device=device)
    t_mf = torch.maximum(ln_1, ln_2)
    r_mf = torch.minimum(ln_1, ln_2)
    if config.pytorch_dmf_max_interval is not None:
        r_mf = torch.maximum(r_mf, t_mf - config.pytorch_dmf_max_interval)
    x_t_mf = t_mf[:, None, None] * noise_mf + (1 - t_mf[:, None, None]) * actions
    v_tgt_mf = noise_mf - actions

    def model_fn(x_t, t, r):
        return model_to_call.predict_velocity_from_preprocessed(
            images, img_masks, lang_tokens, lang_masks, state, x_t, t, r
        )

    primals = (x_t_mf, t_mf, r_mf)
    tangents = (v_tgt_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf))

    extra_losses = {
        "fm_loss": fm_loss.mean().detach(),
        "dmf_interval": (t_mf - r_mf).mean().detach(),
    }

    if config.pytorch_dmf_loss_type == "mf":
        u, du_dt = torch.func.jvp(model_fn, primals, tangents)
        _assert_finite_tensor("u", u)
        _assert_finite_tensor("du_dt", du_dt)
        u_tgt = v_tgt_mf + (r_mf - t_mf)[:, None, None] * du_dt
        if config.pytorch_dmf_target_clip is not None:
            u_tgt = torch.clamp(u_tgt, -config.pytorch_dmf_target_clip, config.pytorch_dmf_target_clip)
        mf_loss = _mean_action_mse(u, u_tgt.detach())
        _assert_finite_tensor("mf_loss", mf_loss)
        extra_losses.update(
            {
                "mf_loss": mf_loss.mean().detach(),
                "du_dt_norm": du_dt.float().norm(dim=-1).mean().detach(),
            }
        )
    elif config.pytorch_dmf_loss_type == "imf":
        u, du_dt = torch.func.jvp(model_fn, primals, tangents)
        _assert_finite_tensor("u", u)
        _assert_finite_tensor("du_dt", du_dt)

        v_inst = model_fn(x_t_mf, t_mf, t_mf)
        _assert_finite_tensor("v_inst", v_inst)

        imf_target = v_tgt_mf.detach()
        imf_pred = u + (t_mf - r_mf)[:, None, None] * du_dt.detach()
        if config.pytorch_dmf_target_clip is not None:
            imf_pred = torch.clamp(imf_pred, -config.pytorch_dmf_target_clip, config.pytorch_dmf_target_clip)
            imf_target = torch.clamp(imf_target, -config.pytorch_dmf_target_clip, config.pytorch_dmf_target_clip)

        imf_u_raw = _mean_action_mse(imf_pred, imf_target)
        imf_v_raw = _mean_action_mse(v_inst, imf_target)
        imf_u_normalizer = _sum_action_mse(imf_pred, imf_target)
        imf_v_normalizer = _sum_action_mse(v_inst, imf_target)
        _assert_finite_tensor("imf_u_loss", imf_u_raw)
        _assert_finite_tensor("imf_v_loss", imf_v_raw)
        if config.pytorch_imf_adaptive_weight:
            imf_u_loss = _adaptive_weighted_loss(
                imf_u_raw,
                imf_u_normalizer,
                config.pytorch_imf_adaptive_eps,
                config.pytorch_imf_adaptive_p,
                config.pytorch_imf_adaptive_min_denom,
            )
            imf_v_loss = _adaptive_weighted_loss(
                imf_v_raw,
                imf_v_normalizer,
                config.pytorch_imf_adaptive_eps,
                config.pytorch_imf_adaptive_p,
                config.pytorch_imf_adaptive_min_denom,
            )
        else:
            imf_u_loss = imf_u_raw
            imf_v_loss = imf_v_raw

        mf_loss = config.pytorch_imf_u_weight * imf_u_loss + config.pytorch_imf_v_weight * imf_v_loss
        _assert_finite_tensor("mf_loss", mf_loss)
        extra_losses.update(
            {
                "mf_loss": mf_loss.mean().detach(),
                "imf_u_loss": imf_u_raw.mean().detach(),
                "imf_v_loss": imf_v_raw.mean().detach(),
                "imf_u_weighted_loss": imf_u_loss.mean().detach(),
                "imf_v_weighted_loss": imf_v_loss.mean().detach(),
                "du_dt_norm": du_dt.float().norm(dim=-1).mean().detach(),
            }
        )
    else:
        raise ValueError(f"Unsupported pytorch_dmf_loss_type: {config.pytorch_dmf_loss_type}")

    mf_weight = _linear_warmup_weight(
        config.pytorch_dmf_mf_weight,
        config.pytorch_dmf_mf_warmup_steps,
        global_step,
    )
    loss = fm_loss + mf_weight * mf_loss
    extra_losses["mf_weight"] = torch.tensor(mf_weight, device=device)
    extra_losses["dmf_aux_contribution"] = (mf_weight * mf_loss).mean().detach()
    return loss, extra_losses


def configure_dmf_split(model, config, is_main):
    if config.pytorch_train_objective != "dmf":
        return None

    model_to_configure = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if not model_to_configure.pi05:
        model_to_configure.set_dmf_depth(None)
        if is_main:
            logging.info("DMF objective enabled without pi0.5; using parameter-free dual-time MVP conditioning.")
        return None

    num_layers = len(model_to_configure.paligemma_with_expert.gemma_expert.model.layers)
    dmf_depth = config.pytorch_dmf_depth if config.pytorch_dmf_depth is not None else num_layers // 2
    model_to_configure.set_dmf_depth(dmf_depth)
    if is_main:
        logging.info(
            f"Enabled pi0.5 DMF action-expert split: layers < {dmf_depth} use t, layers >= {dmf_depth} use r"
        )
    return dmf_depth


def freeze_vlm_parameters(model, is_main):
    model_to_freeze = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    for param in model_to_freeze.paligemma_with_expert.paligemma.parameters():
        param.requires_grad = False
    trainable = sum(param.numel() for param in model_to_freeze.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model_to_freeze.parameters() if not param.requires_grad)
    if is_main:
        logging.info(f"Frozen PaliGemma VLM branch. Trainable params={trainable:,}, frozen params={frozen:,}")


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    dmf_depth = configure_dmf_split(model, config, is_main)
    if config.pytorch_freeze_vlm:
        freeze_vlm_parameters(model, is_main)

    if hasattr(model, "gradient_checkpointing_enable") and config.pytorch_train_objective != "dmf":
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    elif config.pytorch_train_objective == "dmf":
        enable_gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for DMF JVP compatibility")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}, objective={config.pytorch_train_objective}, dmf_depth={dmf_depth}, freeze_vlm={config.pytorch_freeze_vlm}"
        )
        if config.pytorch_train_objective == "dmf":
            logging.info(
                "DMF stability: loss_type=%s, mf_weight=%s, mf_warmup_steps=%s, max_interval=%s, target_clip=%s",
                config.pytorch_dmf_loss_type,
                config.pytorch_dmf_mf_weight,
                config.pytorch_dmf_mf_warmup_steps,
                config.pytorch_dmf_max_interval,
                config.pytorch_dmf_target_clip,
            )
            if config.pytorch_dmf_loss_type == "imf":
                logging.info(
                    "iMF-C: u_weight=%s, v_weight=%s, adaptive=%s, adaptive_p=%s, adaptive_eps=%s, adaptive_min_denom=%s",
                    config.pytorch_imf_u_weight,
                    config.pytorch_imf_v_weight,
                    config.pytorch_imf_adaptive_weight,
                    config.pytorch_imf_adaptive_p,
                    config.pytorch_imf_adaptive_eps,
                    config.pytorch_imf_adaptive_min_denom,
                )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            extra_losses = {}
            if config.pytorch_train_objective == "dmf":
                losses, extra_losses = compute_dmf_mse_loss(model, observation, actions, config, global_step)
            else:
                losses = model(observation, actions)
                # Ensure losses is a tensor and handle different return types
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        **{key: value.item() for key, value in extra_losses.items()},
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_fm_loss = sum(info["fm_loss"] for info in infos if "fm_loss" in info) / max(
                    1, sum("fm_loss" in info for info in infos)
                )
                avg_mf_loss = sum(info["mf_loss"] for info in infos if "mf_loss" in info) / max(
                    1, sum("mf_loss" in info for info in infos)
                )
                avg_mf_weight = sum(info["mf_weight"] for info in infos if "mf_weight" in info) / max(
                    1, sum("mf_weight" in info for info in infos)
                )
                avg_dmf_interval = sum(info["dmf_interval"] for info in infos if "dmf_interval" in info) / max(
                    1, sum("dmf_interval" in info for info in infos)
                )
                avg_du_dt_norm = sum(info["du_dt_norm"] for info in infos if "du_dt_norm" in info) / max(
                    1, sum("du_dt_norm" in info for info in infos)
                )

                def _avg_extra(key):
                    vals = [info[key] for info in infos if key in info]
                    return sum(vals) / len(vals) if len(vals) > 0 else None

                avg_imf_u_loss = _avg_extra("imf_u_loss")
                avg_imf_v_loss = _avg_extra("imf_v_loss")
                avg_imf_u_weighted_loss = _avg_extra("imf_u_weighted_loss")
                avg_imf_v_weighted_loss = _avg_extra("imf_v_weighted_loss")
                avg_dmf_aux_contribution = _avg_extra("dmf_aux_contribution")

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                loss_parts = ""
                if any("fm_loss" in info for info in infos):
                    loss_parts = (
                        f" fm_loss={avg_fm_loss:.4f} mf_loss={avg_mf_loss:.4f}"
                        f" mf_w={avg_mf_weight:.3f} interval={avg_dmf_interval:.3f} du_dt={avg_du_dt_norm:.3f}"
                    )
                    if avg_dmf_aux_contribution is not None:
                        loss_parts += f" aux={avg_dmf_aux_contribution:.4f}"
                    if avg_imf_u_loss is not None and avg_imf_v_loss is not None:
                        loss_parts += f" imf_u={avg_imf_u_loss:.4f} imf_v={avg_imf_v_loss:.4f}"
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f}{loss_parts} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f}{loss_parts} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if any("fm_loss" in info for info in infos):
                        log_payload["fm_loss"] = avg_fm_loss
                        log_payload["mf_loss"] = avg_mf_loss
                        log_payload["mf_weight"] = avg_mf_weight
                        log_payload["dmf_interval"] = avg_dmf_interval
                        log_payload["du_dt_norm"] = avg_du_dt_norm
                        if avg_imf_u_loss is not None:
                            log_payload["imf_u_loss"] = avg_imf_u_loss
                        if avg_imf_v_loss is not None:
                            log_payload["imf_v_loss"] = avg_imf_v_loss
                        if avg_imf_u_weighted_loss is not None:
                            log_payload["imf_u_weighted_loss"] = avg_imf_u_weighted_loss
                        if avg_imf_v_weighted_loss is not None:
                            log_payload["imf_v_weighted_loss"] = avg_imf_v_weighted_loss
                        if avg_dmf_aux_contribution is not None:
                            log_payload["dmf_aux_contribution"] = avg_dmf_aux_contribution
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
