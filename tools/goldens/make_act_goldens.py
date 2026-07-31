#!/usr/bin/env python
"""Generate the PyTorch differential oracle for Rerobot's ACT port.

Run against upstream `lerobot` at f37be3edbee60f3a09a5183788b91eb19f0c07d1. See
`tools/goldens/README.md`. Not part of `cargo test`: the Rust tests read the files
this writes and never invoke Python.

What this pins, all at the reduced state-only configuration and all produced by
upstream's own `ACTPolicy`:

* the normalized batch, computed by upstream's `NormalizerProcessorStep` from the
  committed dataset fixture's `meta/stats.json`;
* the forward pass -- predicted actions, and the latent distribution's `mu` and
  `log(sigma^2)`;
* the loss -- masked L1, the mean KL divergence, and the weighted total;
* the gradients of ten representative parameters, chosen to cover every distinct
  path to the loss (see `REPRESENTATIVE_GRADIENTS`);
* the total gradient norm `clip_grad_norm_` reports;
* the parameters after one `torch.optim.AdamW` step at ACT's own preset.

Three things are held fixed so that the comparison is about the architecture
rather than about two different random number generators:

1. `dropout = 0.0`. `ACTPolicy.forward` only takes its VAE branch in training
   mode, and training mode is also what activates `nn.Dropout`. A dropout mask
   comes from PyTorch's Mersenne stream, which Rerobot does not reproduce (see
   `rerobot_core::random`), so the oracle runs where the mask is the identity on
   both sides.
2. The reparameterization noise is supplied, not drawn: `torch.randn_like` is
   replaced for the duration of the forward pass. Rerobot's
   `Randomness::Fixed` is the matching hook.
3. The weights are exported and loaded rather than initialized twice. Rerobot's
   initializer draws torch's *distributions* from its own stream, so two
   same-seeded runs do not agree on weights and could not be compared without
   this.

Outputs, all under `crates/rerobot-train/tests/fixtures/goldens/`:

    act_oracle.json                  scalars, config, provenance, tensor index
    act_oracle_weights.safetensors   `ACTPolicy.state_dict()` before the step
    act_oracle_tensors.safetensors   inputs, outputs, gradients, post-step values
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "crates/rerobot-train/tests/fixtures/state_only"
OUT = ROOT / "crates/rerobot-train/tests/fixtures/goldens"

UPSTREAM_COMMIT = "f37be3edbee60f3a09a5183788b91eb19f0c07d1"

# Must match `crates/rerobot-train/tests/common/mod.rs::reduced_config`, except for
# `dropout`, which the oracle pins at zero for the reason in the module docstring.
CONFIG = {
    "chunk_size": 2,
    "n_action_steps": 2,
    "dim_model": 32,
    "n_heads": 4,
    "dim_feedforward": 64,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "n_vae_encoder_layers": 1,
    "latent_dim": 8,
    "use_vae": True,
    "pre_norm": False,
    "dropout": 0.0,
    "kl_weight": 10.0,
    "feedforward_activation": "relu",
    "optimizer_lr": 1e-5,
    "optimizer_weight_decay": 1e-4,
    "grad_clip_norm": 10.0,
}

# Which frames of the fixture make up the batch. Deterministic by construction, and
# chosen so that the batch exercises both sides of the action-padding mask: frame 0
# sits well inside its episode, and frame 3 is the last frame of a four-frame
# episode, so the second half of its two-step action chunk is clamped and flagged.
# A batch of two interior frames would leave the mask, and therefore the `num_valid`
# divisor of the masked L1, entirely untested.
BATCH_FRAMES = [0, 3]

# Ten parameters covering every distinct route a gradient can take to the loss.
# The two position embeddings are the sharpest of them: they feed the attention
# *logits* and nothing else, so a softmax whose backward pass does not propagate
# leaves them at exactly zero while every other number still looks healthy. That
# defect really occurred during this port, and this oracle is what would catch it
# coming back.
REPRESENTATIVE_GRADIENTS = [
    "model.action_head.weight",
    "model.action_head.bias",
    "model.decoder.norm.weight",
    "model.decoder_pos_embed.weight",
    "model.encoder_1d_feature_pos_embed.weight",
    "model.encoder.layers.0.self_attn.in_proj_weight",
    "model.decoder.layers.0.multihead_attn.in_proj_weight",
    "model.encoder_robot_state_input_proj.weight",
    "model.encoder_env_state_input_proj.weight",
    "model.vae_encoder_latent_output_proj.weight",
    "model.vae_encoder_cls_embed.weight",
]

# Parameters whose post-AdamW-step value is recorded. One from the head and one
# bias, which between them exercise the decoupled weight decay and the
# bias-corrected moment update on tensors of both ranks.
POST_STEP_PARAMETERS = ["model.action_head.weight", "model.action_head.bias"]


def build_config() -> ACTConfig:
    config = ACTConfig(
        chunk_size=CONFIG["chunk_size"],
        n_action_steps=CONFIG["n_action_steps"],
        dim_model=CONFIG["dim_model"],
        n_heads=CONFIG["n_heads"],
        dim_feedforward=CONFIG["dim_feedforward"],
        n_encoder_layers=CONFIG["n_encoder_layers"],
        n_decoder_layers=CONFIG["n_decoder_layers"],
        n_vae_encoder_layers=CONFIG["n_vae_encoder_layers"],
        latent_dim=CONFIG["latent_dim"],
        use_vae=CONFIG["use_vae"],
        pre_norm=CONFIG["pre_norm"],
        dropout=CONFIG["dropout"],
        kl_weight=CONFIG["kl_weight"],
        feedforward_activation=CONFIG["feedforward_activation"],
        optimizer_lr=CONFIG["optimizer_lr"],
        optimizer_weight_decay=CONFIG["optimizer_weight_decay"],
        device="cpu",
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )
    # What `make_policy` fills in from the dataset. The fixture's three features,
    # classified by `dataset_to_policy_features`.
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(2,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    return config


def read_fixture_frames() -> dict[str, torch.Tensor]:
    """The raw batch, straight out of the fixture's parquet and episode boundaries."""
    import pandas as pd

    data = pd.read_parquet(FIXTURE / "data/chunk-000/file-000.parquet")
    episodes = pd.read_parquet(FIXTURE / "meta/episodes/chunk-000/file-000.parquet")
    ep_from = int(episodes["dataset_from_index"].iloc[0])
    ep_to = int(episodes["dataset_to_index"].iloc[0])

    def column(name: str, rows: list[int]) -> torch.Tensor:
        return torch.tensor(
            np.stack([np.asarray(data[name].iloc[row], dtype=np.float32) for row in rows]),
            dtype=torch.float32,
        )

    chunk = CONFIG["chunk_size"]
    action_rows: list[list[int]] = []
    action_is_pad: list[list[bool]] = []
    for frame in BATCH_FRAMES:
        # `DatasetReader._get_query_indices`, for the action window.
        window = [max(ep_from, min(ep_to - 1, frame + delta)) for delta in range(chunk)]
        action_rows.append(window)
        action_is_pad.append(
            [(frame + delta < ep_from) or (frame + delta >= ep_to) for delta in range(chunk)]
        )

    actions = torch.stack([column("action", rows) for rows in action_rows])
    return {
        "observation.state": column("observation.state", BATCH_FRAMES),
        "observation.environment_state": column("observation.environment_state", BATCH_FRAMES),
        "action": actions,
        "action_is_pad": torch.tensor(action_is_pad, dtype=torch.bool),
    }


def normalize(batch: dict[str, torch.Tensor], config: ACTConfig) -> dict[str, torch.Tensor]:
    """Normalize with upstream's own processor step, from the fixture's stats.json."""
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    raw_stats = json.loads((FIXTURE / "meta/stats.json").read_text())
    stats = {
        key: {name: np.asarray(values, dtype=np.float32) for name, values in entry.items()}
        for key, entry in raw_stats.items()
    }
    step = NormalizerProcessorStep(
        features={**config.input_features, **config.output_features},
        norm_map=config.normalization_mapping,
        stats=stats,
    )
    out = dict(batch)
    for key, feature in {**config.input_features, **config.output_features}.items():
        out[key] = step._apply_transform(batch[key], key, feature.type)
    return out


def write_processor_goldens(config: ACTConfig) -> None:
    """Save the pre/postprocessor artifacts upstream writes into every checkpoint.

    `lerobot/common/train_utils.py:145-155` calls `preprocessor.save_pretrained` and
    `postprocessor.save_pretrained` on the `pretrained_model/` directory, so a
    checkpoint without these four files is not upstream's layout and has lost the
    normalization state a deployment needs. These goldens are what
    `crates/rerobot-train/tests/processor.rs` compares against, so the Rust writer is
    pinned to upstream's own output rather than to itself.
    """
    from lerobot.policies.factory import make_pre_post_processors

    raw_stats = json.loads((FIXTURE / "meta/stats.json").read_text())
    stats = {
        key: {name: np.asarray(values, dtype=np.float32) for name, values in entry.items()}
        for key, entry in raw_stats.items()
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=None, dataset_stats=stats
    )
    out = OUT / "processors"
    if out.exists():
        for stale in out.iterdir():
            stale.unlink()
    out.mkdir(parents=True, exist_ok=True)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)

    written = sorted(path.name for path in out.iterdir())
    expected = [
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
    ]
    if written != expected:
        raise SystemExit(f"unexpected processor artifacts: {written}")
    print(f"  processors  = {len(written)} files in {out.name}/")


def main() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    config = build_config()
    policy = ACTPolicy(config)
    policy.train()

    raw = read_fixture_frames()
    batch = normalize(raw, config)

    # A fixed, non-degenerate stand-in for `torch.randn_like(mu)`. The same values
    # `crates/rerobot-train/tests/goldens.rs` hands Rerobot.
    latent_noise = torch.tensor(
        [[((index % 7) - 3.0) / 4.0 for index in range(config.latent_dim)] for _ in BATCH_FRAMES],
        dtype=torch.float32,
    )
    # Row 1 must differ from row 0, or the batch axis would go unexercised.
    latent_noise[1] = torch.tensor(
        [((index % 5) - 2.0) / 3.0 for index in range(config.latent_dim)], dtype=torch.float32
    )

    weights = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}

    original_randn_like = torch.randn_like
    calls = 0

    def fixed_randn_like(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        if tensor.shape != latent_noise.shape:
            raise AssertionError(
                f"randn_like was called with {tuple(tensor.shape)}, expected "
                f"{tuple(latent_noise.shape)}; the oracle only knows how to fix the latent draw"
            )
        return latent_noise.clone()

    torch.randn_like = fixed_randn_like
    try:
        actions_hat, (mu, log_sigma_x2) = policy.model(dict(batch))
    finally:
        torch.randn_like = original_randn_like
    if calls != 1:
        raise SystemExit(f"expected exactly one randn_like call, saw {calls}")

    # `ACTPolicy.forward`, written out so the oracle does not depend on the policy
    # wrapper also being importable in isolation.
    abs_err = torch.nn.functional.l1_loss(batch["action"], actions_hat, reduction="none")
    valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
    num_valid = valid_mask.sum() * abs_err.shape[-1]
    l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
    mean_kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
    loss = l1_loss + mean_kld * config.kl_weight

    loss.backward()

    gradients = {}
    named = dict(policy.named_parameters())
    for name in REPRESENTATIVE_GRADIENTS:
        parameter = named[name]
        if parameter.grad is None:
            raise SystemExit(f"{name} has no gradient; the oracle cannot pin it")
        gradients[name] = parameter.grad.detach().clone()

    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), CONFIG["grad_clip_norm"])

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=CONFIG["optimizer_lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=CONFIG["optimizer_weight_decay"],
    )
    optimizer.step()

    post_step = {name: named[name].detach().clone() for name in POST_STEP_PARAMETERS}

    OUT.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(OUT / "act_oracle_weights.safetensors"))

    tensors = {
        "input/observation.state": batch["observation.state"],
        "input/observation.environment_state": batch["observation.environment_state"],
        "input/action": batch["action"],
        "input/action_is_pad": batch["action_is_pad"].to(torch.uint8),
        "input/latent_noise": latent_noise,
        "output/actions": actions_hat.detach(),
        "output/mu": mu.detach(),
        "output/log_sigma_x2": log_sigma_x2.detach(),
    }
    for name, tensor in gradients.items():
        tensors[f"grad/{name}"] = tensor
    for name, tensor in post_step.items():
        tensors[f"post_step/{name}"] = tensor
    save_file({key: value.contiguous() for key, value in tensors.items()},
              str(OUT / "act_oracle_tensors.safetensors"))

    meta = {
        "generator": "tools/goldens/make_act_goldens.py",
        "upstream_package": "lerobot",
        "upstream_version": "0.6.1",
        "upstream_commit": UPSTREAM_COMMIT,
        "torch_version": torch.__version__,
        "torch_seed": 0,
        "config": CONFIG,
        "batch_frames": BATCH_FRAMES,
        "dataset_fixture": "crates/rerobot-train/tests/fixtures/state_only",
        "scalars": {
            "l1_loss": float(l1_loss.item()),
            "kld_loss": float(mean_kld.item()),
            "total_loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "num_valid_scalars": int(num_valid.item()),
        },
        "state_dict_keys": sorted(weights),
        "gradient_keys": REPRESENTATIVE_GRADIENTS,
        "post_step_keys": POST_STEP_PARAMETERS,
        "optimizer": {
            "type": "adamw",
            "lr": CONFIG["optimizer_lr"],
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": CONFIG["optimizer_weight_decay"],
            "grad_clip_norm": CONFIG["grad_clip_norm"],
        },
    }
    (OUT / "act_oracle.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    write_processor_goldens(config)

    print(f"wrote {OUT}")
    print(f"  l1_loss     = {meta['scalars']['l1_loss']!r}")
    print(f"  kld_loss    = {meta['scalars']['kld_loss']!r}")
    print(f"  total_loss  = {meta['scalars']['total_loss']!r}")
    print(f"  grad_norm   = {meta['scalars']['grad_norm']!r}")
    print(f"  state dict  = {len(weights)} tensors")


if __name__ == "__main__":
    main()
