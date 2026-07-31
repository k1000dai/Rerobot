#!/usr/bin/env python
"""Check that upstream `lerobot` can read a checkpoint `lerobot-train` wrote.

Run against upstream at f37be3edbee60f3a09a5183788b91eb19f0c07d1. Not part of
`cargo test`, and not a fixture generator: this verifies interoperability in the
direction the Rust tests cannot, by loading Rerobot's output with upstream's own
code.

    cargo run -p rerobot-cli --bin lerobot-train -- \\
        --dataset.repo_id=rerobot/state_only_slice \\
        --dataset.root=crates/rerobot-train/tests/fixtures/state_only \\
        --output_dir=/tmp/rr-run --policy.type=act \\
        --steps=1 --batch_size=2 \\
        --policy.chunk_size=2 --policy.n_action_steps=2 \\
        --policy.dim_model=32 --policy.n_heads=4 --policy.dim_feedforward=64 \\
        --policy.n_encoder_layers=1 --policy.n_decoder_layers=1 \\
        --policy.n_vae_encoder_layers=1 --policy.latent_dim=8

    "$LEROBOT/.venv/bin/python" tools/goldens/verify_checkpoint_upstream.py \\
        /tmp/rr-run/checkpoints/000001

Four things are checked, and each would fail loudly rather than silently degrade:

1. `config.json` parses through upstream's Draccus loader into an `ACTConfig`,
   with the features Rerobot resolved from the dataset intact;
2. `model.safetensors` loads into a real `ACTPolicy` with `strict=True`, so every
   tensor name and shape is upstream's and none is missing or extra;
3. that policy runs a forward pass on Rerobot's trained weights;
4. `optimizer_param_groups.json` *and* `optimizer_state.safetensors` are loaded by
   upstream's own `lerobot.optim.optimizers.load_optimizer_state`, which is the
   function a resume calls. It reaches `Optimizer.load_state_dict`, so a missing or
   extra parameter-group key is a hard failure there;
5. the restored optimizer can take a step;
6. both processor artifacts rebuild into real `PolicyProcessorPipeline`s with the
   steps upstream's own `make_pre_post_processors` would have built, which is how a
   deployment recovers the normalization the weights were trained under.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <checkpoints/NNNNNN>")
    checkpoint = Path(sys.argv[1])
    pretrained = checkpoint / "pretrained_model"
    training_state = checkpoint / "training_state"
    for path in (
        pretrained / "config.json",
        pretrained / "model.safetensors",
        pretrained / "train_config.json",
        pretrained / "policy_preprocessor.json",
        pretrained / "policy_preprocessor_step_3_normalizer_processor.safetensors",
        pretrained / "policy_postprocessor.json",
        pretrained / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        training_state / "training_step.json",
        training_state / "optimizer_state.safetensors",
        training_state / "optimizer_param_groups.json",
    ):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    config = ACTConfig.from_pretrained(str(pretrained))
    print(f"1. config.json -> {type(config).__name__}")
    print(f"   chunk_size={config.chunk_size} dim_model={config.dim_model} "
          f"latent_dim={config.latent_dim} n_heads={config.n_heads}")
    print(f"   input_features={list(config.input_features)}")
    print(f"   output_features={list(config.output_features)}")
    if not config.input_features or not config.output_features:
        raise SystemExit("the checkpoint's features are empty; they were not resolved")

    policy = ACTPolicy(config)
    weights = load_file(str(pretrained / "model.safetensors"))
    result = policy.load_state_dict(weights, strict=True)
    print(f"2. model.safetensors -> load_state_dict(strict=True): {result}")

    policy.eval()
    state_dim = config.input_features["observation.state"].shape[0]
    env_dim = config.input_features["observation.environment_state"].shape[0]
    batch = {
        "observation.state": torch.zeros(2, state_dim),
        "observation.environment_state": torch.zeros(2, env_dim),
    }
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    if not bool(torch.isfinite(actions).all()):
        raise SystemExit("upstream inference on Rerobot's weights produced non-finite actions")
    print(f"3. upstream forward pass on Rerobot's weights -> {tuple(actions.shape)}, all finite")

    # The real thing: upstream's own loader, on the whole `training_state/`
    # directory. This reaches `Optimizer.load_state_dict`, which compares the saved
    # parameter groups' key *set* against the live optimizer's and raises
    # `ValueError: Dictionary keys do not match.` on any difference, and which
    # unflattens and installs `optimizer_state.safetensors` as real moment tensors.
    #
    # An earlier version of this script overlaid the saved group onto a fresh one and
    # never read the safetensors at all, which is why it did not notice that
    # `decoupled_weight_decay` was missing. Calling the loader is the point.
    from lerobot.optim.optimizers import load_optimizer_state

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.optimizer_weight_decay,
    )
    before = optimizer.state_dict()
    if before["state"]:
        raise SystemExit("the fresh optimizer already has state; the check would prove nothing")

    load_optimizer_state(optimizer, training_state)

    after = optimizer.state_dict()
    if not after["state"]:
        raise SystemExit(
            "lerobot.optim.load_optimizer_state installed no state; "
            "optimizer_state.safetensors was empty or was ignored"
        )
    groups = after["param_groups"]
    if len(groups) != 2:
        raise SystemExit(f"expected two parameter groups, got {len(groups)}")
    for index, group in enumerate(groups):
        if "decoupled_weight_decay" not in group:
            raise SystemExit(f"param group {index} lost decoupled_weight_decay")
    # Every entry must carry all three of torch's AdamW slots, as real tensors.
    for index, entry in sorted(after["state"].items()):
        for slot in ("step", "exp_avg", "exp_avg_sq"):
            if slot not in entry:
                raise SystemExit(f"state[{index}] has no {slot}")
            if not isinstance(entry[slot], torch.Tensor):
                raise SystemExit(f"state[{index}][{slot}] is not a tensor")
    steps = {int(entry["step"].item()) for entry in after["state"].values()}
    print(
        f"4. lerobot.optim.load_optimizer_state -> {len(after['state'])} parameters restored, "
        f"step values {sorted(steps)}, lr={groups[0]['lr']}"
    )

    # And the restored optimizer can actually take a step, which is what a resume
    # would do next. A malformed moment tensor would surface here rather than later.
    for parameter in policy.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    print("5. the restored optimizer took a step without raising")

    # The processor artifacts, through upstream's own pipeline loader. This is what a
    # deployment does to recover the normalization the weights were trained under, so
    # a checkpoint that cannot be rebuilt here is not a usable pretrained artifact.
    from lerobot.processor import PolicyProcessorPipeline

    for name, expected in [
        (
            "policy_preprocessor",
            [
                "RenameObservationsProcessorStep",
                "AddBatchDimensionProcessorStep",
                "DeviceProcessorStep",
                "NormalizerProcessorStep",
            ],
        ),
        ("policy_postprocessor", ["UnnormalizerProcessorStep", "DeviceProcessorStep"]),
    ]:
        pipeline = PolicyProcessorPipeline.from_pretrained(
            str(pretrained), config_filename=f"{name}.json"
        )
        built = [type(step).__name__ for step in pipeline.steps]
        if built != expected:
            raise SystemExit(f"{name} rebuilt as {built}, expected {expected}")
        print(f"6. {name} -> upstream rebuilt {len(built)} steps: {built}")

    step = json.loads((training_state / "training_step.json").read_text())
    print(f"   training_step.json = {step}")
    print("\nupstream lerobot can read this checkpoint.")


if __name__ == "__main__":
    main()
