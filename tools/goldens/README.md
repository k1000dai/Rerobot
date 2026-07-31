# Golden fixture generators

Nothing in this directory is part of the Rust build, and none of it runs during
`cargo test`. These are the Python scripts that produced the committed fixtures
and differential-oracle data under
`crates/rerobot-train/tests/fixtures/`, executed once against upstream
`lerobot` at the pinned commit and then never again unless a fixture has to
change.

They exist so that a reviewer can check the fixtures against upstream rather
than against Rerobot: every number the Rust tests compare to came out of
PyTorch, NumPy or pyarrow here, not out of the code under test.

| Script | Produces | What it pins |
| --- | --- | --- |
| `make_dataset_fixture.py` | `tests/fixtures/state_only/` | A real state-only LeRobot v3.0 dataset, written by upstream's own `LeRobotDataset.create` / `add_frame` / `save_episode` / `finalize`. Rerobot's reader is therefore tested against upstream's writer, not against a hand-rolled parquet file. |
| `make_act_goldens.py` | `tests/fixtures/goldens/` | The whole ACT training step at the reduced state-only configuration: the normalized batch, the forward pass, all three loss terms, eleven representative gradients, the clipped gradient norm, and the parameters after one AdamW step. Consumed by `crates/rerobot-train/tests/goldens.rs`. |
| `verify_checkpoint_upstream.py` | nothing — it is a check | That upstream `lerobot` can *read* a checkpoint `lerobot-train` wrote: the config through Draccus, the weights into a real `ACTPolicy` with `strict=True`, a forward pass on them, and the parameter groups into a real `torch.optim.AdamW`. This is the one direction the Rust tests cannot check. |

## The ACT oracle

`make_act_goldens.py` writes three files:

| File | Contents |
| --- | --- |
| `act_oracle.json` | the loss scalars, the gradient norm, the configuration and batch the oracle was generated at, the upstream commit and torch version, and the tensor index |
| `act_oracle_weights.safetensors` | `ACTPolicy.state_dict()` before the step — 62 tensors under upstream's own names |
| `act_oracle_tensors.safetensors` | the normalized inputs, the fixed latent draw, the forward outputs, the eleven gradients, and the post-step parameters |

Three degrees of freedom have to be removed before two implementations of ACT can
be compared at all, and the generator removes each one deliberately:

* **the weights** — Rerobot draws torch's initialization *distributions* from its
  own generator, so two same-seeded runs do not agree on a single weight. The
  oracle exports the state dict and the Rust test loads it, which doubles as a
  test of the checkpoint format.
* **the latent draw** — `torch.randn_like` is replaced for the duration of the
  forward pass, and the same tensor is handed to Rerobot's `Randomness::Fixed`.
* **dropout** — see below.

The batch is frames 0 and 3 of the fixture: frame 0 is interior, and frame 3 is
the last frame of a four-frame episode, so half of its two-step action chunk is
clamped and flagged. A batch of two interior frames would leave the padding mask
and the masked-L1 divisor untested.

### Why the oracle configuration sets `dropout = 0.0`

`ACTPolicy.forward` only takes its VAE branch when the module is in training mode,
and training mode is also what activates `nn.Dropout`. A dropout mask is drawn
from PyTorch's Mersenne stream, which Rerobot does not reproduce (see
`rerobot_core::random`). The oracle therefore runs at `dropout = 0.0`, where the
mask is the identity on both sides and the comparison is about the architecture
rather than about the RNG.

### Regenerating it

The Rust test refuses fixtures generated at a different upstream commit, and
checks the recorded configuration against the one it builds, so a regeneration
that changes either fails rather than silently moving the goalposts.

## Running them

Upstream `lerobot` at `f37be3edbee60f3a09a5183788b91eb19f0c07d1` must be
checked out and installed, along with `torch`, `numpy`, `pandas` and `pyarrow`.
None of these scripts touches the network.

```shell
LEROBOT=/path/to/lerobot
"$LEROBOT/.venv/bin/python" tools/goldens/make_dataset_fixture.py
"$LEROBOT/.venv/bin/python" tools/goldens/make_act_goldens.py
```

The first two write into the repository and are idempotent: `make_act_goldens.py`
seeds torch explicitly, so rerunning either against the same upstream commit
reproduces the same output, apart from the parquet footers, which embed the
writer's version string.

The third takes the path of a checkpoint an actual run produced:

```shell
cargo run -p rerobot-cli --bin lerobot-train -- \
    --dataset.repo_id=rerobot/state_only_slice \
    --dataset.root=crates/rerobot-train/tests/fixtures/state_only \
    --output_dir=/tmp/rr-run --policy.type=act --steps=1 --batch_size=2 \
    --policy.chunk_size=2 --policy.n_action_steps=2 --policy.dim_model=32 \
    --policy.n_heads=4 --policy.dim_feedforward=64 --policy.n_encoder_layers=1 \
    --policy.n_decoder_layers=1 --policy.n_vae_encoder_layers=1 --policy.latent_dim=8

"$LEROBOT/.venv/bin/python" tools/goldens/verify_checkpoint_upstream.py \
    /tmp/rr-run/checkpoints/000001
```

Last run against the pinned commit, on a checkpoint written by `lerobot-train`:

```text
1. config.json -> ACTConfig
   chunk_size=2 dim_model=32 latent_dim=8 n_heads=4
   input_features=['observation.state', 'observation.environment_state']
   output_features=['action']
2. model.safetensors -> load_state_dict(strict=True): <All keys matched successfully>
3. upstream forward pass on Rerobot's weights -> (2, 2, 2), all finite
4. optimizer_param_groups.json -> torch.optim.AdamW accepted it; lr=1e-05
   training_step.json = {'step': 1, 'num_processes': 1, 'batch_size': 2}

upstream lerobot can read this checkpoint.
```
