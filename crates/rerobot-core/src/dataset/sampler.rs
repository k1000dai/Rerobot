//! Port of `lerobot.datasets.sampler`: `EpisodeAwareSampler` and
//! `compute_sampler_state`.
//!
//! Everything structural is ported: the per-episode boundary representation
//! (`O(num_episodes)` memory, not a materialized frame list), the
//! `drop_n_first_frames` / `drop_n_last_frames` window, the episode filter, the
//! searchsorted position → frame mapping, the eager epoch advance in `__iter__`,
//! `state_dict` / `load_state_dict` resume, and the step → (epoch, offset)
//! arithmetic.
//!
//! **The order within a shuffled epoch is not ported.** Upstream derives it from
//! `torch.randperm` seeded through `numpy.random.SeedSequence([seed, epoch])`,
//! which would require porting NumPy's seed-sequence hash *and* PyTorch's
//! Mersenne Twister together with its permutation algorithm. Rerobot substitutes
//! [`crate::random::shuffled_permutation`], which keeps every property the
//! training loop depends on — a pure function of `(seed, epoch)`, reproducible
//! across processes and platforms, and resumable from an offset — but produces a
//! different sequence. `docs/compatibility.md` records this.

use crate::random::{mix64, GAMMA};
use std::fmt;

/// An episode that contributed no frames, mirroring upstream's `logger.warning`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkippedEpisode {
    /// Index of the episode in the boundary lists.
    pub episode_index: usize,
    /// `dataset_to_index - dataset_from_index` for that episode.
    pub frames: i64,
    /// `drop_n_first_frames` in force.
    pub drop_n_first_frames: i64,
    /// `drop_n_last_frames` in force.
    pub drop_n_last_frames: i64,
}

impl fmt::Display for SkippedEpisode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Episode {} has {} frames but drop_n_first_frames={} and \
             drop_n_last_frames={} removes all frames. Skipping.",
            self.episode_index, self.frames, self.drop_n_first_frames, self.drop_n_last_frames
        )
    }
}

/// Why an `EpisodeAwareSampler` could not be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplerError {
    /// The two boundary lists had different lengths.
    LengthMismatch {
        /// `len(dataset_from_indices)`.
        from_indices: usize,
        /// `len(dataset_to_indices)`.
        to_indices: usize,
    },
    /// `drop_n_first_frames` was negative.
    NegativeDropNFirstFrames(i64),
    /// `drop_n_last_frames` was negative.
    NegativeDropNLastFrames(i64),
    /// An entry of `episode_indices_to_use` was not a valid episode.
    ///
    /// Upstream indexes a NumPy boolean mask with the list, so a negative index
    /// wraps and an out-of-range one raises `IndexError`. Both are refused here.
    EpisodeIndexOutOfRange {
        /// The offending value.
        episode_index: i64,
        /// How many episodes the boundary lists describe.
        total_episodes: usize,
    },
    /// The absolute-to-relative map supplied for an episode-filtered dataset was incomplete.
    InvalidAbsoluteToRelativeMapping {
        /// Absolute frame index needing a mapping entry.
        absolute_index: i64,
        /// Number of entries in the supplied mapping.
        mapping_len: usize,
    },
    /// Every selected episode was empty after the drops.
    NoValidFrames,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                from_indices,
                to_indices,
            } => write!(
                formatter,
                "dataset_from_indices and dataset_to_indices must have the same length, \
                 got {from_indices} and {to_indices}"
            ),
            Self::NegativeDropNFirstFrames(value) => {
                write!(formatter, "drop_n_first_frames must be >= 0, got {value}")
            }
            Self::NegativeDropNLastFrames(value) => {
                write!(formatter, "drop_n_last_frames must be >= 0, got {value}")
            }
            Self::EpisodeIndexOutOfRange {
                episode_index,
                total_episodes,
            } => write!(
                formatter,
                "episode index {episode_index} is out of range for {total_episodes} episodes"
            ),
            Self::InvalidAbsoluteToRelativeMapping {
                absolute_index,
                mapping_len,
            } => write!(
                formatter,
                "absolute frame index {absolute_index} has no relative row in a mapping of length {mapping_len}"
            ),
            Self::NoValidFrames => formatter.write_str(
                "No valid frames remain after applying drop_n_first_frames and \
                 drop_n_last_frames. All episodes were either filtered out or had too few frames.",
            ),
        }
    }
}

impl std::error::Error for SamplerError {}

/// `EpisodeAwareSampler.state_dict()` / `load_state_dict()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerState {
    /// Which epoch's permutation to use.
    pub epoch: u64,
    /// How many positions of that epoch were already consumed.
    pub start_index: usize,
}

/// Sampler over episode frames that stores only per-episode boundaries.
#[derive(Debug, Clone)]
pub struct EpisodeAwareSampler {
    starts: Vec<i64>,
    cumulative_lengths: Vec<usize>,
    num_frames: usize,
    shuffle: bool,
    seed: u64,
    epoch: u64,
    start_index: usize,
    skipped: Vec<SkippedEpisode>,
    /// Optional absolute-to-relative mapping used by an episode-filtered dataset.
    absolute_to_relative: Option<Vec<i64>>,
}

impl EpisodeAwareSampler {
    /// Upstream's `__init__`, with `episode_indices_to_use = None` meaning "all".
    pub fn new(
        dataset_from_indices: &[i64],
        dataset_to_indices: &[i64],
        episode_indices_to_use: Option<&[i64]>,
        drop_n_first_frames: i64,
        drop_n_last_frames: i64,
        shuffle: bool,
        seed: u64,
    ) -> Result<Self, SamplerError> {
        if drop_n_first_frames < 0 {
            return Err(SamplerError::NegativeDropNFirstFrames(drop_n_first_frames));
        }
        if drop_n_last_frames < 0 {
            return Err(SamplerError::NegativeDropNLastFrames(drop_n_last_frames));
        }
        if dataset_from_indices.len() != dataset_to_indices.len() {
            return Err(SamplerError::LengthMismatch {
                from_indices: dataset_from_indices.len(),
                to_indices: dataset_to_indices.len(),
            });
        }

        let total_episodes = dataset_from_indices.len();
        let mut used = vec![episode_indices_to_use.is_none(); total_episodes];
        if let Some(selection) = episode_indices_to_use {
            for episode_index in selection {
                let in_range = usize::try_from(*episode_index)
                    .ok()
                    .filter(|index| *index < total_episodes);
                match in_range {
                    Some(index) => used[index] = true,
                    None => {
                        return Err(SamplerError::EpisodeIndexOutOfRange {
                            episode_index: *episode_index,
                            total_episodes,
                        })
                    }
                }
            }
        }

        let mut starts = Vec::new();
        let mut cumulative_lengths = Vec::new();
        let mut skipped = Vec::new();
        let mut running = 0usize;
        for episode_index in 0..total_episodes {
            let start = dataset_from_indices[episode_index].saturating_add(drop_n_first_frames);
            let length = dataset_to_indices[episode_index]
                .saturating_sub(drop_n_last_frames)
                .saturating_sub(start);
            if used[episode_index] && length <= 0 {
                skipped.push(SkippedEpisode {
                    episode_index,
                    frames: dataset_to_indices[episode_index]
                        .saturating_sub(dataset_from_indices[episode_index]),
                    drop_n_first_frames,
                    drop_n_last_frames,
                });
            }
            if !used[episode_index] || length <= 0 {
                continue;
            }
            starts.push(start);
            running += length as usize;
            cumulative_lengths.push(running);
        }

        if cumulative_lengths.is_empty() {
            return Err(SamplerError::NoValidFrames);
        }

        Ok(Self {
            starts,
            cumulative_lengths,
            num_frames: running,
            shuffle,
            seed,
            epoch: 0,
            start_index: 0,
            skipped,
            absolute_to_relative: None,
        })
    }

    /// `__len__`: the full epoch length, even during a resumed epoch.
    pub fn len(&self) -> usize {
        self.num_frames
    }

    /// Whether the sampler yields nothing. Never true for a constructed
    /// sampler — [`SamplerError::NoValidFrames`] is returned instead — but
    /// required by convention alongside [`Self::len`].
    pub fn is_empty(&self) -> bool {
        self.num_frames == 0
    }

    /// Upstream's `indices` property: every eligible frame in unshuffled order.
    pub fn indices(&self) -> Vec<i64> {
        (0..self.num_frames)
            .map(|position| {
                self.frame_index(position)
                    .expect("positions below len always map")
            })
            .collect()
    }

    /// `_frame_index`: the absolute frame index at a logical position.
    pub fn frame_index(&self, position: usize) -> Option<i64> {
        if position >= self.num_frames {
            return None;
        }
        // `np.searchsorted(cum_lengths, position, side="right")`.
        let episode = self
            .cumulative_lengths
            .partition_point(|end| *end <= position);
        let consumed = if episode == 0 {
            0
        } else {
            self.cumulative_lengths[episode - 1]
        };
        let absolute = self.starts[episode].saturating_add((position - consumed) as i64);
        Some(
            self.absolute_to_relative
                .as_ref()
                .map_or(absolute, |mapping| {
                    mapping
                        [usize::try_from(absolute).expect("absolute frame index is non-negative")]
                }),
        )
    }

    /// `set_epoch`.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    /// `state_dict`.
    pub fn state(&self) -> SamplerState {
        SamplerState {
            epoch: self.epoch,
            start_index: self.start_index,
        }
    }

    /// `load_state_dict`.
    pub fn load_state(&mut self, state: SamplerState) {
        self.epoch = state.epoch;
        self.start_index = state.start_index;
    }

    /// Episodes that contributed nothing, in episode order.
    pub fn skipped_episodes(&self) -> &[SkippedEpisode] {
        &self.skipped
    }

    /// Convert sampled absolute frame indices into the relative rows of a filtered dataset.
    ///
    /// The mapping is indexed by absolute frame number and is validated before use so a
    /// sampler can never silently return a row from a different episode.
    pub fn set_absolute_to_relative(&mut self, mapping: Vec<i64>) -> Result<(), SamplerError> {
        for absolute in self.indices() {
            let row = usize::try_from(absolute).map_err(|_| {
                SamplerError::InvalidAbsoluteToRelativeMapping {
                    absolute_index: absolute,
                    mapping_len: mapping.len(),
                }
            })?;
            if row >= mapping.len()
                || mapping[row] < 0
                || usize::try_from(mapping[row])
                    .ok()
                    .is_none_or(|relative| relative >= self.num_frames)
            {
                return Err(SamplerError::InvalidAbsoluteToRelativeMapping {
                    absolute_index: absolute,
                    mapping_len: mapping.len(),
                });
            }
        }
        self.absolute_to_relative = Some(mapping);
        Ok(())
    }

    /// The seed the permutation is derived from, together with the epoch.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The seed of one epoch's permutation: the `epoch + 1`-th output of
    /// `SplitMix64::new(seed)`, computed in constant time.
    pub fn epoch_seed(&self, epoch: u64) -> u64 {
        mix64(
            self.seed
                .wrapping_add(GAMMA.wrapping_mul(epoch.wrapping_add(1))),
        )
    }

    /// `__iter__`: the frame indices of the current epoch, advancing the epoch
    /// eagerly and clearing the resume offset, exactly as upstream does.
    pub fn next_epoch(&mut self) -> Vec<i64> {
        let epoch = self.epoch;
        let start = self.start_index.min(self.num_frames);
        self.epoch = self.epoch.wrapping_add(1);
        self.start_index = 0;

        if self.shuffle {
            let order =
                crate::random::shuffled_permutation(self.num_frames, self.epoch_seed(epoch));
            order[start..]
                .iter()
                .map(|position| {
                    self.frame_index(*position)
                        .expect("a permutation only holds valid positions")
                })
                .collect()
        } else {
            (start..self.num_frames)
                .map(|position| {
                    self.frame_index(position)
                        .expect("positions below len always map")
                })
                .collect()
        }
    }
}

/// `compute_sampler_state`: which (epoch, offset) an optimization step resumes at.
///
/// # Panics
///
/// If `batch_size` or `num_processes` is zero; neither has a meaning here and
/// upstream would divide by zero.
pub fn compute_sampler_state(
    step: u64,
    num_frames: usize,
    batch_size: usize,
    num_processes: usize,
) -> SamplerState {
    assert!(batch_size > 0, "batch_size must be positive");
    assert!(num_processes > 0, "num_processes must be positive");
    let batches = num_frames.div_ceil(batch_size);
    let batches_per_epoch = batches.div_ceil(num_processes).max(1) as u64;
    let epoch = step / batches_per_epoch;
    let batches_into_epoch = step % batches_per_epoch;
    let start_index = (batches_into_epoch as usize)
        .saturating_mul(batch_size)
        .saturating_mul(num_processes)
        .min(num_frames);
    SamplerState { epoch, start_index }
}
