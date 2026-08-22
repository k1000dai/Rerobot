//! Behaviour parity tests for `lerobot.datasets.sampler` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1: `EpisodeAwareSampler` and
//! `compute_sampler_state`.
//!
//! Everything structural is a port: which frames are eligible, how
//! `drop_n_first_frames` / `drop_n_last_frames` shrink an episode, how a logical
//! position maps back to an absolute frame index, how the epoch advances, and
//! how a training step maps onto an (epoch, offset) pair.
//!
//! The *order within an epoch* is deliberately **not** a port. Upstream draws it
//! from `torch.randperm` seeded through `numpy.random.SeedSequence`; Rerobot
//! does not reproduce PyTorch's Mersenne stream, so it substitutes its own
//! documented permutation ([`rerobot_core::random::shuffled_permutation`]). The
//! tests below therefore pin *determinism and seed dependence* rather than
//! upstream's exact sequence, and `docs/compatibility.md` records the divergence.

use rerobot_core::dataset::sampler::{
    compute_sampler_state, EpisodeAwareSampler, SamplerError, SamplerState,
};

fn one_episode(frames: i64) -> EpisodeAwareSampler {
    EpisodeAwareSampler::new(&[0], &[frames], None, 0, 0, false, 0).unwrap()
}

// ---------------------------------------------------------------------------
// Construction and eligibility
// ---------------------------------------------------------------------------

#[test]
fn the_fixtures_single_episode_yields_its_four_frames_in_order() {
    let sampler = one_episode(4);
    assert_eq!(sampler.len(), 4);
    assert!(!sampler.is_empty());
    assert_eq!(sampler.indices(), vec![0, 1, 2, 3]);
}

#[test]
fn episodes_are_concatenated_by_their_dataset_index_ranges() {
    let sampler = EpisodeAwareSampler::new(&[0, 4, 9], &[4, 9, 11], None, 0, 0, false, 0).unwrap();
    // Lengths 4, 5 and 2: the ranges are half-open and contiguous here, so the
    // eligible frames are every dataset index from 0 to 10.
    assert_eq!(sampler.len(), 11);
    assert_eq!(sampler.indices(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
fn dropping_frames_shrinks_each_episode_at_both_ends() {
    let sampler = EpisodeAwareSampler::new(&[0, 10], &[6, 16], None, 1, 2, false, 0).unwrap();
    // Episode 0 keeps frames 1..4, episode 1 keeps 11..14.
    assert_eq!(sampler.indices(), vec![1, 2, 3, 11, 12, 13]);
}

#[test]
fn an_episode_filter_selects_by_episode_index_not_by_frame() {
    let sampler =
        EpisodeAwareSampler::new(&[0, 4, 9], &[4, 9, 11], Some(&[0, 2]), 0, 0, false, 0).unwrap();
    assert_eq!(sampler.indices(), vec![0, 1, 2, 3, 9, 10]);
}

#[test]
fn a_filtered_sampler_maps_absolute_episode_ranges_to_relative_dataset_rows() {
    let mut sampler =
        EpisodeAwareSampler::new(&[0, 4], &[4, 8], Some(&[1]), 0, 0, false, 0).unwrap();
    sampler
        .set_absolute_to_relative(vec![-1, -1, -1, -1, 0, 1, 2, 3])
        .unwrap();
    assert_eq!(sampler.indices(), vec![0, 1, 2, 3]);
}

#[test]
fn an_incomplete_filtered_sampler_mapping_is_refused_before_iteration() {
    let mut sampler =
        EpisodeAwareSampler::new(&[0, 4], &[4, 8], Some(&[1]), 0, 0, false, 0).unwrap();
    let error = sampler
        .set_absolute_to_relative(vec![-1, -1, -1, -1])
        .unwrap_err();
    assert_eq!(
        error,
        SamplerError::InvalidAbsoluteToRelativeMapping {
            absolute_index: 4,
            mapping_len: 4,
        }
    );
}

#[test]
fn an_episode_that_drops_to_nothing_is_skipped_and_recorded_rather_than_failing() {
    let sampler = EpisodeAwareSampler::new(&[0, 10], &[3, 20], None, 0, 5, false, 0).unwrap();
    // Episode 0 has 3 frames and loses 5, so it contributes nothing.
    assert_eq!(sampler.indices(), vec![10, 11, 12, 13, 14]);
    let skipped = sampler.skipped_episodes();
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0].episode_index, 0);
    assert_eq!(skipped[0].frames, 3);
    assert!(
        skipped[0].to_string().contains("Skipping"),
        "the warning does not mirror upstream's wording: {}",
        skipped[0]
    );
}

#[test]
fn every_episode_dropping_to_nothing_is_the_upstream_value_error() {
    let error = EpisodeAwareSampler::new(&[0], &[3], None, 0, 5, false, 0).unwrap_err();
    assert_eq!(error, SamplerError::NoValidFrames);
    assert!(
        error
            .to_string()
            .contains("No valid frames remain after applying"),
        "message drifted from upstream: {error}"
    );
}

#[test]
fn mismatched_boundary_lists_are_rejected_with_both_lengths() {
    let error = EpisodeAwareSampler::new(&[0, 4], &[4], None, 0, 0, false, 0).unwrap_err();
    assert_eq!(
        error,
        SamplerError::LengthMismatch {
            from_indices: 2,
            to_indices: 1
        }
    );
}

#[test]
fn negative_drop_counts_are_rejected_like_upstream() {
    assert_eq!(
        EpisodeAwareSampler::new(&[0], &[4], None, -1, 0, false, 0).unwrap_err(),
        SamplerError::NegativeDropNFirstFrames(-1)
    );
    assert_eq!(
        EpisodeAwareSampler::new(&[0], &[4], None, 0, -3, false, 0).unwrap_err(),
        SamplerError::NegativeDropNLastFrames(-3)
    );
}

#[test]
fn an_episode_filter_out_of_range_is_rejected_instead_of_indexing_out_of_bounds() {
    assert_eq!(
        EpisodeAwareSampler::new(&[0], &[4], Some(&[1]), 0, 0, false, 0).unwrap_err(),
        SamplerError::EpisodeIndexOutOfRange {
            episode_index: 1,
            total_episodes: 1
        }
    );
    assert_eq!(
        EpisodeAwareSampler::new(&[0], &[4], Some(&[-1]), 0, 0, false, 0).unwrap_err(),
        SamplerError::EpisodeIndexOutOfRange {
            episode_index: -1,
            total_episodes: 1
        }
    );
}

// ---------------------------------------------------------------------------
// Position -> frame index
// ---------------------------------------------------------------------------

#[test]
fn a_logical_position_maps_onto_the_right_episode_by_searchsorted() {
    let sampler =
        EpisodeAwareSampler::new(&[0, 10, 20], &[3, 15, 21], None, 0, 0, false, 0).unwrap();
    // Lengths 3, 5, 1 -> cumulative [3, 8, 9].
    let expected = [0, 1, 2, 10, 11, 12, 13, 14, 20];
    for (position, frame) in expected.iter().enumerate() {
        assert_eq!(
            sampler.frame_index(position),
            Some(*frame),
            "position {position}"
        );
    }
    assert_eq!(sampler.frame_index(9), None);
}

// ---------------------------------------------------------------------------
// Epochs, determinism and resume
// ---------------------------------------------------------------------------

#[test]
fn an_unshuffled_epoch_is_the_frame_order_itself() {
    let mut sampler = one_episode(4);
    assert_eq!(sampler.next_epoch(), vec![0, 1, 2, 3]);
    // Order is epoch-independent when not shuffling.
    assert_eq!(sampler.next_epoch(), vec![0, 1, 2, 3]);
}

#[test]
fn a_shuffled_epoch_is_a_permutation_of_the_same_frames() {
    let mut sampler = EpisodeAwareSampler::new(&[0], &[4], None, 0, 0, true, 1000).unwrap();
    let epoch = sampler.next_epoch();
    let mut sorted = epoch.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3]);
}

#[test]
fn the_same_seed_reproduces_the_same_epochs_and_a_different_seed_does_not() {
    let mut left = EpisodeAwareSampler::new(&[0], &[64], None, 0, 0, true, 7).unwrap();
    let mut right = EpisodeAwareSampler::new(&[0], &[64], None, 0, 0, true, 7).unwrap();
    let mut other = EpisodeAwareSampler::new(&[0], &[64], None, 0, 0, true, 8).unwrap();
    assert_eq!(left.next_epoch(), right.next_epoch());
    assert_eq!(left.next_epoch(), right.next_epoch());

    let mut same_seed = EpisodeAwareSampler::new(&[0], &[64], None, 0, 0, true, 7).unwrap();
    assert_ne!(same_seed.next_epoch(), other.next_epoch());
}

#[test]
fn consecutive_epochs_differ_because_the_permutation_is_a_function_of_seed_and_epoch() {
    let mut sampler = EpisodeAwareSampler::new(&[0], &[64], None, 0, 0, true, 1000).unwrap();
    let first = sampler.next_epoch();
    let second = sampler.next_epoch();
    assert_ne!(first, second);

    // ... and re-selecting epoch 0 reproduces the first one exactly.
    sampler.set_epoch(0);
    assert_eq!(sampler.next_epoch(), first);
}

#[test]
fn iterating_advances_the_epoch_eagerly_and_clears_the_resume_offset() {
    let mut sampler = EpisodeAwareSampler::new(&[0], &[8], None, 0, 0, true, 3).unwrap();
    sampler.load_state(SamplerState {
        epoch: 5,
        start_index: 3,
    });
    assert_eq!(
        sampler.state(),
        SamplerState {
            epoch: 5,
            start_index: 3
        }
    );
    let resumed = sampler.next_epoch();
    assert_eq!(resumed.len(), 5, "the first three positions are skipped");
    assert_eq!(
        sampler.state(),
        SamplerState {
            epoch: 6,
            start_index: 0
        }
    );
    // `__len__` still reports the full epoch during a resumed one.
    assert_eq!(sampler.len(), 8);
}

#[test]
fn a_resumed_epoch_is_the_tail_of_the_full_epoch() {
    let mut full = EpisodeAwareSampler::new(&[0], &[8], None, 0, 0, true, 3).unwrap();
    full.set_epoch(5);
    let whole = full.next_epoch();

    let mut resumed = EpisodeAwareSampler::new(&[0], &[8], None, 0, 0, true, 3).unwrap();
    resumed.load_state(SamplerState {
        epoch: 5,
        start_index: 3,
    });
    assert_eq!(resumed.next_epoch(), whole[3..].to_vec());
}

// ---------------------------------------------------------------------------
// `compute_sampler_state`
// ---------------------------------------------------------------------------

#[test]
fn a_step_maps_onto_an_epoch_and_an_offset() {
    // 4 frames, batch 2, one process -> 2 batches per epoch.
    assert_eq!(
        compute_sampler_state(0, 4, 2, 1),
        SamplerState {
            epoch: 0,
            start_index: 0
        }
    );
    assert_eq!(
        compute_sampler_state(1, 4, 2, 1),
        SamplerState {
            epoch: 0,
            start_index: 2
        }
    );
    assert_eq!(
        compute_sampler_state(2, 4, 2, 1),
        SamplerState {
            epoch: 1,
            start_index: 0
        }
    );
    assert_eq!(
        compute_sampler_state(5, 4, 2, 1),
        SamplerState {
            epoch: 2,
            start_index: 2
        }
    );
}

#[test]
fn the_batch_count_per_epoch_is_a_double_ceiling_over_batch_size_and_world_size() {
    // ceil(ceil(10 / 3) / 2) == ceil(4 / 2) == 2 batches per rank per epoch.
    assert_eq!(
        compute_sampler_state(3, 10, 3, 2),
        SamplerState {
            epoch: 1,
            start_index: 6
        }
    );
}

#[test]
fn the_start_index_is_capped_at_the_frame_count() {
    // The `min` upstream calls defensive: a large batch cannot point past the end.
    assert_eq!(
        compute_sampler_state(0, 4, 16, 1),
        SamplerState {
            epoch: 0,
            start_index: 0
        }
    );
    assert_eq!(
        compute_sampler_state(1, 4, 16, 1),
        SamplerState {
            epoch: 1,
            start_index: 0
        }
    );
}
