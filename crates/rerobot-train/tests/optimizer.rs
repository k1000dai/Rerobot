//! Behaviour tests for `torch.optim.AdamW` and `torch.nn.utils.clip_grad_norm_`,
//! derived from `torch/optim/adamw.py` and `torch/nn/utils/clip_grad.py` and from
//! how `lerobot/scripts/lerobot_train.py::update_policy` calls them.
//!
//! These are checked against hand-computed closed forms rather than against a
//! reference run, because the update rule is short enough to write out and that is
//! what makes the check independent of the implementation.

use candle_core::{DType, Device, Tensor, Var};
use rerobot_core::policy::act::AdamWConfig;
use rerobot_train::model::params::NamedParameter;
use rerobot_train::optim::{
    act_parameter_groups, clip_grad_norm, AdamW, GroupSettings, ParameterGroup,
};

fn parameter(name: &str, values: &[f32]) -> NamedParameter {
    let tensor = Tensor::from_vec(values.to_vec(), values.len(), &Device::Cpu).unwrap();
    NamedParameter {
        name: name.to_owned(),
        value: Var::from_tensor(&tensor).unwrap(),
    }
}

fn settings(lr: f64, weight_decay: f64) -> GroupSettings {
    GroupSettings {
        lr,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay,
    }
}

/// Run one step against an explicit gradient, by building a graph whose gradient
/// with respect to the parameter is exactly `gradient`.
fn step_with_gradient(parameters: &[NamedParameter], optimizer: &mut AdamW, gradients: &[&[f32]]) {
    let mut loss = Tensor::new(0f32, &Device::Cpu).unwrap();
    for (parameter, gradient) in parameters.iter().zip(gradients) {
        let coefficients =
            Tensor::from_vec(gradient.to_vec(), gradient.len(), &Device::Cpu).unwrap();
        // d/dp of sum(p * g) is g.
        let term = (parameter.value.as_tensor() * &coefficients)
            .unwrap()
            .sum_all()
            .unwrap();
        loss = (&loss + &term).unwrap();
    }
    let store = loss.backward().unwrap();
    optimizer.step(parameters, &store).unwrap();
}

fn read(parameter: &NamedParameter) -> Vec<f32> {
    parameter
        .value
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

// ---------------------------------------------------------------------------
// The update rule
// ---------------------------------------------------------------------------

#[test]
fn the_first_step_moves_by_the_learning_rate_times_the_sign_of_the_gradient() {
    // step 1: m = 0.1 g, v = 0.001 g^2, m̂ = g, v̂ = g^2, so the update is
    // lr * g / (|g| + eps) = lr * sign(g), to within the epsilon.
    let parameters = vec![parameter("p", &[1.0, 1.0, 1.0])];
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.0),
        }],
        1,
    )
    .unwrap();
    step_with_gradient(&parameters, &mut optimizer, &[&[2.0, -3.0, 0.5]]);
    let values = read(&parameters[0]);
    assert!((values[0] - 0.9).abs() < 1e-6, "got {values:?}");
    assert!((values[1] - 1.1).abs() < 1e-6, "got {values:?}");
    assert!((values[2] - 0.9).abs() < 1e-6, "got {values:?}");
}

#[test]
fn weight_decay_is_decoupled_and_applied_before_the_moment_update() {
    // torch: `p.mul_(1 - lr * weight_decay)` first, then the Adam step. With a zero
    // gradient the Adam step contributes nothing, so the decay is observable alone.
    let parameters = vec![parameter("p", &[2.0])];
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.5),
        }],
        1,
    )
    .unwrap();
    step_with_gradient(&parameters, &mut optimizer, &[&[0.0]]);
    // 2 * (1 - 0.1 * 0.5) = 1.9, and the Adam term is 0 / (0 + eps) = 0.
    assert!((read(&parameters[0])[0] - 1.9).abs() < 1e-6);
}

#[test]
fn a_zero_weight_decay_leaves_the_parameter_scale_alone() {
    let parameters = vec![parameter("p", &[2.0])];
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.0),
        }],
        1,
    )
    .unwrap();
    step_with_gradient(&parameters, &mut optimizer, &[&[0.0]]);
    assert_eq!(read(&parameters[0])[0], 2.0);
}

#[test]
fn the_bias_corrections_track_the_step_count() {
    // Two steps with the same gradient. The second update is
    // lr * m̂ / (sqrt(v̂) + eps) with m̂ = m / (1 - 0.9^2) and v̂ = v / (1 - 0.999^2);
    // for a constant gradient that is again very nearly lr * sign(g), so a constant
    // gradient must produce two nearly equal steps.
    let parameters = vec![parameter("p", &[0.0])];
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.0),
        }],
        1,
    )
    .unwrap();
    step_with_gradient(&parameters, &mut optimizer, &[&[1.0]]);
    let after_first = read(&parameters[0])[0];
    step_with_gradient(&parameters, &mut optimizer, &[&[1.0]]);
    let after_second = read(&parameters[0])[0];
    assert!(
        (after_first + 0.1).abs() < 1e-6,
        "first step: {after_first}"
    );
    let second_delta = after_second - after_first;
    assert!(
        (second_delta + 0.1).abs() < 1e-3,
        "second step moved by {second_delta}, not about -0.1"
    );
}

#[test]
fn each_group_uses_its_own_learning_rate() {
    let parameters = vec![parameter("fast", &[0.0]), parameter("slow", &[0.0])];
    let mut optimizer = AdamW::new(
        vec![
            ParameterGroup {
                params: vec![0],
                settings: settings(1.0, 0.0),
            },
            ParameterGroup {
                params: vec![1],
                settings: settings(0.01, 0.0),
            },
        ],
        2,
    )
    .unwrap();
    step_with_gradient(&parameters, &mut optimizer, &[&[1.0], &[1.0]]);
    assert!((read(&parameters[0])[0] + 1.0).abs() < 1e-5);
    assert!((read(&parameters[1])[0] + 0.01).abs() < 1e-5);
}

#[test]
fn a_parameter_in_two_groups_is_refused_and_so_is_one_in_none() {
    let duplicate = AdamW::new(
        vec![
            ParameterGroup {
                params: vec![0],
                settings: settings(0.1, 0.0),
            },
            ParameterGroup {
                params: vec![0],
                settings: settings(0.1, 0.0),
            },
        ],
        1,
    )
    .unwrap_err();
    assert!(
        duplicate.to_string().contains("more than one group"),
        "unexpected: {duplicate}"
    );

    let uncovered = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.0),
        }],
        2,
    )
    .unwrap_err();
    assert!(
        uncovered.to_string().contains("never be updated"),
        "unexpected: {uncovered}"
    );

    let out_of_range = AdamW::new(
        vec![ParameterGroup {
            params: vec![5],
            settings: settings(0.1, 0.0),
        }],
        1,
    )
    .unwrap_err();
    assert!(
        out_of_range.to_string().contains("1 parameters"),
        "unexpected: {out_of_range}"
    );
}

#[test]
fn the_act_preset_becomes_a_main_group_and_a_backbone_group() {
    let preset = AdamWConfig {
        lr: 1e-5,
        weight_decay: 1e-4,
        grad_clip_norm: 10.0,
        betas: [0.9, 0.999],
        eps: 1e-8,
    };
    let groups = act_parameter_groups([vec![0, 1], vec![2]], &preset, 3e-4);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].params, vec![0, 1]);
    assert_eq!(groups[0].settings.lr, 1e-5);
    assert_eq!(groups[1].params, vec![2]);
    assert_eq!(
        groups[1].settings.lr, 3e-4,
        "the backbone group carries optimizer_lr_backbone"
    );
    // Everything except the learning rate is shared.
    assert_eq!(
        groups[0].settings.weight_decay,
        groups[1].settings.weight_decay
    );
    assert_eq!(groups[0].settings.betas, groups[1].settings.betas);
    assert_eq!(groups[0].settings.eps, groups[1].settings.eps);
}

// ---------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------

#[test]
fn the_returned_norm_is_the_total_before_clipping() {
    // Two parameters with gradients 3 and 4: the 2-norm of the concatenation is 5.
    let parameters = vec![parameter("a", &[0.0]), parameter("b", &[0.0])];
    let mut loss = Tensor::new(0f32, &Device::Cpu).unwrap();
    for (parameter, gradient) in parameters.iter().zip([3.0f32, 4.0]) {
        let coefficient = Tensor::from_vec(vec![gradient], 1, &Device::Cpu).unwrap();
        loss = (&loss
            + &(parameter.value.as_tensor() * &coefficient)
                .unwrap()
                .sum_all()
                .unwrap())
            .unwrap();
    }
    let mut store = loss.backward().unwrap();
    let norm = clip_grad_norm(&parameters, &mut store, 1.0).unwrap();
    assert!((norm - 5.0).abs() < 1e-6, "got {norm}");
}

#[test]
fn clipping_scales_by_max_norm_over_the_total_plus_the_torch_epsilon() {
    let parameters = vec![parameter("a", &[0.0])];
    let coefficient = Tensor::from_vec(vec![10.0f32], 1, &Device::Cpu).unwrap();
    let loss = (parameters[0].value.as_tensor() * &coefficient)
        .unwrap()
        .sum_all()
        .unwrap();
    let mut store = loss.backward().unwrap();
    let norm = clip_grad_norm(&parameters, &mut store, 2.0).unwrap();
    assert!((norm - 10.0).abs() < 1e-6);
    // torch: coefficient = max_norm / (total_norm + 1e-6).
    let scaled = store
        .get(parameters[0].value.as_tensor())
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let expected = 10.0 * (2.0 / (10.0 + 1e-6));
    assert!((scaled[0] - expected as f32).abs() < 1e-4, "got {scaled:?}");
}

#[test]
fn a_gradient_already_inside_the_ball_is_left_alone_rather_than_scaled_up() {
    // torch clamps the coefficient at 1, so a small gradient is not amplified.
    let parameters = vec![parameter("a", &[0.0])];
    let coefficient = Tensor::from_vec(vec![0.5f32], 1, &Device::Cpu).unwrap();
    let loss = (parameters[0].value.as_tensor() * &coefficient)
        .unwrap()
        .sum_all()
        .unwrap();
    let mut store = loss.backward().unwrap();
    clip_grad_norm(&parameters, &mut store, 10.0).unwrap();
    let after = store
        .get(parameters[0].value.as_tensor())
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(after, vec![0.5]);
}

#[test]
fn an_infinite_max_norm_measures_without_clipping() {
    // This is `update_policy`'s `grad_clip_norm <= 0` branch: torch calls
    // `clip_grad_norm_(..., float("inf"))` purely to report the norm.
    let parameters = vec![parameter("a", &[0.0])];
    let coefficient = Tensor::from_vec(vec![7.0f32], 1, &Device::Cpu).unwrap();
    let loss = (parameters[0].value.as_tensor() * &coefficient)
        .unwrap()
        .sum_all()
        .unwrap();
    let mut store = loss.backward().unwrap();
    let norm = clip_grad_norm(&parameters, &mut store, f64::INFINITY).unwrap();
    assert!((norm - 7.0).abs() < 1e-6);
    assert_eq!(
        store
            .get(parameters[0].value.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        vec![7.0]
    );
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[test]
fn the_stored_moments_do_not_keep_the_graph_they_were_computed_from() {
    // A gradient candle returns still points at the graph that produced it, so a
    // moment built from one keeps that step's whole forward pass reachable -- and
    // because step N + 1's moment is built from step N's, undetached moments chain
    // every step of a run together. On a GPU that is a few hundred megabytes per
    // step and an out-of-memory failure within a couple of dozen of them; the
    // numbers stay correct right up until the allocation fails, so nothing else in
    // this file would notice.
    let parameters = vec![parameter("p", &[1.0, 2.0]), parameter("q", &[3.0])];
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0, 1],
            settings: settings(0.1, 0.01),
        }],
        2,
    )
    .unwrap();

    // `step_with_gradient`'s loss is linear in the parameter, so its gradient is the
    // coefficient tensor and carries no graph at all -- there is nothing for a
    // moment to retain, and the assertion below would hold with or without the fix.
    // A model's gradients are not like that: they are computed *through* the
    // activations, so they point back at them. `sum(p * p * c)` is the smallest
    // thing with that property, its gradient `2 * p * c` being a tensor built from
    // the parameter itself.
    let quadratic_step = |optimizer: &mut AdamW| {
        let mut loss = Tensor::new(0f32, &Device::Cpu).unwrap();
        for parameter in &parameters {
            let value = parameter.value.as_tensor();
            let term = (value * value).unwrap().sum_all().unwrap();
            loss = (&loss + &term).unwrap();
        }
        let store = loss.backward().unwrap();
        assert!(
            store
                .get(parameters[0].value.as_tensor())
                .expect("the parameter has a gradient")
                .track_op(),
            "the test's own gradient carries no graph, so it cannot detect a retained one"
        );
        optimizer.step(&parameters, &store).unwrap();
    };

    for _ in 0..3 {
        quadratic_step(&mut optimizer);
        assert!(
            !rerobot_train::optim::any_moment_tracks_its_graph(&optimizer),
            "a stored AdamW moment is still attached to its backward graph"
        );
    }
    // And the state really was populated, so the assertion above was not vacuous.
    assert_eq!(rerobot_train::optim::moment_summary(&optimizer).len(), 2);
}

#[test]
fn the_state_round_trips_through_its_safetensors_form() {
    let parameters = vec![parameter("p", &[1.0, 2.0])];
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.0),
        }],
        1,
    )
    .unwrap();
    step_with_gradient(&parameters, &mut optimizer, &[&[1.0, -1.0]]);
    let saved = optimizer.state_tensors(&Device::Cpu).unwrap();
    assert!(saved.contains_key("state/0/step"));
    assert!(saved.contains_key("state/0/exp_avg"));
    assert!(saved.contains_key("state/0/exp_avg_sq"));

    let mut restored = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.0),
        }],
        1,
    )
    .unwrap();
    restored.load_state_tensors(&parameters, &saved).unwrap();
    let reread = restored.state_tensors(&Device::Cpu).unwrap();
    for (key, tensor) in &saved {
        let other = reread.get(key).expect("the key survives the round trip");
        let difference = (tensor - other)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(difference, 0.0, "{key} did not round trip");
    }
}

#[test]
fn a_state_file_missing_a_moment_is_refused() {
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.0),
        }],
        1,
    )
    .unwrap();
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "state/0/step".to_owned(),
        Tensor::new(1f32, &Device::Cpu).unwrap(),
    );
    let parameters = vec![parameter("p", &[0.0])];
    let error = optimizer
        .load_state_tensors(&parameters, &tensors)
        .unwrap_err();
    assert!(
        error.to_string().contains("state/0/exp_avg"),
        "unexpected: {error}"
    );
}

#[test]
fn a_parameter_with_no_gradient_is_skipped_without_advancing_its_moments() {
    // torch's `if p.grad is None: continue`. Skipping means the moment state must
    // not advance either, or a sparsely-updated parameter would get bias
    // corrections computed for steps it never took.
    let parameters = vec![parameter("touched", &[0.0]), parameter("untouched", &[3.0])];
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0, 1],
            settings: settings(0.1, 0.0),
        }],
        2,
    )
    .unwrap();
    // Only the first parameter appears in the graph.
    let coefficient = Tensor::from_vec(vec![1.0f32], 1, &Device::Cpu).unwrap();
    let loss = (parameters[0].value.as_tensor() * &coefficient)
        .unwrap()
        .sum_all()
        .unwrap();
    let store = loss.backward().unwrap();
    optimizer.step(&parameters, &store).unwrap();

    assert!((read(&parameters[0])[0] + 0.1).abs() < 1e-5);
    assert_eq!(
        read(&parameters[1])[0],
        3.0,
        "a parameter with no gradient must not be touched at all"
    );
    let saved = optimizer.state_tensors(&Device::Cpu).unwrap();
    assert!(saved.contains_key("state/0/step"));
    assert!(
        !saved.contains_key("state/1/step"),
        "the skipped parameter must have no state at all"
    );
}

#[test]
fn a_freshly_built_optimizer_has_no_state_and_reports_the_first_groups_rate() {
    let optimizer = AdamW::new(
        vec![
            ParameterGroup {
                params: vec![0],
                settings: settings(1e-5, 0.0),
            },
            ParameterGroup {
                params: Vec::new(),
                settings: settings(3e-4, 0.0),
            },
        ],
        1,
    )
    .unwrap();
    assert_eq!(optimizer.first_lr(), 1e-5);
    assert!(optimizer.state_tensors(&Device::Cpu).unwrap().is_empty());
    assert_eq!(optimizer.groups().len(), 2);
}

#[test]
fn the_step_counter_is_written_as_a_zero_dimensional_tensor_like_torch() {
    let parameters = vec![parameter("p", &[0.0])];
    let mut optimizer = AdamW::new(
        vec![ParameterGroup {
            params: vec![0],
            settings: settings(0.1, 0.0),
        }],
        1,
    )
    .unwrap();
    step_with_gradient(&parameters, &mut optimizer, &[&[1.0]]);
    let saved = optimizer.state_tensors(&Device::Cpu).unwrap();
    let step = &saved["state/0/step"];
    assert_eq!(step.dims().len(), 0, "torch stores step as a 0-d tensor");
    assert_eq!(step.dtype(), DType::F32);
    assert_eq!(step.to_scalar::<f32>().unwrap(), 1.0);
}
