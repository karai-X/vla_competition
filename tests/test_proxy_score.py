import copy
import json

import numpy as np
import pytest

from parc2026.evaluation.proxy_score import (
    FIELDS, METRICS_VERSION, NAMES, InitialPoseCapturePolicy, aggregate_scores,
    bind_score_config, episode_score, load_config, motion_metrics, save_motion,
)


def config():
    return {"config_id": "test", "config_sha256": "frozen", "official_score": False,
            "metrics_version": METRICS_VERSION, "weights": dict.fromkeys(NAMES, .2),
            "reference_costs": dict.fromkeys(NAMES, 1.), "weight_sensitivity_bounds": [.1, .3],
            "normalization": "reference_over_reference_plus_cost", "max_environment_steps": 300}


def episode():
    return {"simulator_success": True, "collision": False, "steps": 100,
            "motion_metrics": {"metrics_version": METRICS_VERSION, "sampling": "initial_and_post_action",
                               "control_dt": .05, **dict.fromkeys(FIELDS, 1.), "steps": 100}}


def test_cubic_motion_has_analytic_jerk_and_known_path():
    t = np.arange(21) * .05
    p = np.column_stack((t**3, t * 0, t * 0))
    q = np.tile([0., 0., 0., 1.], (len(t), 1))
    m = motion_metrics(p, q, steps=20)
    assert m["rms_cartesian_jerk"] == pytest.approx(6.)
    assert m["cartesian_path_length"] == pytest.approx(1.)
    assert m["orientation_path_length"] == 0
    assert m["simulated_time_sec"] == 1.
    slow = motion_metrics(p, q, steps=20, dt=.1)
    assert slow["rms_cartesian_jerk"] == pytest.approx(6 / 8)


def test_quaternion_sign_and_scale_do_not_change_rotation_distance():
    angles = np.linspace(0, np.pi / 2, 21)
    q = np.column_stack((angles * 0, angles * 0, np.sin(angles / 2), np.cos(angles / 2)))
    q[::2] *= -3
    p = np.column_stack((angles, angles * 0, angles * 0))
    assert motion_metrics(p, q, steps=20)["orientation_path_length"] == pytest.approx(np.pi / 2)


def test_rest_and_too_short_sequences_cannot_fake_perfect_smoothness():
    q = np.tile([0., 0., 0., 1.], (101, 1))
    m = motion_metrics(np.zeros((101, 3)), q, steps=100)
    assert m["sparc"] is None
    r = episode(); r["motion_metrics"] = m
    assert episode_score(r, config())["score"] is None
    short = motion_metrics(np.zeros((2, 3)), q[:2], steps=1)
    assert short["rms_cartesian_jerk"] is None


def test_reference_value_maps_to_half_and_lower_cost_is_better():
    c = config(); c["reference_costs"]["time"] = 100
    r = episode()
    assert episode_score(r, c)["score"] == pytest.approx(.5)
    for name, field in zip(NAMES, FIELDS):
        better = copy.deepcopy(r)
        better["motion_metrics"][field] /= 2
        if field == "steps":
            better["steps"] = 50; better["motion_metrics"]["steps"] = 50
        assert episode_score(better, c)["score"] > .5


@pytest.mark.parametrize("change", [{"simulator_success": False}, {"collision": True}, {"steps": 301}])
def test_failure_collision_and_budget_gates_are_zero_without_motion(change):
    r = episode(); r.update(change); r.pop("motion_metrics")
    assert episode_score(r, config())["score"] == 0


def test_aggregate_denominator_includes_failures_and_reports_weight_sensitivity():
    c = config(); c["reference_costs"]["time"] = 100
    good = episode(); bad = dict(good, collision=True)
    result = aggregate_scores([good, bad], c, expected_episodes=2)
    assert result["score"] == pytest.approx(.25)
    sensitivity = result["weight_sensitivity"]
    assert sensitivity["lower"]["score"] <= result["score"] <= sensitivity["upper"]["score"]
    assert sum(sensitivity["lower"]["weights"].values()) == pytest.approx(1.)
    assert result["score"] <= .5  # collision-free success rate


def test_partial_and_old_unrecorded_rollouts_do_not_receive_fabricated_scores():
    assert aggregate_scores([episode()], config(), expected_episodes=100)["score"] is None
    old = episode(); old.pop("motion_metrics")
    report = aggregate_scores([old], config())
    assert report["score"] is None and report["unscorable_episode_indices"] == [0]
    old["simulator_success"] = "false"
    with pytest.raises(ValueError): episode_score(old, config())


def test_missing_frames_and_invalid_poses_are_rejected():
    with pytest.raises(ValueError): motion_metrics(np.zeros((5, 3)), np.ones((5, 4)), steps=5)
    with pytest.raises(ValueError): motion_metrics(np.zeros((5, 3)), np.zeros((5, 4)), steps=4)
    with pytest.raises(ValueError): motion_metrics(np.full((5, 3), np.nan), np.ones((5, 4)), steps=4)


def test_saved_trace_can_reproduce_metrics_without_simulator(tmp_path):
    p = np.column_stack((np.arange(8)**3 * 1e-4, np.zeros(8), np.zeros(8)))
    q = np.tile([0., 0., 0., 1.], (8, 1))
    saved = save_motion(tmp_path, "test:episode", p, q, steps=7)
    with np.load(tmp_path / saved["motion_archive"]["path"], allow_pickle=False) as f:
        actual = motion_metrics(f["positions"], f["quaternions_xyzw"], steps=int(f["steps"]))
    assert actual == saved["motion_metrics"]


def test_metric_capture_delegates_identical_actions_and_resets():
    class Policy:
        def reset(self, **kwargs): self.seed = kwargs["seed"]
        def get_action(self, obs): return obs["action"]
    delegate = Policy(); policy = InitialPoseCapturePolicy(delegate)
    obs = {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0., 0., 0., 1.]), "action": object()}
    policy.reset(seed=42, instruction="task")
    assert policy.get_action(obs) is obs["action"]
    assert policy.get_action(obs) is obs["action"]
    assert delegate.seed == 42 and len(policy.initial_poses) == 1


def test_score_config_validation_and_resume_freeze(tmp_path, monkeypatch):
    import parc2026.evaluation.proxy_score as module
    cfg = config(); cfg.pop("config_sha256")
    path = tmp_path / "config.json"; path.write_text(json.dumps(cfg))
    monkeypatch.setattr(module, "DEFAULT_CONFIG", path)
    assert load_config(path)["weights"] == cfg["weights"]
    output = tmp_path / "run"
    bind_score_config(output)
    bind_score_config(output)
    cfg["reference_costs"]["jerk"] = 2.; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): bind_score_config(output)
    cfg["weights"]["time"] = -.1; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): load_config(path)
