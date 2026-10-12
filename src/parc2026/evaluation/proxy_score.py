"""Gated five-metric motion-quality score with explicit weight assumptions.

A trajectory scores only if it succeeds without displacing non-target objects
and within the step budget; then steps, jerk, SPARC, path length and rotation
are normalised as ref / (ref + cost) and weighted. The weights are a choice,
so every comparison is also reported across a weight-sensitivity range.
This module only observes and scores completed trajectories.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .sparc import compute_sparc

METRICS_VERSION = "eef_initial_and_post_action_20hz_v1"
NAMES = ("time", "jerk", "sparc", "trajectory", "rotation")
FIELDS = ("steps", "rms_cartesian_jerk", "sparc_cost", "cartesian_path_length", "orientation_path_length")
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/parc_proxy_score_v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion_metrics(positions, quaternions, *, steps, dt=0.05, sampling="initial_and_post_action"):
    p = np.asarray(positions, dtype=np.float64)
    q = np.asarray(quaternions, dtype=np.float64)
    expected = steps + 1 if sampling == "initial_and_post_action" else steps
    if sampling not in ("initial_and_post_action", "pre_action"):
        raise ValueError("Unknown trajectory sampling convention")
    if not isinstance(steps, (int, np.integer)) or steps < 1 or dt <= 0 or not np.isfinite(dt):
        raise ValueError("Invalid physical step count or dt")
    if p.shape != (expected, 3) or q.shape != (expected, 4):
        raise ValueError("Trajectory frames must match every physical control step")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite observed pose")
    lengths = np.linalg.norm(q, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Zero quaternion is not an observed orientation")
    q = q / lengths[:, None]
    delta = np.diff(p, axis=0)
    path = float(np.linalg.norm(delta, axis=1).sum())
    angle = 2 * np.arccos(np.clip(np.abs(np.sum(q[:-1] * q[1:], axis=1)), 0, 1))
    jerk = np.linalg.norm(np.diff(p, n=3, axis=0) / dt**3, axis=1)
    sparc = float(compute_sparc(p, dt)) if len(p) >= 4 else float("nan")
    valid_sparc = sparc if np.isfinite(sparc) else None
    return {
        "metrics_version": METRICS_VERSION, "sampling": sampling,
        "control_dt": dt, "steps": int(steps), "simulated_time_sec": steps * dt,
        "pose_frames": len(p), "cartesian_path_length": path,
        "orientation_path_length": float(angle.sum()),
        "avg_cartesian_jerk": float(jerk.mean()) if len(jerk) else None,
        "rms_cartesian_jerk": float(np.sqrt(np.mean(jerk**2))) if len(jerk) else None,
        "sparc": valid_sparc, "sparc_cost": -valid_sparc if valid_sparc is not None else None,
        "unavailable": [name for name, ok in (("jerk", len(jerk) > 0), ("sparc", valid_sparc is not None)) if not ok],
    }


class MotionRecorder:
    def __init__(self, observation):
        self.positions, self.quaternions = [], []
        self.observe(observation)

    def observe(self, observation):
        self.positions.append(np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy())
        self.quaternions.append(np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy())

    def save(self, output, episode_key, steps):
        return save_motion(output, episode_key, self.positions, self.quaternions, steps=steps)


class InitialPoseCapturePolicy:
    """Observe the first policy input while delegating every action unchanged."""

    def __init__(self, policy):
        self.policy = policy
        self.initial_poses = []
        self.pending = False

    def reset(self, *args, **kwargs):
        self.pending = True
        return self.policy.reset(*args, **kwargs)

    def get_action(self, observation):
        if self.pending:
            self.initial_poses.append((np.asarray(observation["robot0_eef_pos"]).copy(),
                                       np.asarray(observation["robot0_eef_quat"]).copy()))
            self.pending = False
        return self.policy.get_action(observation)


def save_motion(output, episode_key, positions, quaternions, *, steps, sampling="initial_and_post_action"):
    metrics = motion_metrics(positions, quaternions, steps=steps, sampling=sampling)
    stream = io.BytesIO()
    np.savez_compressed(stream, positions=np.asarray(positions, dtype=np.float64),
                        quaternions_xyzw=np.asarray(quaternions, dtype=np.float64),
                        control_dt=np.array(0.05), steps=np.array(steps), sampling=np.array(sampling))
    payload = stream.getvalue(); checksum = hashlib.sha256(payload).hexdigest()
    key = hashlib.sha256(str(episode_key).encode()).hexdigest()[:16]
    relative = Path("motion") / f"{key}_{checksum[:16]}.npz"
    target = Path(output) / relative; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != checksum:
            raise ValueError("Existing trajectory archive does not match")
    else:
        temporary = target.with_suffix(".partial"); temporary.write_bytes(payload); temporary.replace(target)
    return {"motion_metrics": metrics, "motion_archive": {"path": str(relative), "sha256": checksum}}


def load_config(path=None):
    path = Path(path or DEFAULT_CONFIG); cfg = json.loads(path.read_text())
    if cfg.get("official_score") is not False:
        raise ValueError("A proxy configuration must explicitly disclaim official-score equivalence")
    weights = cfg["weights"]; refs = cfg["reference_costs"]
    if set(weights) != set(NAMES) or set(refs) != set(NAMES):
        raise ValueError("Exactly five metric weights and references are required")
    if not all(np.isfinite(v) and v >= 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Weights must be finite, nonnegative, and sum to one")
    if not all(np.isfinite(v) and v > 0 for v in refs.values()):
        raise ValueError("Reference costs must be finite and strictly positive")
    if cfg["normalization"] != "reference_over_reference_plus_cost" or cfg["metrics_version"] != METRICS_VERSION:
        raise ValueError("Unsupported normalization or metric version")
    if type(cfg["max_environment_steps"]) is not int or cfg["max_environment_steps"] < 1:
        raise ValueError("Invalid scoring step budget")
    lo, hi = cfg["weight_sensitivity_bounds"]
    if not 0 <= lo <= hi <= 1 or not 5 * lo <= 1 <= 5 * hi:
        raise ValueError("Infeasible sensitivity bounds")
    cfg = dict(cfg, config_sha256=digest(path))
    return cfg


def episode_score(row, cfg):
    for key in ("simulator_success", "collision"):
        if type(row.get(key)) is not bool:
            raise ValueError(f"Missing or non-boolean {key}")
    steps = row.get("steps")
    if (type(steps) is int and steps == 0 and not row["simulator_success"]
            and not row["collision"] and row.get("termination_reason") == "reset_http_error"
            and row.get("reset_error")):
        return {"status": "reset_failed_zero", "score": 0., "contributions": dict.fromkeys(NAMES, 0.)}
    if type(steps) is not int or steps < 1:
        raise ValueError("Missing or invalid physical steps")
    gated = not row["simulator_success"] or row["collision"] or steps > cfg["max_environment_steps"]
    if gated:
        return {"status": "gated_zero", "score": 0., "contributions": dict.fromkeys(NAMES, 0.)}
    m = row.get("motion_metrics", {})
    if m.get("metrics_version") != METRICS_VERSION or m.get("sampling") != "initial_and_post_action":
        return {"status": "unavailable_motion_metrics", "score": None}
    if m.get("steps") != steps or not np.isclose(m.get("control_dt", 0), 0.05):
        raise ValueError("Motion metrics have a different step count or control frequency")
    values = [m.get(f) for f in FIELDS]
    if any(v is None or not np.isfinite(v) or v < 0 for v in values):
        return {"status": "unavailable_motion_metrics", "score": None}
    norm = {name: cfg["reference_costs"][name] / (cfg["reference_costs"][name] + value)
            for name, value in zip(NAMES, values)}
    return {"status": "scored", "score": sum(cfg["weights"][k] * norm[k] for k in NAMES),
            "contributions": norm}


def weight_extreme(components, lower, upper, maximize):
    weights = dict.fromkeys(NAMES, lower); remaining = 1 - lower * len(NAMES)
    for name in sorted(NAMES, key=lambda k: components[k], reverse=maximize):
        extra = min(remaining, upper - lower); weights[name] += extra; remaining -= extra
    return {"score": sum(weights[k] * components[k] for k in NAMES), "weights": weights}


def aggregate_scores(rows, cfg, *, expected_episodes=None):
    rows = list(rows); expected = len(rows) if expected_episodes is None else expected_episodes
    if expected < len(rows):
        raise ValueError("More episodes than the expected evaluation inventory")
    scores = [episode_score(r, cfg) for r in rows]
    missing = [i for i, s in enumerate(scores) if s["score"] is None]
    result = {"official_score": False, "config_id": cfg["config_id"],
              "config_sha256": cfg["config_sha256"], "weights": cfg["weights"],
              "normalization": cfg["normalization"], "reference_costs": cfg["reference_costs"],
              "completed_episodes": len(rows), "expected_episodes": expected,
              "unscorable_episode_indices": missing, "score": None,
              "status": "incomplete" if len(rows) != expected else "unavailable_motion_metrics"}
    if expected and len(rows) == expected and not missing:
        components = {k: sum(s["contributions"][k] for s in scores) / expected for k in NAMES}
        low, high = cfg["weight_sensitivity_bounds"]
        result.update(status="complete", score=sum(cfg["weights"][k] * components[k] for k in NAMES),
                      gated_component_means=components,
                      weight_sensitivity={"lower": weight_extreme(components, low, high, False),
                                          "upper": weight_extreme(components, low, high, True),
                                          "interpretation": "assumed weight range only; not an official-score confidence interval"})
    result["raw_metrics"] = {}
    for label, selected in (("all_episodes", rows), ("collision_free_successes", [r for r in rows if r["simulator_success"] and not r["collision"]])):
        result["raw_metrics"][label] = {}
        for field in ("steps", "rms_cartesian_jerk", "sparc", "cartesian_path_length", "orientation_path_length"):
            values = [r["steps"] if field == "steps" else r.get("motion_metrics", {}).get(field) for r in selected]
            values = [v for v in values if v is not None and np.isfinite(v)]
            result["raw_metrics"][label][field] = {"count": len(values), "mean": float(np.mean(values)) if values else None}
    return result


def bind_score_config(output):
    """Prevent a resumed evaluation from silently changing score assumptions."""
    cfg = load_config()
    path = Path(output) / "proxy_score_config.json"
    if path.exists():
        if digest(path) != cfg["config_sha256"]:
            raise ValueError("Score configuration changed; use a new evaluation output")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial"); temporary.write_bytes(DEFAULT_CONFIG.read_bytes()); temporary.replace(path)
    return cfg


def default_score_summary(rows, expected_episodes=None, output=None):
    cfg = bind_score_config(output) if output is not None else load_config()
    return aggregate_scores(rows, cfg, expected_episodes=expected_episodes)
