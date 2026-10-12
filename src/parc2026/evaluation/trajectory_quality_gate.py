"""Reusable quality-change test; physical success/safety require separate replay.

Weights and normalization are explicit local assumptions, never official values.
The worst-case test compares ungated quality, including for over-budget parents.
"""
import numpy as np

from .proxy_score import NAMES, FIELDS, weight_extreme


def assess_quality_change(before, after, config, *, maximum_steps=280, minimum_gain=1e-4):
    old=dict(zip(NAMES,(before[k] for k in FIELDS)))
    new=dict(zip(NAMES,(after[k] for k in FIELDS)))
    if any(v is None or not np.isfinite(v) or v<0 for v in [*old.values(),*new.values()]):
        return dict(accepted=False,reasons=['invalid_motion_metric'])
    norm=lambda costs:{k:config['reference_costs'][k]/(config['reference_costs'][k]+costs[k]) for k in NAMES}
    a,b=norm(old),norm(new)
    changes={k:b[k]-a[k] for k in NAMES}
    worst=weight_extreme(changes,*config['weight_sensitivity_bounds'],False)
    score=lambda x:sum(config['weights'][k]*x[k] for k in NAMES)
    reasons=[]
    if new['time']>maximum_steps:reasons.append('over_'+str(maximum_steps)+'_steps')
    if new['time']>old['time']:reasons.append('more_steps')
    # Additional guard for already scorable parents. A long-parent rescue still
    # has to improve ungated quality for every weight in the assumed range.
    if old['time']<=300 and any(new[k]>old[k]*1.05+1e-8 for k in NAMES):reasons.append('cost_regression_over_5pct')
    if worst['score']<=minimum_gain:reasons.append('no_robust_proxy_improvement')
    before_gate=float(old['time']<=300)
    return dict(accepted=not reasons,reasons=reasons,before_costs=old,after_costs=new,
                before_quality=score(a),after_quality=score(b),worst_weight_improvement=worst,
                before_gated_proxy=before_gate*score(a),after_gated_proxy=float(new['time']<=300)*score(b),
                parent_over_evaluation_budget=not bool(before_gate),
                pareto_no_cost_regression=all(new[k]<=old[k]+max(1e-8,old[k]*1e-6) for k in NAMES),
                config_sha256=config['config_sha256'],official_score=False,
                physical_replay_success_and_safety_required=True)
