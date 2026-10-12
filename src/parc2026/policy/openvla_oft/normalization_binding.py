"""Bind fixed checkpoint statistics to the model that executes predict_action."""
from __future__ import annotations


def bind_checkpoint_statistics(model, statistics, key):
    if key not in statistics:
        raise KeyError('Checkpoint does not contain normalization key: '+key)
    unwrap=getattr(model,'get_base_model',None)
    implementation=unwrap() if callable(unwrap) else model
    # PEFT forwards predict_action as a bound method of its underlying model.
    # An attribute assigned only to the outer wrapper is invisible to that method.
    implementation.norm_stats=statistics
    if implementation is not model:
        model.norm_stats=statistics
    getter=getattr(implementation,'get_action_stats',None)
    if callable(getter) and getter(key)!=statistics[key]['action']:
        raise ValueError('Inference normalization binding failed')
    return implementation
