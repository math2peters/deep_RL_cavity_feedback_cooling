"""Shared differentiator baseline controller used by public evaluation scripts."""

import numpy as np


class DifferentiatorController:
    """Version-stable differentiator baseline used for paper comparisons."""

    def __init__(self, gain=0.75, base_action=0.8):
        self.gain = gain
        self.base_action = base_action

    def predict(self, observation, deterministic=True):
        previous_count = observation[-7]
        current_count = observation[-3]
        raw_action = self.base_action - self.gain * (current_count - previous_count)
        action = np.sqrt(abs(np.array([raw_action])))
        action = np.sign(raw_action) * action
        return action, None


def is_differentiator_model(model_path):
    """Return True when a model path refers to the differentiator baseline."""
    return "differentiator" in str(model_path)
