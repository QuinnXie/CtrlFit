"""Persistence helpers for fitted ctrlfit surrogate models."""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional, Tuple

from jax_sysid.models import Model


def save_training_info(training_info: Dict[str, Any], filename: str) -> None:
    """Save model parameters and training metadata to a pickle file."""

    model = training_info["model"]
    save_data = dict(training_info)
    save_data["model_params"] = model.params
    save_data["model_x0"] = getattr(model, "x0", None)
    save_data.pop("model", None)
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(save_data, f)


def load_training_info(filename: str, model: Optional[Model] = None) -> Tuple[Model, Dict[str, Any]]:
    """Load a saved surrogate model.

    The package core does not define architectures, so custom models must be
    recreated by user code and passed through the model argument.
    """

    with open(filename, "rb") as f:
        save_data = pickle.load(f)

    if model is None:
        raise ValueError("Pass model=... when loading; model definitions live in user/example code")

    model.params = save_data["model_params"]
    if save_data.get("model_x0") is not None:
        model.x0 = save_data["model_x0"]
    training_info = dict(save_data)
    training_info["model"] = model
    return model, training_info


__all__ = [
    "load_training_info",
    "save_training_info",
]
