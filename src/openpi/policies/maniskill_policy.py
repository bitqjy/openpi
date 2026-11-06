import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import libero_policy


@dataclasses.dataclass(frozen=True)
class ManiSkillInputs(libero_policy.LiberoInputs):
    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)
        # Optional supervision travels alongside observations, never in state/prompt.
        for key in ("phase_targets", "phase_confidence"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class ManiSkillOutputs(transforms.DataTransformFn):
    """Converts padded pi0 model actions back to ManiSkill action space."""

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
