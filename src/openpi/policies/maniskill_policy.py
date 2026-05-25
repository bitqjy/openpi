import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import libero_policy


ManiSkillInputs = libero_policy.LiberoInputs


@dataclasses.dataclass(frozen=True)
class ManiSkillOutputs(transforms.DataTransformFn):
    """Converts padded pi0 model actions back to ManiSkill action space."""

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
