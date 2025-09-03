from typing import TypedDict
import numpy as np


class UncertaintyDict(TypedDict):
    predictive: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray