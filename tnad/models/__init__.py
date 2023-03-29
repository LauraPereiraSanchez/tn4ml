""" Module models
"""

from .model import (
    Model,
    load_model,
    LossWrapper,
    _fit
)

from .smpo import(
    SpacedMatrixProductOperator
)

from .deep_model import(
    DeepTensorNetwork
)