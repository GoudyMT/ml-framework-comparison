"""
DNN architecture for D2 (UCI HAR activity classification).

WHAT THIS FILE IS:
    JUST the model class. No I/O, no MLflow, no scaler. The trained
    weights live in the registry as a `.pth` state_dict; this class is
    the architecture spec those weights deserialize INTO.

    Loading flow:
        1. Instantiate this class    (random initial weights)
        2. torch.load(state_dict)    (the trained weights from disk)
        3. model.load_state_dict(sd) (overwrites random with trained)
        4. model.eval()              (inference mode)
    The loader (app/services/dnn_loader.py) does steps 2-4. This file
    only owns step 1.

WHY THE ARCHITECTURE LIVES IN CODE:
    PyTorch state_dicts contain TENSOR VALUES keyed by parameter
    name (e.g., "net.0.weight" -> shape (256, 561)). They do NOT
    contain the architectural graph - just the weights for an
    architecture you must define separately.

    This is a fundamental difference from sklearn estimators (which
    pickle the whole class instance, architecture + weights together).
    It means the deployment service MUST replicate the trained class
    exactly. Mismatched architecture -> load_state_dict raises
    "Missing key(s)" or "Unexpected key(s)".

WHY THIS ARCHITECTURE (561 -> 256 -> 128 -> 6 + BatchNorm + Dropout):
    Pulled from the modeling phase: 96.03% test accuracy on UCI HAR,
    178K parameters, 0.42 us/sample inference. The hidden sizes,
    batch norm placement, and dropout rate were tuned during training
    (see PyTorch/09-dnn/pipeline.ipynb). Deployment freezes those
    choices - this class is NOT configurable; if the model is retrained
    with different hidden sizes, this file is updated and the model
    version is bumped in the registry.
"""

from typing import cast

import torch
from torch import nn

from app.schemas.dnn import INPUT_DIM, N_CLASSES

# Architectural constants - hardcoded to the trained model's shape.
# Hidden sizes come from the modeling-phase metrics.json:
# "architecture": "561-256(BN+DO)-128(BN+DO)-6"
HIDDEN_1: int = 256
HIDDEN_2: int = 128

# Dropout rate from training. At eval() time, Dropout is a no-op
# (passes input through unchanged), so this rate is informational at
# inference. We keep the layers in the Sequential chain anyway because
# they affect index-based state_dict keys (see class docstring).
DROPOUT: float = 0.3


class DNN(nn.Module):
    """
    UCI HAR DNN classifier.

    Sequential layout:
        [0] Linear(561 -> 256)
        [1] BatchNorm1d(256)
        [2] ReLU
        [3] Dropout(0.3)
        [4] Linear(256 -> 128)
        [5] BatchNorm1d(128)
        [6] ReLU
        [7] Dropout(0.3)
        [8] Linear(128 -> 6)        # logits, NOT softmax-applied

    SEQUENTIAL INDEX MATTERS:
        State_dict keys encode the layer index ("net.0.weight",
        "net.4.bias", etc.). ReLU + Dropout have no parameters, so
        they don't appear in the state_dict - BUT they still occupy
        slots in the Sequential chain. Removing them would shift the
        next Linear layer's index from 4 to 2, and load_state_dict
        would fail to match keys.

        Lesson: when defining a deployment class, copy the training
        Sequential layout EXACTLY, parameterless layers included.

    OUTPUT IS LOGITS, NOT PROBABILITIES:
        Standard PyTorch convention: the network returns raw logits
        (unbounded floats), and the loss function (CrossEntropyLoss)
        applies log-softmax internally during training. At inference
        we apply softmax in the ROUTER, not here. Reasons:
            - Keeps the model class identical to training
            - argmax(logits) == argmax(softmax(logits)), so for the
              "predicted_class" output we don't need softmax at all
            - For "probabilities" output we apply softmax once in the
              handler, not on every forward call
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.BatchNorm1d(HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_2, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, INPUT_DIM) = (B, 561).
                Must be float32 to match the trained weights.

        Returns:
            Logits tensor of shape (batch, N_CLASSES) = (B, 6).
            Apply torch.softmax(dim=1) in the caller to get
            probabilities; argmax(dim=1) for the predicted class.
        """
        # nn.Module's __call__ wraps forward() and runs registered
        # hooks (BatchNorm uses its eval-mode running stats only when
        # the model is in eval() mode - that's the loader's responsibility).
        # cast() narrows the inferred type: PyTorch's stubs declare
        # nn.Sequential.__call__ as returning Any, but at runtime it
        # returns a Tensor. Without cast, mypy strict raises no-any-return.
        return cast(torch.Tensor, self.net(x))
