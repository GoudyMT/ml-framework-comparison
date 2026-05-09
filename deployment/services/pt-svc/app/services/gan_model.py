"""
DCGAN Generator architecture for D3 (CIFAR-10 image generation).

WHAT THIS FILE IS:
    JUST the generator class. No I/O, no MLflow, no denormalization.
    The trained weights live in the registry as a `.pth` state_dict;
    this class is the architecture spec those weights deserialize INTO.

    Loading flow:
        1. Instantiate this class    (random initial weights)
        2. torch.load(state_dict)    (the trained weights from disk)
        3. model.load_state_dict(sd) (overwrites random with trained)
        4. model.eval()              (inference mode)
    The loader (app/services/gan_loader.py) does steps 2-4. This file
    only owns step 1.

    Note that we deploy ONLY the generator, not the discriminator. At
    inference we sample noise -> decode to images; the discriminator's
    job (telling real from fake) was a training-time aid that we don't
    need any more.

WHY THE ARCHITECTURE LIVES IN CODE:
    Same reason as DNN: PyTorch state_dicts contain TENSOR VALUES keyed
    by parameter name (e.g., "main.0.weight" -> shape (100, 256, 4, 4)).
    They do NOT contain the architectural graph. The deployment service
    must replicate the trained class exactly. Mismatched architecture
    -> load_state_dict raises "Missing key(s)" or "Unexpected key(s)".

    The state_dict for our trained DCGAN has 19 tensors, 1,069,827
    parameters, and the keys "main.0.weight" through "main.9.weight"
    (with batch-norm params on indices 1, 4, 7). The class below has
    been built to match those keys exactly - any drift will break
    load_state_dict.

WHY THIS ARCHITECTURE (z(100,1,1) -> (3,32,32) DCGAN, FID 30.57):
    Pulled from the modeling phase: PyTorch/14-gans/pipeline.ipynb,
    class DCGenerator (lines 429-455). DCGAN paper recipe (Radford
    et al. 2015): four transposed convolutions with kernel=4, batch
    norm after every conv except the last, ReLU activations until the
    final tanh. Each transposed conv doubles the spatial dimension
    (4 -> 8 -> 16 -> 32) and halves the channel count after the first
    layer (256 -> 128 -> 64 -> 3).

    These widths are hardcoded to the trained model's shapes - this
    class is NOT configurable. If retrained with a different generator
    architecture, this file is updated and the model version is bumped
    in the registry.

INPUT/OUTPUT CONTRACT:
    Input  z:  (batch, 100, 1, 1)  float32, drawn from N(0, 1)
    Output:    (batch, 3, 32, 32)  float32, in [-1, 1] (tanh-bounded)

    Denormalization to uint8 [0, 255] for PNG encoding happens in the
    router, not here - same separation as DNN where softmax happens in
    the router. This keeps the architecture class identical to training.
"""

from typing import cast

import torch
from torch import nn

from app.schemas.gan import IMAGE_CHANNELS, LATENT_DIM

# Channel widths at each transposed-conv output stage.

"""
Hardcoded to the trained model's shapes. The DCGAN paper convention
is "ngf" (number of generator features) with widths ngf*4, ngf*2,
ngf - which here would be ngf=64. We don't parameterize that because
the trained weights are tied to these exact widths. Retraining with
different widths means updating this file and bumping the registry
version (same rule as DNN's HIDDEN_1/HIDDEN_2).
"""
CHANNELS_1: int = 256       # after first ConvTranspose:  (256, 4, 4)
CHANNELS_2: int = 128       # after second ConvTranspose: (128, 8, 8)
CHANNELS_3: int = 64        # after third ConvTranspose:  (64, 16, 16)
# Final ConvTranspose maps CHANNELS_3 -> IMAGE_CHANNELS (3) at 32x32.

"""
Transposed-conv hyperparameters that are constant across all four
layers in this generator. Pulled out as named constants so the
Sequential below reads cleanly instead of as a wall of magic numbers.
"""
KERNEL_SIZE: int = 4
"""
First layer uses stride=1, padding=0 to expand 1x1 -> 4x4. The
remaining three layers use stride=2, padding=1 to double the
spatial dimension at each step (4 -> 8, 8 -> 16, 16 -> 32).
"""
STRIDE_FIRST: int = 1
PAD_FIRST: int = 0
STRIDE_REST: int = 2
PAD_REST: int = 1


class DCGenerator(nn.Module):
    """
    DCGAN generator: noise -> 32x32 RGB image.

    Sequential layout (must match state_dict keys exactly):
        [0] ConvTranspose2d(100, 256, k=4, s=1, p=0, bias=False)  # 1x1 -> 4x4
        [1] BatchNorm2d(256)
        [2] ReLU(inplace=True)
        [3] ConvTranspose2d(256, 128, k=4, s=2, p=1, bias=False)  # 4x4 -> 8x8
        [4] BatchNorm2d(128)
        [5] ReLU(inplace=True)
        [6] ConvTranspose2d(128, 64,  k=4, s=2, p=1, bias=False)  # 8x8 -> 16x16
        [7] BatchNorm2d(64)
        [8] ReLU(inplace=True)
        [9] ConvTranspose2d(64,  3,   k=4, s=2, p=1, bias=False)  # 16x16 -> 32x32
        [10] Tanh()

    SEQUENTIAL INDEX MATTERS:
        State_dict keys encode the layer index ("main.0.weight",
        "main.4.bias", etc.). Activation layers (ReLU, Tanh) have no
        parameters, so they don't appear in the state_dict - BUT they
        still occupy slots in the Sequential chain. Removing them would
        shift the next ConvTranspose layer's index, and load_state_dict
        would fail to match keys.

    WHY bias=False ON EVERY CONVTRANSPOSE:
        DCGAN convention - any conv layer immediately followed by
        BatchNorm has its bias zeroed out by the BN normalization
        anyway, so the explicit bias is redundant and just wastes
        parameters. The final ConvTranspose has no BN after it, so
        in principle it COULD use bias - but the source notebook chose
        bias=False there too (likely for symmetry; the tanh saturates
        the output regardless of the small bias contribution).

        Either way: the trained state_dict has zero "*.bias" entries
        for the four ConvTranspose layers; flipping bias=True here
        would generate Unexpected keys at load time.

    WHY ReLU(True):
        The `True` argument is `inplace=True`, which overwrites the
        input tensor instead of allocating a new one - small memory
        win during training. ReLU has no parameters, so this choice
        doesn't show up in the state_dict; we mirror the source
        notebook for fidelity, but using ReLU() (not inplace) would
        also load successfully.

    OUTPUT IS IN [-1, 1], NOT [0, 255]:
        Tanh squashes the final layer's output to [-1, 1]. The router
        denormalizes via (x + 1) * 127.5 -> uint8 [0, 255] and encodes
        as PNG. Reasons for keeping that split:
            - Class identical to the training class (no extra layers)
            - Denormalization is plain tensor math; doesn't deserve to
              be a layer
            - Easier to write the smoke test: it can compare raw [-1,1]
              tensor outputs bit-exact, before encoding artifacts
    """

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(
                LATENT_DIM, CHANNELS_1,
                KERNEL_SIZE, STRIDE_FIRST, PAD_FIRST, bias=False,
            ),
            nn.BatchNorm2d(CHANNELS_1),
            nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(
                CHANNELS_1, CHANNELS_2,
                KERNEL_SIZE, STRIDE_REST, PAD_REST, bias=False,
            ),
            nn.BatchNorm2d(CHANNELS_2),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                CHANNELS_2, CHANNELS_3,
                KERNEL_SIZE, STRIDE_REST, PAD_REST, bias=False,
            ),
            nn.BatchNorm2d(CHANNELS_3),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                CHANNELS_3, IMAGE_CHANNELS,
                KERNEL_SIZE, STRIDE_REST, PAD_REST, bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Noise tensor of shape (batch, LATENT_DIM, 1, 1) =
                (B, 100, 1, 1). Must be float32 to match the trained
                weights. Standard practice samples z from N(0, 1)
                via torch.randn(...), but any float tensor of the
                right shape will run.

        Returns:
            Generated images tensor of shape (batch, 3, 32, 32) in
            [-1, 1] (tanh-bounded). Apply (x + 1) * 127.5 -> uint8
            in the caller to get displayable images.
        """
        # cast() narrows the inferred type: PyTorch's stubs declare
        # nn.Sequential.__call__ as returning Any, but at runtime it
        # returns a Tensor. Without cast, mypy strict raises no-any-return.
        return cast(torch.Tensor, self.main(z))
