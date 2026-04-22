"""
Variational Autoencoder utilities for Model #19.

Provides the VAE mathematical primitives (reparameterization trick,
KL divergence) plus VQ-VAE-specific ops (straight-through estimator,
commitment loss) plus visualization helpers (latent traversal,
interpolation).
"""

import torch
import torch.nn.functional as F


def reparameterize(mu, logvar):
    """
    VAE reparameterization trick: sample z = mu + eps * sigma, eps ~ N(0, I).

    Decouples the stochastic sampling from the learned parameters so
    gradients flow from z back through mu and logvar. Without this,
    sampling from N(mu, sigma^2) directly would block backprop.

    Args:
        mu: Tensor (B, D) of encoder mean predictions.
        logvar: Tensor (B, D) of encoder log-variance predictions.
            Using logvar instead of sigma^2 keeps the output in the
            real line (encoder doesn't need a softplus to stay positive).

    Returns:
        z: Tensor (B, D) latent sample. Same shape as mu.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence_gaussian(mu, logvar):
    """
    Analytical KL divergence from N(mu, sigma^2) to N(0, I).

    Closed-form for diagonal Gaussians:
        KL = -0.5 * sum_d (1 + logvar_d - mu_d^2 - exp(logvar_d))

    Returns per-batch KL (summed over latent dims). Training loss
    should average over batch dim — this matches the ELBO convention
    where we sum over latent and average over batch.

    Args:
        mu: Tensor (B, D) means.
        logvar: Tensor (B, D) log-variances.

    Returns:
        kl: Tensor (B,) per-sample KL values. Always >= 0.
    """
    return -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=-1,
    )


def vq_straight_through(z_e, codebook):
    """
    Vector-quantize z_e to nearest codebook entry with straight-through gradient.

    Forward pass: finds nearest codebook entry by L2 distance, returns it.
    Backward pass: gradient w.r.t. z_e is identity (argmin is non-differentiable
    so we pretend quantization was a no-op for grad purposes). This is the
    "straight-through estimator" from van den Oord et al. 2017.

    The trick: `z_q = z_e + (z_q_flat - z_e).detach()`. Forward reads z_q_flat;
    backward sees the identity map from z_e because the added term has no
    gradient.

    Args:
        z_e: Tensor (B, ..., D). Continuous encoder output. The last dim must
            match codebook dim; intermediate dims (e.g., spatial H, W for conv
            VQ-VAE) are handled by flattening and reshaping back.
        codebook: Tensor (K, D). K learned codes of dim D.

    Returns:
        z_q: Tensor same shape as z_e. Quantized latent with straight-through grad.
        indices: LongTensor same shape as z_e[..., 0]. Selected code index per
            spatial location (for codebook utilization tracking).
    """
    # Flatten trailing dims to (N, D) for distance computation
    original_shape = z_e.shape
    z_e_flat = z_e.reshape(-1, z_e.shape[-1])  # (N, D)

    # Squared distance: ||x - c||^2 = x.x - 2*x.c + c.c
    distances = (
        z_e_flat.pow(2).sum(dim=1, keepdim=True)          # (N, 1)
        - 2 * z_e_flat @ codebook.T                       # (N, K)
        + codebook.pow(2).sum(dim=1, keepdim=True).T      # (1, K)
    )  # (N, K)

    indices = distances.argmin(dim=1)                     # (N,)
    z_q_flat = codebook[indices]                          # (N, D)
    z_q = z_q_flat.reshape(original_shape)

    # Straight-through: forward uses z_q_flat, backward passes grad to z_e unchanged
    z_q = z_e + (z_q - z_e).detach()

    # Reshape indices to (B, ...) — drop last D dim
    indices = indices.reshape(*original_shape[:-1])
    return z_q, indices


def vq_commit_loss(z_e, z_q, beta=0.25):
    """
    VQ-VAE commitment loss (van den Oord et al. 2017, eq. 3).

    Two terms:
        codebook_loss: ||sg[z_e] - z_q||^2 -- trains the codebook to match z_e
        commit_loss:   ||z_e - sg[z_q]||^2 -- trains the encoder to commit to codes

    `sg[.]` denotes stop-gradient. Without the commitment term the encoder can
    drift arbitrarily; beta=0.25 (paper default) keeps encoder tethered to
    nearest code without dominating reconstruction loss.

    Args:
        z_e: Tensor, continuous encoder output (pre-quantization).
        z_q: Tensor, quantized version from vq_straight_through (same shape).
        beta: Commitment weight. Paper: 0.25. Sensible range: 0.1-1.0.

    Returns:
        loss: scalar Tensor. Add to reconstruction loss during training.
    """
    codebook_loss = F.mse_loss(z_q, z_e.detach())
    commitment_loss = F.mse_loss(z_e, z_q.detach())
    return codebook_loss + beta * commitment_loss


def latent_traversal(decoder, base_z, dim, n_steps=11, range_=(-3.0, 3.0)):
    """
    Traverse one latent dimension while fixing all others. Used for
    beta-VAE disentanglement visualization (V3): if beta is well-tuned,
    each dim controls a single semantic factor (stroke width, tilt, etc.)
    that the reader can see by watching the decoded output sweep.

    Args:
        decoder: Callable mapping (N, D) latent to (N, *image_shape).
            Typically model.decode or lambda z: model.forward_decode(z).
        base_z: Tensor (D,). The latent point to traverse from (often
            a specific sample's posterior mean).
        dim: int. Which latent dim to vary.
        n_steps: int. Number of decoded images along the traversal.
        range_: (min, max). Range of values to sweep dim across.
            Default [-3, 3] covers ~99% of N(0, 1) mass.

    Returns:
        images: Tensor (n_steps, *image_shape). Decoded traversal,
            ready for plot_generated_grid or matplotlib display.
    """
    values = torch.linspace(range_[0], range_[1], n_steps, device=base_z.device)
    batch = base_z.unsqueeze(0).repeat(n_steps, 1)       # (n_steps, D)
    batch[:, dim] = values
    with torch.no_grad():
        return decoder(batch)


def interpolate_latent(decoder, z1, z2, n_steps=10):
    """
    Linear interpolation between two latent points, decoding each step.
    Shows whether the latent space is geometrically smooth — well-trained
    VAE should produce gradual visual transitions (one digit morphing
    into another), while a broken latent shows abrupt jumps.

    Args:
        decoder: Callable (N, D) -> (N, *image_shape).
        z1: Tensor (D,). Starting latent.
        z2: Tensor (D,). Ending latent.
        n_steps: int. Number of interpolation points (including endpoints).

    Returns:
        images: Tensor (n_steps, *image_shape). Decoded interpolation.
    """
    alphas = torch.linspace(0.0, 1.0, n_steps, device=z1.device).unsqueeze(1)  # (n_steps, 1)
    zs = (1 - alphas) * z1.unsqueeze(0) + alphas * z2.unsqueeze(0)              # (n_steps, D)
    with torch.no_grad():
        return decoder(zs)