"""
GAN-specific computation utilities.
Reusable for GANs (#14), VAE (#19).

Functions:
    compute_fid — Frechet Inception Distance between real and generated images
"""

import numpy as np


def compute_fid(real_images, fake_images, batch_size=50, device='cuda'):
    """
    Compute Frechet Inception Distance (FID) between real and generated images.

    Uses InceptionV3 pool3 features (2048-dim) to compare the distribution
    of real vs generated images. Lower FID = better quality + diversity.

    Args:
        real_images: numpy array (N, H, W, 3) or (N, 3, H, W), float32.
                     Any range accepted — will be rescaled to [0, 1] for Inception.
        fake_images: numpy array, same format as real_images.
        batch_size: Images per batch for feature extraction (manages GPU memory).
        device: 'cuda' or 'cpu' for PyTorch InceptionV3.

    Returns:
        fid_score: float, Frechet distance between the two distributions.
    """
    import torch
    from torchvision.models import inception_v3
    from scipy.linalg import sqrtm

    # Helper: preprocess images for InceptionV3
    def preprocess(images):
        """Convert numpy images to Inception-ready tensors."""
        x = images.copy()

        # Handle channel-first (N, 3, H, W) -> keep as-is
        # Handle channel-last (N, H, W, 3) -> transpose
        if x.ndim == 4 and x.shape[-1] == 3:
            x = np.transpose(x, (0, 3, 1, 2))  # -> (N, 3, H, W)

        # Rescale to [0, 1] if in [-1, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0

        # Inception expects 299x299 — resize via interpolation
        t = torch.tensor(x, dtype=torch.float32)
        t = torch.nn.functional.interpolate(t, size=(299, 299), mode='bilinear',
                                            align_corners=False)

        # Inception normalization: ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std
        return t

    # Load InceptionV3 (pool3 features)
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove classification head -> 2048-dim features
    model = model.to(device)
    model.eval()

    # Extract features in batches
    def get_features(images):
        """Extract 2048-dim InceptionV3 features from images."""
        all_features = []
        n = len(images)
        for i in range(0, n, batch_size):
            batch = preprocess(images[i:i + batch_size]).to(device)
            with torch.no_grad():
                features = model(batch)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    print(f"    Extracting features from {len(real_images)} real images...")
    real_features = get_features(real_images)
    print(f"    Extracting features from {len(fake_images)} fake images...")
    fake_features = get_features(fake_images)

    # Compute FID
    # FID = ||mu_r - mu_f||^2 + Tr(sigma_r + sigma_f - 2 * sqrt(sigma_r @ sigma_f))
    mu_r, sigma_r = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_f, sigma_f = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    cov_sqrt, _ = sqrtm(sigma_r @ sigma_f, disp=False)

    # Numerical stability: discard imaginary component from sqrtm
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = np.dot(diff, diff) + np.trace(sigma_r + sigma_f - 2.0 * cov_sqrt)
    return float(fid)