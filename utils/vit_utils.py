"""
Vision Transformer utilities for Model #17.
Augmentation collation (MixUp/CutMix), DeiT distillation loss,
and attention visualization helpers used across PyTorch ViT pipelines.
"""

import numpy as np
import torch
import torch.nn.functional as F

def apply_mixup_cutmix(images, labels, num_classes, mixup_alpha=0.8,
                       cutmix_alpha=1.0, prob=0.5):
    """
    Apply MixUp or CutMix to a batch, returning mixed images and soft labels.

    At each call, randomly picks MixUp (prob 1-prob) or CutMix (prob) for the
    whole batch. Both return soft one-hot targets that account for the mix
    ratio -- the training loss must handle soft labels (typical pattern:
        loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    ). Label smoothing should be disabled when this is active; soft labels
    already provide regularization and double-smoothing hurts.

    MixUp (Zhang et al. 2017): linear blend of two images + two labels.
        x_mixed = lam * x_a + (1-lam) * x_b
        y_mixed = lam * y_a + (1-lam) * y_b

    CutMix (Yun et al. 2019): paste a rectangular patch from image B into A.
        Lambda is adjusted based on actual patch area after boundary clipping.

    Apply AFTER the default collate in the training loop (operates on batched
    tensors for efficiency), NOT in the dataset's __getitem__.

    Args:
        images: (B, C, H, W) float tensor.
        labels: (B,) long tensor of class indices (must be dtype=long).
        num_classes: Number of classes (for one-hot conversion).
        mixup_alpha: Beta distribution alpha for MixUp (DeiT standard: 0.8).
        cutmix_alpha: Beta distribution alpha for CutMix (DeiT standard: 1.0).
        prob: Probability of using CutMix vs MixUp per batch (0.5 = 50/50).

    Returns:
        mixed_images: (B, C, H, W) float tensor with MixUp or CutMix applied.
        soft_labels: (B, num_classes) float tensor of mixed one-hot targets.
    """
    B = images.size(0)
    device = images.device

    # Random permutation pairs each sample with another in the batch
    perm = torch.randperm(B, device=device)

    # Hard labels -> float one-hot, for soft-label arithmetic
    labels_onehot = F.one_hot(labels, num_classes=num_classes).float()
    labels_perm = labels_onehot[perm]

    # Pick MixUp or CutMix for the whole batch
    if np.random.rand() < prob:
        # CutMix: paste a rectangular region from perm into current
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        H, W = images.shape[-2:]

        # Box side proportional to sqrt(1 - lam) so box area ~= (1 - lam) * total
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Random box center, clipped to image boundaries
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)

        # Paste box from perm into clone of original
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

        # Boundary clipping changes actual box area -> recompute lam
        lam_actual = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
        mixed_labels = lam_actual * labels_onehot + (1.0 - lam_actual) * labels_perm
    else:
        # MixUp: linear blend of both images and labels
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed_images = lam * images + (1.0 - lam) * images[perm]
        mixed_labels = lam * labels_onehot + (1.0 - lam) * labels_perm

    return mixed_images, mixed_labels

def distillation_loss(student_cls_logits, student_dist_logits, teacher_logits,
                      labels, distill_type='hard', alpha=0.5, temperature=3.0):
    """
    DeiT distillation loss with dual [CLS] + [DIST] classification heads.

    DeiT (Touvron et al. 2020) splits the student's classification into two heads:
        [CLS] head  -- trained against ground-truth labels (possibly soft if MixUp active)
        [DIST] head -- trained against teacher's predictions
    At inference, the two heads' logits are averaged for final prediction.

    Hard distillation (DeiT's finding -- best for ViT on small data):
        loss = alpha * CE(cls_logits, labels) + (1 - alpha) * CE(dist_logits, teacher_argmax)
        Teacher's argmax is a discrete target -- no temperature needed.

    Soft distillation (Hinton 2015, provided for comparison):
        loss = alpha * CE(cls_logits, labels) + (1 - alpha) * T^2 * KL(student/T || teacher/T)
        Temperature T softens both distributions; T^2 scales gradient magnitude.

    Handles both hard (long (B,)) and soft (float (B, C)) label formats, so this
    composes correctly with apply_mixup_cutmix output.

    Contract: teacher_logits must be produced with torch.no_grad() by the caller.
    This function does NOT freeze the teacher.

    Args:
        student_cls_logits: (B, num_classes) logits from student's [CLS] head.
        student_dist_logits: (B, num_classes) logits from student's [DIST] head.
        teacher_logits: (B, num_classes) logits from teacher (detached, no grad).
        labels: (B,) long OR (B, num_classes) float. Ground-truth for CLS head.
        distill_type: 'hard' (DeiT default) or 'soft' (Hinton KD).
        alpha: Weight on CLS head's ground-truth loss (1 - alpha goes to DIST).
               Default 0.5 balances both heads equally (DeiT spec).
        temperature: Softening temperature for soft distillation (ignored for hard).
                     Default 3.0 is standard from Hinton 2015.

    Returns:
        Scalar loss tensor.
    """
    # CLS head loss: CE against ground-truth (handles soft or hard labels)
    if labels.dim() == 1:
        cls_loss = F.cross_entropy(student_cls_logits, labels)
    else:
        # Soft labels from MixUp/CutMix -- manual soft-target CE
        cls_loss = -(labels * F.log_softmax(student_cls_logits, dim=-1)).sum(dim=-1).mean()

    # DIST head loss branches on distillation type
    if distill_type == 'hard':
        # Teacher's argmax is a discrete target -> plain CE
        teacher_targets = teacher_logits.argmax(dim=-1)
        dist_loss = F.cross_entropy(student_dist_logits, teacher_targets)
    elif distill_type == 'soft':
        # KL divergence between temperature-softened teacher and student
        T = temperature
        student_log_soft = F.log_softmax(student_dist_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        # T^2 preserves gradient scale per Hinton 2015
        dist_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (T * T)
    else:
        raise ValueError(f"distill_type must be 'hard' or 'soft', got {distill_type!r}")

    return alpha * cls_loss + (1.0 - alpha) * dist_loss

def attention_rollout(attn_maps_list):
    """
    Compute attention rollout across all encoder layers (Abnar & Zuidema 2020).

    Attention rollout tracks how much each output token depends on each input
    token through the full stack of self-attention layers, accounting for
    residual connections. Standard per-layer attention only shows one-hop
    dependencies; rollout propagates them end-to-end.

    Algorithm:
        For each layer l = 1, ..., L:
            A_bar_l = 0.5 * (mean_heads(A_l) + I)     [residual-augmented]
        rollout = A_bar_L @ A_bar_{L-1} @ ... @ A_bar_1

    The (A + I) / 2 step is the key insight from the paper: residual connections
    mean each token partially 'passes through' unchanged. Without this, rollout
    over-attributes to attended tokens and under-attributes to the token's own
    identity. Rows of A_bar_l sum to 1 (since both A_l's rows and I's rows do),
    so rollout also has row-sum = 1 -- a valid attention distribution.

    Args:
        attn_maps_list: List of post-softmax attention tensors, one per encoder
                        layer, each shaped (B, n_heads, seq_len, seq_len). Rows
                        of each attention matrix should sum to 1.

    Returns:
        rollout: (B, seq_len, seq_len) tensor where entry [b, i, j] is the
                 end-to-end attention from token i to token j at batch b.

    Raises:
        ValueError: if attn_maps_list is empty.
    """
    if len(attn_maps_list) == 0:
        raise ValueError("attn_maps_list must contain at least one layer")

    rollout = None
    for attn in attn_maps_list:
        # Average across heads: (B, n_heads, N, N) -> (B, N, N)
        A_avg = attn.mean(dim=1)

        # Add identity for residual, rescale so rows still sum to 1
        B, N, _ = A_avg.shape
        I = torch.eye(N, device=A_avg.device).unsqueeze(0).expand(B, -1, -1)
        A_with_res = 0.5 * A_avg + 0.5 * I

        # Chain into running rollout: new layer operates on previous layer's output
        if rollout is None:
            rollout = A_with_res
        else:
            rollout = A_with_res @ rollout

    return rollout

def cls_attention_map(attn_maps_list, image_hw=32, patch_size=4, method='rollout',
                      layer_idx=-1, has_dist_token=False):
    """
    Extract [CLS] token's attention to image patches, reshape to spatial grid,
    and upsample to the original image resolution for overlay visualization.

    ViT's [CLS] token sits at position 0 of the sequence; its row in the
    attention matrix tells us which image patches the classifier is 'looking
    at' when making its decision. Reshaping that row back to 2D and upsampling
    to image resolution produces a heatmap we can overlay on the original.

    Supports two extraction methods:
        'last'    -- attention from a single chosen layer (default: last, idx=-1).
                     Good for seeing one layer's behavior in isolation.
        'rollout' -- Abnar & Zuidema rollout across all layers.
                     Recommended for final visualizations -- captures end-to-end
                     dependency including residual contributions.

    Args:
        attn_maps_list: List of (B, n_heads, seq_len, seq_len) attention tensors.
        image_hw: Image side length (H == W). CIFAR-100 -> 32.
        patch_size: Patch side length. Must divide image_hw evenly. ViT-Small -> 4.
        method: 'rollout' (default) or 'last'.
        layer_idx: Which layer to use for method='last'. Default -1 (final layer).
        has_dist_token: True if the ViT has a [DIST] token at position 1 (DeiT V3).
                        When True, we extract position 0 -> 2: (excluding both
                        CLS and DIST tokens from keys).

    Returns:
        heatmap: (B, image_hw, image_hw) float tensor in [0, 1]. Can be overlaid
                 directly on the original RGB image using matplotlib's alpha
                 blending (e.g. alpha=0.5).

    Raises:
        ValueError: if method is not 'rollout' or 'last', or if image_hw is not
                    divisible by patch_size.
    """
    if method not in ('rollout', 'last'):
        raise ValueError(f"method must be 'rollout' or 'last', got {method!r}")
    if image_hw % patch_size != 0:
        raise ValueError(f"image_hw={image_hw} must be divisible by patch_size={patch_size}")

    # Grid dimensions from patchification: 32/4 = 8 patches per side -> 64 patches
    grid_hw = image_hw // patch_size

    # Index where patch tokens start (after special tokens)
    patch_start = 2 if has_dist_token else 1

    # Get attention matrix (B, seq_len, seq_len) in the chosen form
    if method == 'rollout':
        attn = attention_rollout(attn_maps_list)
    else:
        # Single layer: average heads to match rollout's shape
        attn = attn_maps_list[layer_idx].mean(dim=1)

    # CLS row: attention FROM [CLS] (position 0) TO patch tokens
    # Shape: (B, seq_len) -> (B, num_patches)
    cls_attn = attn[:, 0, patch_start:] # type: ignore

    # Normalize to [0, 1] per-sample for stable overlay alpha
    # (rollout rows sum to 1 but patch-only slice sums to less; per-sample
    # min-max makes visual intensity comparable across samples)
    cls_min = cls_attn.min(dim=-1, keepdim=True).values
    cls_max = cls_attn.max(dim=-1, keepdim=True).values
    cls_norm = (cls_attn - cls_min) / (cls_max - cls_min + 1e-8)

    # Reshape flat patch sequence -> 2D spatial grid: (B, num_patches) -> (B, 1, grid_h, grid_w)
    B = cls_norm.size(0)
    heatmap = cls_norm.reshape(B, 1, grid_hw, grid_hw)

    # Upsample to image resolution for overlay: (B, 1, grid, grid) -> (B, 1, H, W)
    # Bicubic gives smooth heatmaps; nearest-neighbor would show visible patch blocks
    heatmap = F.interpolate(heatmap, size=(image_hw, image_hw),
                            mode='bicubic', align_corners=False)

    # Clamp after bicubic (overshoots can produce values slightly outside [0, 1])
    heatmap = heatmap.clamp(0.0, 1.0).squeeze(1)

    return heatmap