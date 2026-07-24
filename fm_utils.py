import threading
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict, Union
import math
import os
import time
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
import wandb
import random
import json
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR, _LRScheduler


class FourierFeatureMapping(nn.Module):
    """
    Random Fourier Feature Mapping for coordinates.
    Maps coord_dim → 2*num_ff via [sin(2π·v·x), cos(2π·v·x)].
    v is a [num_ff, coord_dim] matrix of random frequencies, optionally trainable.
    """
    def __init__(self, num_ff, coord_dim, trainable=True):
        super().__init__()
        v = torch.randn(num_ff, coord_dim)
        if trainable:
            self.v = nn.Parameter(v)
        else:
            self.register_buffer('v', v)

    def forward(self, x):  # x: [..., coord_dim]
        vx = x @ self.v.T  # [..., num_ff]
        return torch.cat([torch.sin(2 * math.pi * vx),
                          torch.cos(2 * math.pi * vx)], dim=-1)  # [..., 2*num_ff]


class SetEncoder(nn.Module):
    """
    Encodes a sparse set of observations into a sequence of tokens and a pooled context vector,
    using a dedicated positional encoding for the coordinates.
    """

    def __init__(self, num_freqs, d_model, nhead, num_layers, coord_dim=3,
                 num_ff_coord=0, ffm_trainable=True, use_freq_ctx=False):
        super().__init__()
        self.d_model = d_model
        self.num_freqs = num_freqs
        self.use_freq_ctx = use_freq_ctx

        # An MLP for the "what": the ATF values
        self.value_tokenizer = nn.Sequential(
            nn.Linear(num_freqs, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Optional Fourier Feature Mapping for coordinates.
        # num_ff_coord=0 → legacy raw coord input (backward compatible)
        # num_ff_coord>0 → coords expanded to 2*num_ff_coord via sin/cos before MLP
        # A separate MLP for the "where": the coordinate conditioning vector.
        # coord_dim=3  → only relative mic position  (default, backward compatible)
        # coord_dim=9  → [rel_pos(3), d_walls(6)]  (geo_conditioning=True)
        if num_ff_coord > 0:
            self.coord_ffm = FourierFeatureMapping(num_ff_coord, coord_dim, trainable=ffm_trainable)
            # FFMv1: FFM already provides non-linearity via sin/cos, so one linear projection suffices
            self.positional_encoder = nn.Linear(2 * num_ff_coord, d_model)
        else:
            self.coord_ffm = None
            # Legacy: 2-layer MLP on raw coords (coord_dim=3 or 9)
            self.positional_encoder = nn.Sequential(
                nn.Linear(coord_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )

        # Transformer encoder remains the same
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.y_null_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Per-frequency context MLP: for each (mic, freq) pair, embed [coord_feats, freq_idx, atf_val] → d_model.
        # Masked mean over M → [B, F, d_model] frequency-specific conditioning for UNet FiLM.
        if use_freq_ctx:
            coord_feat_dim = 2 * num_ff_coord if num_ff_coord > 0 else coord_dim
            self.freq_ctx_mlp = nn.Sequential(
                nn.Linear(coord_feat_dim + 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
        else:
            self.freq_ctx_mlp = None

    def forward(self, obs_coords_rel, obs_values, obs_mask):
        """
        Args:
            obs_coords_rel (Tensor): Relative mic coordinates [B, M_max, 3]
            obs_values (Tensor): ATF magnitudes at those mics [B, M_max, 20]
            obs_mask (Tensor): Boolean mask indicating valid observations [B, M_max]
        Returns:
            encoded_tokens [B, M_max, d_model], pooled_context [B, d_model], freq_contexts [B, F, d_model] or None
        """
        # ### <<< CHANGE 2: Process values and positions separately

        # 1. Create value embeddings
        value_tokens = self.value_tokenizer(obs_values)

        # 2. Create positional embeddings (optionally via FFM first)
        coords = self.coord_ffm(obs_coords_rel) if self.coord_ffm is not None else obs_coords_rel
        positional_tokens = self.positional_encoder(coords)

        # 3. Add them together to get the final input tokens for the transformer
        tokens = value_tokens + positional_tokens

        # 4. Use transformer to let observations communicate with each other
        padding_mask = ~obs_mask
        encoded_tokens = self.transformer_encoder(tokens, src_key_padding_mask=padding_mask)

        # 5. Create the pooled context vector (this logic remains the same)
        masked_tokens = encoded_tokens.masked_fill(~obs_mask.unsqueeze(-1), 0.0)
        num_valid_tokens = obs_mask.sum(dim=1, keepdim=True)
        pooled_context = masked_tokens.sum(dim=1) / (num_valid_tokens + 1e-8)

        # 6. Optionally compute per-frequency contexts [B, F, d_model]
        freq_contexts = None
        if self.freq_ctx_mlp is not None:
            B, M_max, F = obs_values.shape
            freq_idx = torch.arange(F, device=obs_values.device, dtype=obs_values.dtype) / F  # [F]
            # coords already computed above: [B, M_max, coord_feat_dim]
            coords_exp = coords.unsqueeze(2).expand(B, M_max, F, -1)       # [B, M, F, coord_feat_dim]
            freq_exp   = freq_idx.view(1, 1, F, 1).expand(B, M_max, F, 1)  # [B, M, F, 1]
            atf_exp    = obs_values.unsqueeze(-1)                           # [B, M, F, 1]
            token_in   = torch.cat([coords_exp, freq_exp, atf_exp], dim=-1) # [B, M, F, coord_feat_dim+2]
            freq_tokens = self.freq_ctx_mlp(token_in)                       # [B, M, F, d_model]
            # Masked mean over M
            mask_exp = obs_mask.unsqueeze(-1).unsqueeze(-1)                 # [B, M, 1, 1]
            freq_tokens = freq_tokens.masked_fill(~mask_exp, 0.0)
            n_valid = obs_mask.sum(dim=1).view(B, 1, 1)                     # [B, 1, 1]
            freq_contexts = freq_tokens.sum(dim=1) / (n_valid + 1e-8)      # [B, F, d_model]

        return encoded_tokens, pooled_context, freq_contexts


def get_dataset_version_from_model_name(model_name: str) -> str:
    """
    Determine dataset version based on model name.

    Args:
        model_name: The model name string

    Returns:
        Dataset version string ('r1', 'r2', 'r3', 'r4')
    """
    if model_name is None:
        return 'r1'

    model_name_upper = model_name.upper()

    if 'SMOKE' in model_name_upper:
        return 'smoke'
    elif 'BIG_8192R4' in model_name_upper:
        return 'r4'
    elif 'BIG8192' in model_name_upper:
        return 'r3'
    elif 'BIGDATA' in model_name_upper:
        return 'r2'
    else:
        return 'r1'


def get_src_splits_for_version(dataset_version: str) -> dict:
    """
    Get the source splits configuration for a given dataset version.

    Args:
        dataset_version: Dataset version ('r1', 'r2', 'r3', 'r4')

    Returns:
        Dictionary with src_splits configuration
    """
    dataset_configs = {
        # Small split (102 sources) for quickly verifying an install end-to-end.
        # Generate with: python generate_dataset.py -s 102
        'smoke': {"train": [0, 80], "valid": [80, 90], "test": [90, 102]},
        'r1': {"train": [0, 820], "valid": [820, 922], "test": [922, 1024]},
        'r1val52': {"train": [0, 820], "valid": [820, 872], "test": [922, 1024]},  # 52 val sources (multiple of batch_size=4) for faster val LSD
        'r2': {"train": [[0, 820], [1024, 4096]], "valid": [820, 922], "test": [922, 1024]},
        'r3': {"train": [[0, 820], [1024, 8192]], "valid": [820, 922], "test": [922, 1024]},
        'r4': {"train": [[0, 820], [1324, 8192]], "valid": [[820, 922], [1024, 1324]], "test": [922, 1024]}
    }

    if dataset_version not in dataset_configs:
        print(f"Warning: Unknown dataset version '{dataset_version}', defaulting to 'r1'")
        dataset_version = 'r20'

    return dataset_configs[dataset_version]


def parse_source_indices(src_splits_config, mode: str) -> List[int]:
    """
    Parse source indices from src_splits configuration.
    Supports both old format [start, end] and new format [[start1, end1], [start2, end2], ...]

    Args:
        src_splits_config: Dictionary containing src_splits configuration
        mode: Mode string ('train', 'valid', 'test')

    Returns:
        List of source indices
    """
    split_config = src_splits_config[mode]

    # Check if it's the new format (list of lists) or old format (single list)
    if isinstance(split_config[0], list):
        # New format: [[start1, end1], [start2, end2], ...]
        indices = []
        for start, end in split_config:
            indices.extend(range(start, end))
        return indices
    else:
        # Old format: [start, end]
        return list(range(split_config[0], split_config[1]))


def model_factory(config, device):
    """
    Reads the config and returns the correctly instantiated and loaded models.
    """
    model_cfg = config['model']
    # Use the presence of the version key to decide which architecture to build
    architecture = model_cfg.get('architecture_version')
    setversion = model_cfg.get('setencoder_version')

    # --- Compute actual channel count from the freq subband ---
    # freq_from defaults to 0 for backward compatibility (full 0..freq_up_to range)
    freq_from = model_cfg.get('freq_from', 0)
    num_freqs = model_cfg['freq_up_to'] - freq_from

    # --- Instantiate models based on version ---
    freq_ctx = model_cfg.get('freq_ctx', False)
    init_kernel_size = model_cfg.get('init_kernel_size', 3)

    if setversion == "v3":
        print("--- Creating set encoder v3 ---")
        coord_dim = model_cfg.get('coord_dim', 3)
        set_encoder = SetEncoder(
            num_freqs=num_freqs,
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            num_layers=model_cfg['num_encoder_layers'],
            coord_dim=coord_dim,
            use_freq_ctx=freq_ctx
        ).to(device)

    elif setversion == "v12" or setversion is None:
        print("--- Creating set encoder v12 ---")
        coord_dim = model_cfg.get('coord_dim', 3)
        num_ff_coord = model_cfg.get('num_ff_coord', 0)
        set_encoder = SetEncoder_v12(
            num_freqs=num_freqs,
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            num_layers=model_cfg['num_encoder_layers'],
            coord_dim=coord_dim,
            num_ff_coord=num_ff_coord,
            use_freq_ctx=freq_ctx
        ).to(device)

    freq_channel_bias = model_cfg.get('freq_channel_bias', False)
    freq_film = model_cfg.get('freq_film', False)

    if architecture == "v2_residual_context":
        print("--- Creating (v2) architecture ---")
        unet_3d = CrossAttentionUNet3D_RED3d(
            in_channels=num_freqs,
            out_channels=num_freqs,
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            init_kernel_size=init_kernel_size,
            freq_channel_bias=freq_channel_bias,
            freq_film=freq_film,
            freq_ctx=freq_ctx
        ).to(device)

    elif architecture == "v1_legacy" or architecture is None:
        print("--- Creating v1 architecture: standard 3d unet ---")
        # Instantiate the old U-Net and ODE wrapper for old checkpoints
        unet_3d = CrossAttentionUNet3D(
            in_channels=num_freqs,
            out_channels=num_freqs,
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            init_kernel_size=init_kernel_size,
            freq_channel_bias=freq_channel_bias,
            freq_film=freq_film,
            freq_ctx=freq_ctx
        ).to(device)

    elif architecture == "v4_DiT":
        # Instantiate the old U-Net and ODE wrapper for old checkpoints
        unet_3d = DiffusionTransformer3D(
            in_channels=num_freqs,
            out_channels=num_freqs,
            patch_size=model_cfg['patch_size'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)

    elif architecture == "v3_attention":
        print("--- Creating (v3) architecture with Self-Attention ---")
        unet_3d = CrossAttentionUNet3D_v3(
            in_channels=num_freqs,
            out_channels=num_freqs,
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            input_size=11,  # Assuming your cube dimension is 11
            init_kernel_size=init_kernel_size,
            freq_channel_bias=freq_channel_bias,
            freq_film=freq_film,
            freq_ctx=freq_ctx
        ).to(device)

    return set_encoder, unet_3d


class ThreePhaseScheduler(_LRScheduler):
    """
    A custom LR scheduler that implements a three-phase schedule:
    1. Linear warm-up from a start_factor to 1.0.
    2. Cosine annealing decay from the peak LR down to a minimum LR.
    3. Constant "coast" phase at the minimum LR.
    """

    def __init__(self, optimizer, total_iterations, warmup_iterations, decay_iterations,
                 peak_lr, min_lr, start_factor=0.01, last_epoch=-1):
        self.total_iters = total_iterations
        self.warmup_iters = warmup_iterations
        self.decay_iters = decay_iterations
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.start_factor = start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Phase 1: Linear Warm-up
            start_lr = self.peak_lr * self.start_factor
            progress = self.last_epoch / self.warmup_iters
            return [start_lr + (self.peak_lr - start_lr) * progress]

        elif self.last_epoch < self.decay_iters:
            # Phase 2: Cosine Decay
            progress = (self.last_epoch - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
            cos_out = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (self.peak_lr - self.min_lr) * cos_out]

        else:
            # Phase 3: Coast at min_lr
            return [self.min_lr]


# early stopping taken from: https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/utils.py#L72

class EarlyStopping(object):
    """Early Stopping Monitor"""

    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):

        metrics_val = metrics.cpu().item()
        if self.best is None:
            self.best = metrics_val
            return False

        if np.isnan(metrics_val):
            return True

        if self.is_better(metrics_val, self.best):
            self.num_bad_epochs = 0
            self.best = metrics_val
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta


class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """

    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1))  # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None


class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """

    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, c, h, w)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples)  # (num_samples, c, h, w)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t)  # (num_samples, c, h, w)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """
        pass


class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)


class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - beta_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt beta_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)


class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        return torch.ones_like(t)


class LinearBeta(Beta):
    """
    Implements beta_t = 1-t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        return - torch.ones_like(t)


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        alpha_t = self.alpha(t)  # (num_samples, 1, 1, 1)
        beta_t = self.beta(t)  # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t)  # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t)  # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - diffusion_coefficient: shape (bs, c, h, w)
        """
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
            - dt: time, shape (bs, 1, 1, 1)
        Returns:
            - nxt: state at time t + dt (bs, c, h, w)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w)
        """
        silent = kwargs.pop('silent', False)
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1), disable=silent):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    # def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs): #PREVIOUSLY
    #     return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        # Get the model's output (the "drift"), which has 20 channels
        drift = self.ode.drift_coefficient(xt, t, **kwargs)

        # Case 1: Standard models where input and output shapes match.
        if xt.shape == drift.shape:
            # If shapes match, apply the Euler update to all channels
            x_next = xt + drift * h

        # Case 2: Inpainting models where the input `xt` has one extra channel (the mask).
        # for ATF 2D inpainting model.
        # Separate the current state `xt` into its data and mask components
        elif xt.shape[1] == drift.shape[1] + 1:
            print("2D inpaint: Applying Euler update to data channels only, preserving the mask channel.")
            xt_data = xt[:, :-1]  # The first 20 channels (frequencies)
            xt_mask = xt[:, -1:]  # The last channel (the mask)

            # Apply the Euler update ONLY to the data channels
            updated_xt_data = xt_data + drift * h

            # Re-combine the updated data with the original, unchanged mask
            x_next = torch.cat([updated_xt_data, xt_mask], dim=1)

        else:  # Case 3: The shapes are incompatible.
            raise ValueError(
                f"Incompatible shapes for Euler. `xt` is {xt.shape} "
                f"output `drift` is {drift.shape}."
            )

        # --- NEW: Optional Data Consistency ("Pasting") Step ---
        if kwargs.get('paste_observations'):
            # print("Pasting known observations into the state after the Euler step.")
            z_true = kwargs.get('z_true')
            x0 = kwargs.get('x0')
            obs_indices = kwargs.get('obs_indices')

            if z_true is not None and x0 is not None:
                # Create the mask for pasting
                full_mask_3d = torch.zeros_like(z_true)
                # Flatten view to use flat indices
                full_mask_flat = full_mask_3d.view(1, -1)
                full_mask_flat[:, obs_indices] = 1

                paste_mask = full_mask_flat.view(*z_true.shape)
                # Get the correct value for the known data on the straight noise-to-data path
                t_next = t + h
                known_path_slice = (1 - t_next) * x0 + t_next * z_true

                # Replace the values at the M known locations
                x_next = x_next * (1 - paste_mask) + known_path_slice * paste_mask
            else:
                assert False, "For pasting, z_true and x0 must be provided in kwargs."

        return x_next

    @torch.no_grad()
    def simulate_trajectory(self, x: torch.Tensor, max_timesteps: int, y: torch.Tensor):
        """
        Simulates the ODE and returns the state at each timestep.
        """
        ts = torch.linspace(0, 1, max_timesteps + 1).to(x.device)
        trajectory = [x.clone()]

        for i in range(max_timesteps):
            t_current = ts[i]
            t_next = ts[i + 1]
            h = t_next - t_current

            # Reshape t for drift_coefficient
            t_reshaped = t_current.view(1, 1, 1, 1).expand(x.shape[0], -1, -1, -1)

            x = self.step(x, t_reshaped, h, y=y)
            trajectory.append(x.clone())

        return torch.stack(trajectory)


class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        # print("diffusion coefficient: ", self.sde.diffusion_coefficient(xt, t, **kwargs) )
        return xt + self.sde.drift_coefficient(xt, t, **kwargs) * h + self.sde.diffusion_coefficient(xt, t,
                                                                                                     **kwargs) * torch.sqrt(
            h) * torch.randn_like(xt)


def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )


MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


def get_model_info(model: nn.Module, model_name: str = "Model") -> dict:
    """
    Get comprehensive model information including parameter counts and memory usage.

    Args:
        model: PyTorch model
        model_name: Name for display purposes

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Calculate model size in MB using the existing function
    model_size_bytes = model_size_b(model)
    model_size_mb = model_size_bytes / (1024 ** 2)

    info = {
        'name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_bytes': model_size_bytes,
        'model_size_mb': model_size_mb,
        'total_params_str': f"{total_params:,}",
        'trainable_params_str': f"{trainable_params:,}",
        'model_size_str': f"{model_size_mb:.2f} MB"
    }

    return info


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """Print formatted model information."""
    info = get_model_info(model, model_name)

    print(f"=== {info['name']} Information ===")
    print(f"Total parameters: {info['total_params_str']}")
    print(f"Trainable parameters: {info['trainable_params_str']}")
    if info['non_trainable_params'] > 0:
        print(f"Non-trainable parameters: {info['non_trainable_params']:,}")
    print(f"Model size: {info['model_size_str']}")
    print("=" * (len(info['name']) + 18))


class Trainer(ABC):
    def __init__(self, models: Dict[str, nn.Module]):
        super().__init__()
        if not isinstance(models, dict) or not models:
            raise ValueError("`models` must be a non-empty dictionary of nn.Modules.")

        self.models = models
        # For convenience, self.model can point to the primary model
        self.model = next(iter(self.models.values()))
        self.optimizer = None

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    # **NEW: Abstract method for validation loss**
    @torch.no_grad()
    @abstractmethod
    def get_valid_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        # Collect parameters from ALL models provided
        all_params = []
        for model in self.models.values():
            all_params.extend(list(model.parameters()))
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        return self.optimizer

    def train(self, num_iterations: int, device: torch.device, lr: float,
              warmup_iterations: Optional[int] = None,
              decay_iterations: Optional[int] = None,
              min_lr: Optional[float] = None,
              valid_sampler: Optional[Sampleable] = None,
              save_path: str = "model.pt",
              checkpoint_path: str = "checkpoints",
              validation_interval: Optional[int] = None,
              lsd_validation_interval: Optional[int] = None,
              checkpoint_interval: Optional[int] = None,
              start_iteration: int = 0,
              config: dict = None,
              early_stopping_patience: int = 1400,  # was 1000
              resume_checkpoint_path: Optional[str] = None,
              resume_checkpoint_state: Optional[dict] = None,
              save_start_iter: int = 0,
              **kwargs):

        print("--- Model(s) Summary ---")
        for name, model in self.models.items():
            print(f"  - {name}: {model_size_b(model) / MiB:.3f} MiB")
            model.to(device)

        # Start
        opt = self.get_optimizer(lr)

        if decay_iterations is None or decay_iterations <= warmup_iterations:
            decay_iterations = num_iterations
            print(
                f"Using 2-Phase LR schedule: Warm-up for {warmup_iterations} iters, then Cosine Annealing for {num_iterations - warmup_iterations} iters.")
        else:
            print(
                f"Using 3-Phase LR schedule: Warm-up ({warmup_iterations} iters) -> Cosine Decay ({decay_iterations - warmup_iterations} iters) -> Coast ({num_iterations - decay_iterations} iters).")

        # --- MODIFIED: Create the new 3-Phase Learning Rate Scheduler ---
        if warmup_iterations > 0 and decay_iterations is not None:
            # If decay_iterations is not specified, default to the old behavior (decay over all iterations)
            scheduler = ThreePhaseScheduler(
                optimizer=opt,
                total_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
                decay_iterations=decay_iterations,
                peak_lr=lr,
                min_lr=min_lr
            )

        else:
            # Fallback to the original scheduler if no warm-up is specified
            print("Using LR schedule: Cosine Annealing without warm-up.")
            scheduler = CosineAnnealingLR(opt, T_max=num_iterations, eta_min=min_lr)

        # NEW: Initialize the EarlyStopping monitor (tracks LSD — the real metric)
        early_stopper = EarlyStopping(patience=early_stopping_patience)
        # --- State Tracking ---
        best_val_loss = float("inf")  # FM-MSE (kept for reference)
        best_val_lsd  = float("inf")  # LSD in dB (primary save criterion)
        best_iteration = start_iteration

        # --- Timing Tracking ---
        training_start_time = time.time()
        best_val_loss_time = None
        _total_val_time = 0.0  # accumulated validation wall-clock time (excluded from per-epoch estimate)

        # Unified resume logic: load from an explicit checkpoint path/state if provided
        # checkpoint = None
        # if resume_checkpoint_state is not None:
        #     checkpoint = resume_checkpoint_state
        #     print("Resuming from provided in-memory checkpoint state")
        # elif resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
        #     print(f"Loading checkpoint from {resume_checkpoint_path}")
        #     checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        #
        # if checkpoint is not None:
        #     # Restore model weights if present
        #     model_state = checkpoint.get('model_state_dict')
        #     if model_state is not None:
        #         self.model.load_state_dict(model_state)
        #
        #     # Restore trainer-specific state if present (e.g., y_null)
        #     if hasattr(self, 'y_null') and checkpoint.get('y_null') is not None:
        #         self.y_null.data = checkpoint['y_null'].to(device)
        #
        #     # Restore optimizer if present
        #     if checkpoint.get('optimizer_state_dict') is not None:
        #         opt.load_state_dict(checkpoint['optimizer_state_dict'])
        #         print("Optimizer state restored from checkpoint.")
        #
        #         print(f"Overwriting optimizer LR with new command-line value: {lr}")
        #         for param_group in opt.param_groups:
        #             param_group['lr'] = lr
        #
        #     # Adopt iteration and best metrics if available
        #     iter_value = checkpoint.get('iteration', None)
        #     if isinstance(iter_value, (int, float)):
        #         start_iteration = int(iter_value)
        #         print("starting from iteration", start_iteration)
        #     else:
        #         start_iteration = checkpoint["config"]["training"].get("num_iterations") + 1
        #         print("starting from iteration", start_iteration)
        #     if start_iteration is None or not isinstance(start_iteration, int) or start_iteration <= 0:
        #         assert start_iteration >= 0, "start_iteration must be a non-negative integer"
        #
        #     best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        #     best_iteration = checkpoint.get('best_iteration', best_iteration)
        #     print(f"Resumed state. start_iteration={start_iteration}, best_val_loss={best_val_loss}, best_iteration={best_iteration}")

        # if resume_checkpoint_state:
        #     print("Resuming from provided in-memory checkpoint state")
        #
        # --- Resume: restore model/optimizer/scheduler state if a checkpoint was provided ---
        if resume_checkpoint_state is not None:
            print("--- Restoring checkpoint state ---")

            # 1. Model weights (supports new multi-model dict and legacy single-model key)
            if 'model_states' in resume_checkpoint_state:
                for key, state_dict in resume_checkpoint_state['model_states'].items():
                    if key in self.models:
                        self.models[key].load_state_dict(state_dict)
                        print(f"  - '{key}' weights restored")
            elif 'model_state_dict' in resume_checkpoint_state:
                self.model.load_state_dict(resume_checkpoint_state['model_state_dict'])
                print("  - model weights restored (legacy key)")

            # 2. Optimizer (momentum, second moments, etc.)
            if 'optimizer_state_dict' in resume_checkpoint_state:
                opt.load_state_dict(resume_checkpoint_state['optimizer_state_dict'])
                print("  - optimizer state restored")

            # 3. LR scheduler position — critical for correct LR at resumed step.
            #    After setting last_epoch = N-1, the next scheduler.step() inside the
            #    loop computes the LR for step N, which is exactly correct.
            start_iteration = resume_checkpoint_state.get('iteration', start_iteration)
            scheduler.last_epoch = start_iteration - 1
            print(f"  - scheduler.last_epoch set to {start_iteration - 1}")

            # 4. Best metric tracking (so new best-model saves are compared correctly)
            best_val_loss = resume_checkpoint_state.get('best_val_loss', best_val_loss)
            best_val_lsd  = resume_checkpoint_state.get('best_val_lsd',  best_val_lsd)
            best_iteration = resume_checkpoint_state.get('best_iteration', best_iteration)
            print(f"  - start_iteration={start_iteration}, best_val_lsd={best_val_lsd:.4f} dB at iter {best_iteration}")

            # 5. NOTE: RNG state is NOT saved or restored because torch.manual_seed(42 + iteration)
            #    seeds each training step independently. Resuming from step N automatically gives
            #    bit-identical draws for every subsequent step — no global RNG state needed.
            print("--- Checkpoint restored. Continuing training... ---\n")


        # --- TRAINING LOOP ---
        if lsd_validation_interval is None:
            lsd_validation_interval = validation_interval
        batch_size = kwargs.get('batch_size')
        # dataset_size = len(self.path.p_data.spectrograms)
        dataset_size = len(self.path.p_data)
        experiment_dir = os.path.dirname(save_path)
        _config_path = os.path.join(experiment_dir, 'config.json')
        _log_path = os.path.join(experiment_dir, 'log.txt')

        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            if hours > 0:
                return f"{hours}h {minutes}m {secs:.1f}s"
            elif minutes > 0:
                return f"{minutes}m {secs:.1f}s"
            else:
                return f"{secs:.1f}s"

        for model in self.models.values():
            model.train()

        pbar = tqdm(range(start_iteration, num_iterations))
        for iteration in pbar:
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = self.get_train_loss(iteration=iteration, **kwargs)
            _t_bwd0 = time.time(); torch.cuda.synchronize() if iteration < 0 else None
            loss.backward()
            torch.cuda.synchronize() if iteration < 0 else None
            if iteration < 0: print(f"  [iter {iteration}] backward={time.time()-_t_bwd0:.3f}s")
            opt.step()
            scheduler.step()

            # Calculate and display the current epoch number**
            current_lr = scheduler.get_last_lr()[0]
            current_epoch = (iteration + 1) * batch_size / dataset_size
            if iteration % 100 == 0:
                loss_float = loss.item()  # single GPU sync every 100 iters
                if wandb.run is not None:
                    wandb.log({"train_loss": loss_float, "epoch": current_epoch, "iteration": iteration,
                               "learning_rate": current_lr}, step=iteration)
                pbar.set_description(f'Epoch: {current_epoch:.2f}, Iter: {iteration}, Loss: {loss_float:.5f}')

            # **NEW: Validation loop**
            _do_val = valid_sampler and (iteration + 1) % validation_interval == 0
            _do_lsd = valid_sampler and (iteration + 1) % lsd_validation_interval == 0
            if _do_val or _do_lsd:
                _val_start = time.time()
                for model in self.models.values():
                    model.eval()

                # Save RNG state: validation uses torch.manual_seed() internally;
                # restoring afterwards ensures training randomness is unaffected.
                _cpu_rng = torch.get_rng_state()
                _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                val_loss, val_lsd = self.get_valid_loss(valid_sampler=valid_sampler,
                                                        compute_lsd=_do_lsd, **kwargs)

                # Restore training RNG state
                torch.set_rng_state(_cpu_rng)
                if _cuda_rng is not None:
                    torch.cuda.set_rng_state_all(_cuda_rng)

                _lsd_str = f'{val_lsd.item():.4f} dB' if val_lsd is not None else '(skipped)'
                if wandb.run is not None:
                    log_dict = {"val_loss": val_loss.item(), "epoch": current_epoch, "iteration": iteration}
                    if val_lsd is not None:
                        log_dict["val_lsd"] = val_lsd.item()
                    wandb.log(log_dict, step=iteration)
                val_desc = (f'Epoch: {current_epoch:.4f}, Iter: {iteration}, '
                            f'Loss: {loss.item():.5f}, Val MSE: {val_loss.item():.5f}, Val LSD: {_lsd_str}')
                pbar.set_description(val_desc)
                if (iteration + 1) % 1000 == 0:
                    _elapsed = time.time() - training_start_time
                    _its = (iteration + 1) / _elapsed if _elapsed > 0 else 0
                    print(f"{val_desc} | elapsed: {format_time(_elapsed)}, avg: {_its:.2f} iter/s", flush=True)

                # Always update FM-MSE tracker (reference only)
                _new_best_fm = val_loss.item() < best_val_loss
                best_val_loss = min(best_val_loss, val_loss.item())

                # ── Primary criterion: LSD when available, FM-MSE as fallback ──
                _lsd_improved = val_lsd is not None and val_lsd < best_val_lsd
                _fm_fallback  = val_lsd is None and _new_best_fm
                if _lsd_improved or _fm_fallback:
                    if _lsd_improved:
                        best_val_lsd = val_lsd
                    best_val_loss_time = time.time()
                    _elapsed = best_val_loss_time - training_start_time
                    _metric_str = (f"New best LSD: {best_val_lsd:.4f} dB" if _lsd_improved
                                   else f"New best FM-Loss: {best_val_loss:.5f} (LSD skipped)")
                    _best_msg = (f"** [Iter {iteration} | Epoch {current_epoch:.1f}] "
                                 f"{_metric_str} | "
                                 f"Elapsed: {format_time(_elapsed)} **")
                    print(_best_msg)
                    with open(_log_path, 'a') as _lf:
                        _lf.write(_best_msg + '\n')
                        # f"(FM-MSE: {val_loss.item():.5f}, train loss: {loss.item():.5f}). ")

                    y_null_to_save = None
                    if hasattr(self, 'set_encoder') and self.set_encoder is not None:
                        y_null_to_save = self.set_encoder.y_null_token
                    elif hasattr(self, 'y_null'):
                        y_null_to_save = self.y_null

                    best_model_state = {
                        'model_states': {key: model.state_dict() for key, model in self.models.items()},
                        'optimizer_state_dict': opt.state_dict(),
                        'iteration': iteration + 1,
                        'best_val_loss': best_val_loss,   # FM-MSE for reference
                        'best_val_lsd':  best_val_lsd,    # LSD — primary metric
                        'best_iteration': iteration,
                        'config': config,
                        'wandb_run_id': config.get('wandb_run_id'),
                        'y_null_token': y_null_to_save,
                        'is_best': True
                    }

                    if iteration >= save_start_iter:
                        torch.save(best_model_state, save_path)
                    best_iteration = iteration

                    # Update config.json with latest best metrics
                    config['best_val_lsd']   = float(best_val_lsd)
                    config['best_val_loss']  = float(best_val_loss)
                    config['best_iteration'] = best_iteration
                    os.makedirs(os.path.dirname(_config_path), exist_ok=True)  # guard against CephFS eviction
                    with open(_config_path, 'w') as _f:
                        json.dump(config, _f, indent=4)

                # Re-enable training mode after validation
                for model in self.models.values():
                    model.train()
                _total_val_time += time.time() - _val_start

                # Early stopping tracks LSD (only when LSD was computed)
                if val_lsd is not None and early_stopper.step(val_lsd):
                    print(f"--- Early stopping triggered at iteration {iteration} with val LSD: {val_lsd:.4f} dB ---")
                    flag_save = True
                    break

            else:
                pbar.set_description(f'Epoch: {current_epoch:.2f}, Iter: {iteration}')

            # --- Periodic Checkpointing Logic ---
            if (iteration + 1) % checkpoint_interval == 0:
                print(f"\n--- Saving checkpoint at iteration {iteration + 1} ---")
                ckpt_save_path = os.path.join(checkpoint_path, f"ckpt_{iteration + 1}.pt")

                y_null_to_save = getattr(self.models.get('set_encoder'), 'y_null_token', getattr(self, 'y_null', None))

                # Save checkpoint for resuming training (latest state)
                checkpoint_state = {
                    'iteration': iteration + 1,
                    'model_states': {key: model.state_dict() for key, model in self.models.items()},
                    # 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_iteration': best_iteration,
                    'config': config,
                    'wandb_run_id': config.get('wandb_run_id'),
                    'is_best': False,  # Flag to indicate this is latest checkpoint
                    'y_null_token': y_null_to_save,
                }
                torch.save(checkpoint_state, ckpt_save_path)

        # --- Save final checkpoint ---
        final_iteration = iteration + 1
        if final_iteration == num_iterations or flag_save:
            final_ckpt_path = os.path.join(checkpoint_path, f"ckpt_final_{final_iteration}.pt")
            print(f"\n--- Saving final checkpoint at iteration {final_iteration} to {final_ckpt_path} ---")

            y_null_to_save = getattr(self.models.get('set_encoder'), 'y_null_token', getattr(self, 'y_null', None))

            final_checkpoint_state = {
                'iteration': final_iteration,
                # 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_lsd':  best_val_lsd,
                'best_iteration': best_iteration,
                'config': config,
                'wandb_run_id': config.get('wandb_run_id'),
                'is_best': False,
                'is_final': True,  # Flag to indicate this is the final state
                'model_states': {key: model.state_dict() for key, model in self.models.items()},
                'y_null_token': y_null_to_save
            }
            torch.save(final_checkpoint_state, final_ckpt_path)

        self.model.eval()

        # --- Calculate and Print Timing Information ---
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        # Calculate time to best validation loss
        if best_val_loss_time is not None:
            time_to_best_val_loss = best_val_loss_time - training_start_time
        else:
            time_to_best_val_loss = total_training_time

        # Calculate time per epoch — subtract validation time so we measure pure training speed
        total_epochs = (iteration + 1) * batch_size / dataset_size
        pure_train_time = total_training_time - _total_val_time
        time_per_epoch = pure_train_time / total_epochs if total_epochs > 0 else 0
        # Real wall-clock epoch time (train + validation + overhead)
        time_per_epoch_real = total_training_time / total_epochs if total_epochs > 0 else 0

        # Rename model.pt -> model_{iter}_{metric}_{val}.pt now that training is done
        if os.path.exists(save_path):
            if best_val_lsd < float("inf"):
                named_path = os.path.join(experiment_dir, f"model_{best_iteration}_lsd{float(best_val_lsd):.4f}.pt")
            elif best_val_loss < float("inf"):
                named_path = os.path.join(experiment_dir, f"model_{best_iteration}_fmMSE{float(best_val_loss):.5f}.pt")
            else:
                named_path = None

            if named_path:
                os.rename(save_path, named_path)
                config['best_model_path'] = named_path
                with open(_config_path, 'w') as _f:
                    json.dump(config, _f, indent=4)
                print(f"Best model saved as: {os.path.basename(named_path)}")

        summary_lines = [
            f"--- Training finished. Best LSD: {best_val_lsd:.4f} dB | Best FM-MSE: {best_val_loss:.5f} | at iteration {best_iteration} | at epoch {current_epoch} |. ---",
            f"--- TIMING SUMMARY ---",
            f"Total wall-clock time:       {format_time(total_training_time)}",
            f"  of which validation:       {format_time(_total_val_time)}",
            f"  pure training time:        {format_time(pure_train_time)}",
            f"Time to reach best LSD:      {format_time(time_to_best_val_loss)}",
            f"Avg time per epoch (train):  {format_time(time_per_epoch)}",
            f"Avg time per epoch (TOTAL real): {time_per_epoch_real:.2f} sec",

            f"Total epochs completed:      {total_epochs:.2f}",
            f"--- END TIMING SUMMARY ---",
        ]
        for line in summary_lines:
            print(line)
        with open(_log_path, 'a') as _lf:
            _lf.write('\n'.join(summary_lines) + '\n')
        print("--- END TIMING SUMMARY ---")

        # Run evaluate.py on the best model and append output to log.txt
        _best_model_path = config.get('best_model_path') if config else None
        if _best_model_path and os.path.exists(_best_model_path):
            _data_dir = (config.get('data', {}).get('data_dir', '') if config else '').rstrip('/')
            _eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate.py')
            import subprocess, sys
            _eval_cmd = [sys.executable, _eval_script,
                         '--model_path', _best_model_path,
                         '--data_dir', _data_dir]
            print(f"\n--- Running evaluate.py on best model ---")
            print(f"CMD: {' '.join(_eval_cmd)}\n")
            with open(_log_path, 'a') as _lf:
                _lf.write(f"\n--- evaluate.py on best model ---\n")
                _lf.write(f"CMD: {' '.join(_eval_cmd)}\n")
            with subprocess.Popen(_eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as _proc:
                with open(_log_path, 'a') as _lf:
                    for _line in _proc.stdout:
                        print(_line, end='', flush=True)
                        _lf.write(_line)
            print(f"--- evaluate.py finished ---")
        else:
            print("Skipping post-training evaluation (no best model path found).")

        return best_val_lsd


class ATF3DSampler(torch.nn.Module, Sampleable):
    """
        Loads and serves full 3D ATF magnitude cubes.
        Each sample is a tensor of shape [64, 11, 11, 11] (freq, Z, Y, X).
        """

    def __init__(self, data_path: str, mode: str, src_splits: dict, freq_up_to: int, freq_from: int = 0,
                 normalize: bool = True, model_name: str = None):
        super().__init__()
        self.mode = mode
        self.src_splits = src_splits
        self.normalize = normalize
        self.mean = None
        self.std = None
        self.freq_up_to = freq_up_to
        self.freq_from = freq_from

        # Determine dataset version from model name
        self.dataset_version = get_dataset_version_from_model_name(model_name)
        print(f"Using dataset version: {self.dataset_version} (from model_name: {model_name})")

        # Cache filename encodes the subband [freq_from, freq_up_to) so different subbands
        # don't clobber each other.  When freq_from=0 the name matches the old convention
        # (except the '0to' prefix), so existing caches are NOT reused — re-generation is safe
        # because the slice is now explicit.
        freq_tag = f"{self.freq_from}to{self.freq_up_to}" if self.freq_from != 0 else f"{self.freq_up_to}"
        processed_file = os.path.join(data_path,
                                      f'processed_atf3d_{self.mode}_freqs{freq_tag}_{self.dataset_version}.pt')

        # Check if preprocessed file exists and matches config
        print(f"Looking for pre-processed file: {processed_file}")
        recreate_file = False
        if os.path.exists(processed_file):
            print(f"Loading pre-processed ATF-3D {self.mode} data from {processed_file}")
            data = torch.load(processed_file)

            # Check if config matches actual data
            if 'sample_info' in data:
                actual_source_ids = data['sample_info'].flatten().tolist()
                expected_source_ids = parse_source_indices(src_splits, self.mode)

                if sorted(actual_source_ids) != sorted(expected_source_ids):
                    print(f"⚠️  CONFIG MISMATCH DETECTED for {self.mode} data!")
                    print(f"   Config expects: {len(expected_source_ids)} sources")
                    print(f"   Preprocessed file contains: {len(actual_source_ids)} sources")
                    print(f"   Recreating preprocessed file to match config...")
                    recreate_file = True
                else:
                    # Data matches config, load it
                    self.cubes = data['cubes']
                    self.source_coords = data['source_coords']
                    self.grid_xyz = data['grid_xyz']

                    # FIX: Assign the tensor to self so valid_sampler.sample_info exists
                    self.sample_info = data['sample_info']

                    if self.normalize:
                        self.mean = data.get('mean')
                        self.std = data.get('std')
                    recreate_file = False
                    print("no need to recreate preprocessing files")
            else:
                print(f"   Warning: No sample_info in preprocessed file, recreating...")
                recreate_file = True
        else:
            recreate_file = True

        if recreate_file:
            self._create_preprocessed_data(data_path, src_splits, processed_file)

        self.dummy = torch.nn.Buffer(torch.zeros(1))
        print(f"Loaded {len(self.cubes)} ATF-3D cubes for {self.mode} set.")
        print(f"Cube tensor shape: {self.cubes.shape}")

    def _create_preprocessed_data(self, data_path, src_splits, processed_file):
        """
        Create preprocessed data from NPZ files according to the config.
        """
        print(f"Processing ATF-3D {self.mode} data from .npz files...")
        source_indices = parse_source_indices(src_splits, self.mode)
        all_cubes = []
        all_source_coords = []
        all_sample_info = []

        # --- Grid construction (assuming it's constant for all sources) ---
        # --- Colleague's Fix 1: Establish a canonical grid order and permutation ---
        first_npz_file = os.path.join(data_path, f"data_s{source_indices[0] + 1:04d}.npz")
        with np.load(first_npz_file) as data:
            mic_pos = data['posMic']  # shape (1331, 3)
            x, y, z = mic_pos[:, 0], mic_pos[:, 1], mic_pos[:, 2]

            # Sort by z, then y, then x to get a canonical C-style row-major order
            perm = np.lexsort((x, y, z))

            unique_x, unique_y, unique_z = sorted(np.unique(x)), sorted(np.unique(y)), sorted(np.unique(z))
            self.nx, self.ny, self.nz = len(unique_x), len(unique_y), len(unique_z)

        # Create the canonical grid that matches the flattened order
        zz, yy, xx = torch.meshgrid(
            torch.tensor(unique_z, dtype=torch.float32),
            torch.tensor(unique_y, dtype=torch.float32),
            torch.tensor(unique_x, dtype=torch.float32),
            indexing='ij'
        )
        self.grid_xyz = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

        for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
            npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")
            if not os.path.exists(npz_file): continue

            with np.load(npz_file) as data_single:
                atf_mag_algn = data_single['atf_mag_algn']  # (1331, 64)
                np_of_mics, np_of_freqs = atf_mag_algn.shape
                source_pos = data_single['posSrc']  # (3,)

                # Reorder rows into the canonical layout using the permutation
                atf_perm = torch.tensor(atf_mag_algn[perm], dtype=torch.float32)  # [1331, 64]

                # Reshape the ordered data into the 3D cube
                full_cube = atf_perm.T.contiguous().view(np_of_freqs, self.nz, self.ny, self.nx)  # [64, 11, 11, 11]
                cube = full_cube[self.freq_from:self.freq_up_to, :, :, :]

                all_cubes.append(cube)
                all_source_coords.append(torch.tensor(source_pos, dtype=torch.float32))
                all_sample_info.append(torch.tensor([src_id], dtype=torch.int32))

        self.cubes = torch.stack(all_cubes)
        self.source_coords = torch.stack(all_source_coords)
        self.sample_info = torch.stack(all_sample_info)

        if self.normalize and self.mode == 'train':
            self.mean = self.cubes.mean()
            self.std = self.cubes.std()
            self.cubes = (self.cubes - self.mean) / (self.std + 1e-8)

        # Save the processed data
        save_data = {
            'cubes': self.cubes,
            'source_coords': self.source_coords,
            'sample_info': self.sample_info,
            'grid_xyz': self.grid_xyz,
            'nxnyz': (self.nx, self.ny, self.nz),
        }

        if self.normalize:
            save_data.update({'mean': self.mean, 'std': self.std})
        torch.save(save_data, processed_file)
        print(f"Saved processed ATF-3D {self.mode} data to {processed_file}")

    def _infer_src_splits_from_ids(self, source_ids):
        """
        Infer the src_splits format from actual source IDs in preprocessed data.
        Returns either [start, end] or [[start1, end1], [start2, end2], ...] format.
        """
        sorted_ids = sorted(source_ids)

        # Find consecutive ranges
        ranges = []
        start = sorted_ids[0]
        prev = sorted_ids[0]

        for i in range(1, len(sorted_ids)):
            current = sorted_ids[i]
            if current != prev + 1:  # Gap found
                ranges.append([start, prev + 1])  # End is exclusive
                start = current
            prev = current

        # Add the last range
        ranges.append([start, prev + 1])

        # Return single range or multiple ranges format
        if len(ranges) == 1:
            return ranges[0]  # [start, end]
        else:
            return ranges  # [[start1, end1], [start2, end2], ...]

    def __len__(self):
        return len(self.cubes)

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        indices = torch.randint(0, len(self.cubes), (num_samples,))
        z_full_batch = self.cubes[indices]
        src_xyz_batch = self.source_coords[indices]
        return z_full_batch.to(self.dummy.device), src_xyz_batch.to(self.dummy.device), indices


class SetEncoder_v12(nn.Module):
    # this was v1 and v2 models' setencoder, now siwtcihng to v3 for sadding positional encodings separately
    # Updated to support variable coord_dim (3 or 9 for geo-conditioning) and optional FFM on coords
    """Encodes a sparse set of observations into a sequence of tokens. and a pooled context vector."""

    def __init__(self, num_freqs=64, d_model=256, nhead=4, num_layers=3, coord_dim=3,
                 num_ff_coord=0, ffm_trainable=True, use_freq_ctx=False):
        super().__init__()
        self.d_model = d_model
        self.num_freqs = num_freqs
        self.use_freq_ctx = use_freq_ctx

        # Optional Fourier Feature Mapping for coordinates
        # num_ff_coord=0 → use raw coords (legacy, coord_dim=3)
        # num_ff_coord>0 → coords → FFM → 2*num_ff_coord dims before concat
        if num_ff_coord > 0:
            self.coord_ffm = FourierFeatureMapping(num_ff_coord, coord_dim, trainable=ffm_trainable)
            coord_feat_dim = 2 * num_ff_coord
        else:
            self.coord_ffm = None
            coord_feat_dim = coord_dim

        # MLP to tokenize each observation: [coord_feats, values] -> d_model
        self.tokenizer_mlp = nn.Sequential(
            nn.Linear(coord_feat_dim + num_freqs, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer encoder to mix observation tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.y_null_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Per-frequency context MLP (same design as SetEncoder v3 version)
        if use_freq_ctx:
            self.freq_ctx_mlp = nn.Sequential(
                nn.Linear(coord_feat_dim + 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
        else:
            self.freq_ctx_mlp = None

    def forward(self, obs_coords_rel, obs_values, obs_mask):
        """
        Args:
            obs_coords_rel (Tensor): Relative mic coordinates [B, M_max, 3]
            obs_values (Tensor): ATF magnitudes at those mics [B, M_max, 64]
            obs_mask (Tensor): Boolean mask indicating valid observations [B, M_max]
        Returns:
            tokens [B, M_max, d_model], pooled_context [B, d_model], freq_contexts [B, F, d_model] or None
        """
        # 1. Concatenate coordinates (optionally FFM-expanded) and values for each observation
        coords = self.coord_ffm(obs_coords_rel) if self.coord_ffm is not None else obs_coords_rel
        token_features = torch.cat([coords, obs_values], dim=-1)  # [B, M_max, coord_feat_dim + num_freqs]

        # 2. Project each observation to a token of dimension d_model
        tokens = self.tokenizer_mlp(token_features)  # [B, M_max, d_model]

        # 3. Use transformer to let observations communicate with each other
        # The transformer expects a padding mask where True means "ignore"
        padding_mask = ~obs_mask
        tokens = self.transformer_encoder(tokens, src_key_padding_mask=padding_mask)

        # --- NEW: Create the pooled context vector ---
        # To correctly average, we mask the padded tokens before summing.
        masked_tokens = tokens.masked_fill(~obs_mask.unsqueeze(-1), 0.0)
        # Sum valid tokens and divide by the number of valid tokens. Add epsilon for stability.
        num_valid_tokens = obs_mask.sum(dim=1, keepdim=True)
        pooled_context = masked_tokens.sum(dim=1) / (num_valid_tokens + 1e-8)

        # Optionally compute per-frequency contexts [B, F, d_model]
        freq_contexts = None
        if self.freq_ctx_mlp is not None:
            B, M_max, F = obs_values.shape
            freq_idx = torch.arange(F, device=obs_values.device, dtype=obs_values.dtype) / F  # [F]
            coords_exp = coords.unsqueeze(2).expand(B, M_max, F, -1)       # [B, M, F, coord_feat_dim]
            freq_exp   = freq_idx.view(1, 1, F, 1).expand(B, M_max, F, 1)  # [B, M, F, 1]
            atf_exp    = obs_values.unsqueeze(-1)                           # [B, M, F, 1]
            token_in   = torch.cat([coords_exp, freq_exp, atf_exp], dim=-1) # [B, M, F, coord_feat_dim+2]
            freq_tokens = self.freq_ctx_mlp(token_in)                       # [B, M, F, d_model]
            mask_exp = obs_mask.unsqueeze(-1).unsqueeze(-1)                 # [B, M, 1, 1]
            freq_tokens = freq_tokens.masked_fill(~mask_exp, 0.0)
            n_valid = obs_mask.sum(dim=1).view(B, 1, 1)                     # [B, 1, 1]
            freq_contexts = freq_tokens.sum(dim=1) / (n_valid + 1e-8)      # [B, F, d_model]

        return tokens, pooled_context, freq_contexts


# In fm_utils.py, replace your SetEncoder class


# """Part 2: Training for Classifier Free Guidance (CFG) """
class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass


class DDPM_ODE_Sampler(ODE):
    """
    Implements the deterministic Probability Flow ODE sampler for a DDPM.
    This is equivalent to DDIM sampling with eta=0 and is often more stable.
    """

    def __init__(self, noise_predictor_network, set_encoder, scheduler, config, guidance_scale=1.0):
        super().__init__()
        self.epsilon_theta = noise_predictor_network
        self.set_encoder = set_encoder
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale

        # Determine architecture from config to correctly call the forward pass
        self.architecture = config['model'].get('architecture_version', 'v1_legacy')

        # Pre-calculate scheduler values and move them to the model's device
        device = next(self.epsilon_theta.parameters()).device
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

    def get_predicted_noise(self, xt: torch.Tensor, t_continuous: torch.Tensor, **kwargs):
        """Helper function to get the CFG-combined noise prediction."""
        # This part handles context extraction and CFG, same as your DDIMSampler
        obs_coords_rel, obs_values, obs_mask = kwargs['obs_coords_rel'], kwargs['obs_values'], kwargs['obs_mask']
        guided_y_tokens, guided_pooled_context, guided_freq_ctx = self.set_encoder(obs_coords_rel, obs_values, obs_mask)

        B = xt.shape[0]
        unguided_y_tokens = self.set_encoder.y_null_token.expand(B, guided_y_tokens.shape[1], -1)
        unguided_pooled_context = self.set_encoder.y_null_token.squeeze(1).expand(B, -1)
        null_freq_ctx = (self.set_encoder.y_null_token.squeeze(1).expand(B, guided_freq_ctx.shape[1], -1)
                         if guided_freq_ctx is not None else None)

        model_kwargs_guided = {'context': guided_y_tokens, 'context_mask': obs_mask,
                               'pooled_context': guided_pooled_context, 'freq_contexts': guided_freq_ctx}
        model_kwargs_unguided = {'context': unguided_y_tokens, 'context_mask': obs_mask,
                                 'pooled_context': unguided_pooled_context, 'freq_contexts': null_freq_ctx}

        epsilon_theta_guided = self.epsilon_theta(xt, t_continuous.squeeze(), **model_kwargs_guided)
        epsilon_theta_unguided = self.epsilon_theta(xt, t_continuous.squeeze(), **model_kwargs_unguided)

        return (1 - self.guidance_scale) * epsilon_theta_unguided + self.guidance_scale * epsilon_theta_guided

    def drift_coefficient(self, xt: torch.Tensor, t_continuous: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the drift for the Probability Flow ODE from a noise prediction.
        The time `t` here is the continuous time in [0, 1].
        """
        # Map continuous time [0, 1] to discrete timesteps [0, T-1] for the scheduler
        timesteps = (t_continuous.squeeze() * (self.scheduler.num_timesteps - 1)).round().long()

        # Get scheduler values for the current timesteps
        alpha_bar_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)

        # 1. Predict the noise `epsilon` using the network
        predicted_noise = self.get_predicted_noise(xt, t_continuous, **kwargs)

        # 2. Predict the original sample (x0) from the noise prediction
        # Formula: x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        pred_x0 = (xt - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

        # 3. Calculate the drift for the ODE. This is a stable formulation for the VP SDE's reverse ODE.
        # It's derived from the score-based representation u_t = f(x,t) - 0.5 * g(t)^2 * s_t(x)
        # where f is the forward drift and s_t is the score.
        beta_t = self.scheduler.betas.to(xt.device)[timesteps].view(-1, 1, 1, 1, 1)
        drift = -0.5 * beta_t * (xt + pred_x0) / sqrt_one_minus_alpha_bar_t

        return drift


class CFGVectorFieldODE_3D(ODE):
    """
    An ODE wrapper for the 3D U-Net and SetEncoder for the ATF_3D.
    """

    def __init__(self, unet, set_encoder, guidance_scale=1):
        self.unet = unet
        self.set_encoder = set_encoder
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        # The simulator will pass 'y_tokens' and 'obs_mask' through kwargs
        y_tokens = kwargs['y_tokens']
        obs_mask = kwargs['obs_mask']

        # 1. Get the guided prediction
        guided_vector_field = self.unet(xt, t.squeeze(), context=y_tokens, context_mask=obs_mask)

        # 2. Get the unguided prediction
        null_tokens = self.set_encoder.y_null_token.expand(xt.shape[0], y_tokens.shape[1], -1)
        unguided_vector_field = self.unet(xt, t.squeeze(), context=null_tokens, context_mask=obs_mask)

        # Calculate the Mean Squared Error between the two predictions
        # drift_difference = torch.mean((guided_vector_field - unguided_vector_field) ** 2).item()
        # print("---------------------\n")
        # print(f"Drift Difference (MSE) for: {drift_difference:.12f}")
        # print("---------------------\n")

        # 3. Combine using the CFG formula and return a single DRIFT tensor
        combined_field = (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

        return combined_field


class CFGVectorFieldODE_3D_V2(ODE):
    """
    An ODE wrapper for the 3D U-Net and SetEncoder for the ATF_3D.
    """

    def __init__(self, unet, set_encoder, guidance_scale=1):
        self.unet = unet
        self.set_encoder = set_encoder
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        # The simulator will pass 'y_tokens', 'obs_mask', 'pooled_context', and 'freq_contexts' through kwargs
        y_tokens = kwargs['y_tokens']
        obs_mask = kwargs['obs_mask']
        pooled_context = kwargs['pooled_context']
        freq_contexts = kwargs.get('freq_contexts', None)

        B = xt.shape[0]
        null_tokens = self.set_encoder.y_null_token.expand(B, y_tokens.shape[1], -1)
        null_context = self.set_encoder.y_null_token.squeeze(1).expand(B, -1)
        null_freq_ctx = (self.set_encoder.y_null_token.squeeze(1).expand(B, freq_contexts.shape[1], -1)
                         if freq_contexts is not None else None)

        # 1. Get the guided prediction
        guided_vector_field = self.unet(
            xt, t.squeeze(),
            context=y_tokens, context_mask=obs_mask,
            pooled_context=pooled_context, freq_contexts=freq_contexts
        )

        # 2. Get the unguided prediction (null conditioning)
        unguided_vector_field = self.unet(
            xt, t.squeeze(),
            context=null_tokens, context_mask=obs_mask,
            pooled_context=null_context, freq_contexts=null_freq_ctx
        )

        # 3. Combine using the CFG formula
        combined_field = (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

        return combined_field


class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, y_dim: int,
                 **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path
        self.y_dim = y_dim

        # A learned embedding for the unconditional (null) case
        self.y_null = nn.Parameter(torch.randn(1, y_dim))

    #
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z (spectrograms) and y (coordinates) from p_data#
        z, y = self.path.p_data.sample(batch_size)  # (bs, c, h, w), y:(bs, 6)
        # Ensure y is on the correct device
        y = y.to(z.device)
        self.y_null = self.y_null.to(z.device)

        # Step 2: With probability eta, replace the coordinate vector with the null embedding
        is_conditional_mask = (torch.rand(y.shape[0], device=y.device) > self.eta)
        # Reshape for broadcasting: (bs,) -> (bs, 1)
        is_conditional_mask = is_conditional_mask.view(-1, 1)

        # Create the final conditioning tensor for the batch
        y_cond = torch.where(is_conditional_mask, y, self.y_null)

        # Step 3: Sample t (time) and x (noisy spectrogram)
        t = torch.rand(batch_size, 1, 1, 1).to(z.device)
        x = self.path.sample_conditional_path(z, t)

        # Step 4: Regress the model's output against the ground truth vector field
        ut_theta = self.model(x, t, y_cond)
        ut_ref = self.path.conditional_vector_field(x, z, t)

        error = torch.square(ut_theta - ut_ref)
        # Flatten error fr
        # om (bs, c, h, w) to (bs, -1) and sum over dimensions
        loss_per_sample = error.view(batch_size, -1).sum(dim=1)

        # Apply the mask to compute the loss only on conditional samples
        masked_loss = loss_per_sample * is_conditional_mask.squeeze()

        # Average the loss over the number of conditional samples
        # Add a small epsilon to avoid division by zero if no samples are conditional
        num_conditional_samples = is_conditional_mask.sum()
        mean_loss = masked_loss.sum() / (num_conditional_samples + 1e-8)

        # error = torch.einsum('bchw -> b', torch.square(ut_theta - ut_ref))  # (bs,)
        return mean_loss

    # **NEW: Validation loss implementation**
    @torch.no_grad()
    def get_valid_loss(self, valid_sampler: Sampleable, batch_size: int, **kwargs) -> torch.Tensor:
        # Step 1: Sample z and y from the validation data sampler
        z, y = valid_sampler.sample(batch_size)
        y = y.to(z.device)

        # Step 2: For validation, we ONLY use conditional samples. No CFG masking.
        y_cond = y

        # Step 3: Sample t and x
        t = torch.rand(batch_size, 1, 1, 1).to(z.device)
        x = self.path.sample_conditional_path(z, t)

        # Step 4: Calculate loss
        ut_theta = self.model(x, t, y_cond)
        ut_ref = self.path.conditional_vector_field(x, z, t)
        error = torch.square(ut_theta - ut_ref)
        loss_per_sample = error.view(batch_size, -1).sum(dim=1)

        return loss_per_sample.mean()


class ATF3DTrainer(Trainer):
    def __init__(self, path, model, set_encoder, eta, M_range, M_val_fixed, sigma, grid_xyz, loss_type: str, FM_vs_Diff: str,
                 version: bool, setencoderversion: str,
                 coord_mean: torch.Tensor, coord_std: torch.Tensor, idx_mes_pos_path=None,
                 time_weight_towards_end: bool = False, time_weight_mean: float = 1.2, **kwargs):
        super().__init__(models={'unet': model, 'set_encoder': set_encoder})

        # Load deterministic mic permutation matrix for validation
        if idx_mes_pos_path and os.path.exists(idx_mes_pos_path):
            print(f"Loading mic permutation matrix from {idx_mes_pos_path}")
            # Shape (1331, 1024)
            self.perm_matrix = torch.from_numpy(np.load(idx_mes_pos_path)).long()
        else:
            self.perm_matrix = None

        self.path = path
        self.set_encoder = set_encoder
        self.eta = eta
        self.M_range = [int(x) for x in M_range]  # preserve full list (e.g. [5,10,20,50])
        self.M_val_fixed = M_val_fixed
        self.sigma = sigma
        self.time_weight_towards_end = time_weight_towards_end
        self.time_weight_mean = time_weight_mean
        self.dev = next(model.parameters()).device  # cached once — avoids repeated parameter walks
        self.grid_xyz = grid_xyz.to(self.dev)  # (1331, 3)
        self.geo_conditioning = kwargs.get('geo_conditioning', False)
        self.room_dims = kwargs.get('room_dims', None)  # (Lx, Ly, Lz) in metres

        self.version = version
        self.setencoderversion = setencoderversion

        # ### <<< CHANGE 2: Store the coordinate statistics
        self.coord_mean = coord_mean.to(self.dev)
        self.coord_std = coord_std.to(self.dev)

        self.loss_type = loss_type
        self.freq_weight_max = kwargs.get('freq_weight_max', 3.0)
        self.FM_vs_Diff = FM_vs_Diff
        self.sweep_M = kwargs.get('sweep_M', False)
        self.exhaust_M = kwargs.get('exhaust_M', False)

        if self.FM_vs_Diff == 'score_matching':
            self.ddpm_scheduler = DDPMScheduler(num_timesteps=1000)
            # --- NEW: Store scheduler values needed for v-prediction ---
            self.sqrt_alphas_cumprod = self.ddpm_scheduler.sqrt_alphas_cumprod.to(self.dev)
            self.sqrt_one_minus_alphas_cumprod = self.ddpm_scheduler.sqrt_one_minus_alphas_cumprod.to(self.dev)

        if self.loss_type == 'weighted':
            print("--- Using PERCEPTUALLY WEIGHTED training loss. ---")
        elif self.loss_type == 'freq_weighted':
            print(f"--- Using FREQ-BIN WEIGHTED training loss (max weight={self.freq_weight_max}). ---")
        else:
            print("--- Using STANDARD training loss. ---")

        if self.perm_matrix is None:
            print("--- Mode: Tokyo | train: random mics | val: random mics ---")
        else:
            print(f"--- Mode: Training: Tokyo (seeded) | Validation: Fixed (M={self.M_val_fixed}, deterministic) ---")

        # A learnable embedding for the unconditional (null) case
        # d_model = set_encoder.d_model

    def _sample_timesteps(self, batch_size, device):
        if self.time_weight_towards_end:
            alpha = torch.randn(batch_size, device=device) * 2.0 + self.time_weight_mean
            return torch.sigmoid(alpha).view(-1, 1, 1, 1, 1)
        else:
            return torch.rand(batch_size, device=device).view(-1, 1, 1, 1, 1)

    def make_observation_set(self, z_full, src_xyz, sample_indices=None, deterministic=False):
        B, C, D, H, W = z_full.shape
        dev = z_full.device

        grid_xyz = self.grid_xyz.to(dev)  # ensure same device
        src_xyz = src_xyz.to(dev)

        N = self.grid_xyz.shape[0]  # Total number of mics (1331)

        if deterministic:
            if self.perm_matrix is None:
                raise ValueError("Deterministic mode requested but 'idx_mes_pos_path' was not provided/loaded.")
            if sample_indices is None:
                raise ValueError("Deterministic mode requires 'sample_indices' (absolute source IDs).")
            # For validation we use fixed M (as in evaluation)
            M_max = self.M_val_fixed

        else:
            # M_max for tensor padding: works for both [min,max] and discrete lists
            M_max = max(self.M_range)

        obs_coords_rel_list, obs_values_list, obs_mask_list = [], [], []

        # Loop over each sample in the batch to handle variable M
        for i in range(B):
            # Instead of sampling from a range, choose a value from the provided list
            # M = random.choice(self.M_range)

            if deterministic:
                M = self.M_val_fixed
                # Lookup source-specific fixed mic indices
                # sample_indices[i] is the ABSOLUTE source ID for this sample
                abs_src_idx = sample_indices[i] 
                # perm_matrix is [1331, 1024] -> col is source
                obs_indices = self.perm_matrix[:M, abs_src_idx].to(dev)
                # if i == 0:
                    # print(f"DEBUG: Source {abs_src_idx} using mics: {obs_indices.cpu().numpy()}")
            
            # OLD VERSION:  1. Randomly pick M for this sample 
            else:
                # If M_range has >2 values treat as discrete choices (e.g. [5,10,20]);
                # if exactly 2 values treat as [min, max] uniform range (e.g. [5,50]).
                if len(self.M_range) > 2:
                    M = random.choice(self.M_range)
                else:
                    M = torch.randint(self.M_range[0], self.M_range[1] + 1, (1,)).item()
                obs_indices = torch.randperm(N, device=dev)[:M]


            # 3. Gather coordinates and values
            obs_xyz = self.grid_xyz[obs_indices]  # [M, 3]
            obs_coords_rel = obs_xyz - src_xyz[i].unsqueeze(0)  # [M, 3]

            # ### <<< CHANGE 3: Apply the normalization
            # Use a small epsilon for numerical stability
            obs_coords_rel = (obs_coords_rel - self.coord_mean) / (self.coord_std + 1e-8)

            # Flatten spatial dimensions of the cube to easily gather values
            z_flat = z_full[i].view(C, -1)  # [64, 1331]
            obs_values = z_flat[:, obs_indices].transpose(0, 1)  # [M, 64]

            # 4. Pad tensors to max length for batching, gemini commented:
            pad_len = M_max - M
            obs_coords_rel_padded = nn.functional.pad(obs_coords_rel, (0, 0, 0, pad_len))
            obs_values_padded = nn.functional.pad(obs_values, (0, 0, 0, pad_len))
            # target sizes -- chatgpt version:
            # M_max = self.M_range[1]
            #
            # # coords: [M, 3] -> [M_max, 3]
            # obs_coords_rel_padded = torch.zeros(M_max, 3, device=z_full.device, dtype=obs_coords_rel.dtype)
            # obs_coords_rel_padded[:M] = obs_coords_rel
            #
            # # values: [M, 64] -> [M_max, 64]
            # obs_values_padded = torch.zeros(M_max, obs_values.shape[1], device=z_full.device, dtype=obs_values.dtype)
            # obs_values_padded[:M] = obs_values

            # Create a mask: True for valid observations, False for padding
            mask = torch.zeros(M_max, dtype=torch.bool, device=dev)
            mask[:M] = True

            obs_coords_rel_list.append(obs_coords_rel_padded)
            obs_values_list.append(obs_values_padded)
            obs_mask_list.append(mask)

        return torch.stack(obs_coords_rel_list), torch.stack(obs_values_list), torch.stack(obs_mask_list)

    def make_observation_set_fast(self, z_full, src_xyz, sample_indices=None, deterministic=False):
        """
        Vectorized GPU version of make_observation_set.
        Deterministic mode (validation): fixed M mics from perm_matrix per source.
        Random mode (training): independently random M and random mics per sample,
          mathematically equivalent to the loop-based make_observation_set.
        """
        B, C, D, H, W = z_full.shape
        dev = z_full.device
        N = self.grid_xyz.shape[0]  # total mic count (e.g. 1331)

        if deterministic:
            if self.perm_matrix is None or sample_indices is None:
                raise ValueError("Deterministic mode requires perm_matrix and sample_indices.")
            M_max = self.M_val_fixed
            obs_idx = self.perm_matrix[:M_max, sample_indices].T.to(dev)  # [B, M_max]
            mask = torch.ones(B, M_max, dtype=torch.bool, device=dev)

        else:
            M_max = max(self.M_range)

            # ── 1. Sample M independently per sample ────────────────────────
            # Discrete list (e.g. [5,10,20,50]): draw one value per sample
            # Contiguous range (e.g. [5,50]):     uniform integer per sample
            if len(self.M_range) > 2:
                M = torch.tensor(
                    [random.choice(self.M_range) for _ in range(B)],
                    dtype=torch.long, device=dev
                )  # [B]
            else:
                M = torch.randint(self.M_range[0], M_max + 1, (B,), device=dev)  # [B]

            # ── 2. Random mic indices per sample (vectorised randperm) ──────
            # topk on random keys picks M_max indices without sorting all N
            obs_idx = torch.topk(torch.rand(B, N, device=dev), M_max, dim=1).indices  # [B, M_max]

            # ── 3. Mask: True for the first M[i] positions ──────────────────
            ar = torch.arange(M_max, device=dev).unsqueeze(0)  # [1, M_max]
            mask = ar < M.unsqueeze(1)                          # [B, M_max] bool

        # ── 4. Coordinates (relative + normalised, optionally geo-augmented) ──
        obs_xyz = self.grid_xyz[obs_idx]                          # [B, M_max, 3]
        rel = obs_xyz - src_xyz.unsqueeze(1)                      # [B, M_max, 3]
        rel = (rel - self.coord_mean) / (self.coord_std + 1e-8)

        if self.geo_conditioning and self.room_dims is not None:
            # 6 perpendicular wall distances in metres, normalised by half the minimum
            # room dimension (half_min = 1.5 m for a 4×6×3 room) so values ∈ [0, ~4].
            # Dividing by a fixed constant preserves absolute physical scale:
            # the model learns e.g. "d≈0.13 → very close to wall → strong early reflection".
            # This replaces the old abs_src_norm(3)+d_nearest(1) which was partially redundant.
            Lx, Ly, Lz = self.room_dims
            half_min = min(Lx, Ly, Lz) / 2.0
            d_walls = torch.stack([
                src_xyz[:, 0],      Lx - src_xyz[:, 0],   # ←x, x→  wall distances
                src_xyz[:, 1],      Ly - src_xyz[:, 1],   # ←y, y→
                src_xyz[:, 2],      Lz - src_xyz[:, 2],   # ←z, z→
            ], dim=1) / half_min                          # [B, 6]
            d_walls_exp = d_walls.unsqueeze(1).expand(-1, M_max, -1)  # [B, M_max, 6]
            rel = torch.cat([rel, d_walls_exp], dim=-1)  # [B, M_max, 9]

        # ── 5. Gather ATF values ─────────────────────────────────────────────
        z_flat = z_full.view(B, C, -1)                            # [B, C, N]
        idx_expanded = obs_idx.unsqueeze(1).expand(-1, C, -1)     # [B, C, M_max]
        vals = torch.gather(z_flat, 2, idx_expanded).transpose(1, 2)  # [B, M_max, C]

        # Padded positions hold junk values but mask ensures transformer ignores them.
        return rel, vals, mask

    def get_train_loss_flow_matching(self, **kwargs) -> torch.Tensor:
        batch_size = kwargs.get('batch_size')
        iteration = kwargs.get('iteration')

        torch.manual_seed(42 + iteration)  # deterministic per-step: batch, mics, t, x0, CFG mask

        _do_time = iteration < 0

        # 2. Now everything below is tied to 'iteration'
        # This picks the SAME 4 sources every time this iteration is run
        # 1. Sample a batch of complete, clean 3D ATF cubes and their source coordinates
        if _do_time: torch.cuda.synchronize(); _t_data0 = time.time()
        z_full, src_xyz, _ = self.path.p_data.sample(batch_size)
        dev = self.dev
        z_full = z_full.to(dev)
        src_xyz = src_xyz.to(dev)
        if _do_time: torch.cuda.synchronize(); print(f"  [iter {iteration}] data+transfer={time.time()-_t_data0:.3f}s")

        x1 = z_full

        # sweep_M: loop over every M in M_range, average losses (AE-style)
        # t/x0/xt/ut_ref are shared; only the observation set varies per M
        if self.sweep_M:
            t = self._sample_timesteps(batch_size, x1.device)
            x0 = torch.randn_like(x1)
            xt = (1 - (1 - self.sigma) * t) * x0 + t * x1
            ut_ref = x1 - (1 - self.sigma) * x0
            is_conditional_mask = (torch.rand(batch_size, device=x1.device) > self.eta)
            losses = []
            for m in self.M_range:
                orig_range = self.M_range
                self.M_range = [m]
                obs_c, obs_v, obs_msk = self.make_observation_set_fast(z_full, src_xyz, deterministic=False)
                self.M_range = orig_range
                y_tok, pool_ctx, freq_ctx = self.set_encoder(obs_c, obs_v, obs_msk)
                null_tok = self.set_encoder.y_null_token.expand(batch_size, y_tok.shape[1], -1)
                null_ctx = self.set_encoder.y_null_token.squeeze(1).expand(batch_size, -1)
                fin_tok = torch.where(is_conditional_mask.view(-1, 1, 1), y_tok, null_tok)
                fin_ctx = torch.where(is_conditional_mask.view(-1, 1), pool_ctx, null_ctx)
                if freq_ctx is not None:
                    null_fctx = self.set_encoder.y_null_token.squeeze(1).expand(batch_size, freq_ctx.shape[1], -1)
                    fin_fctx = torch.where(is_conditional_mask.view(-1, 1, 1), freq_ctx, null_fctx)
                else:
                    fin_fctx = None
                mk = {'context': fin_tok, 'context_mask': obs_msk, 'pooled_context': fin_ctx, 'freq_contexts': fin_fctx}
                ut_theta = self.model(xt, t, **mk)
                if self.loss_type == 'weighted':
                    with torch.no_grad():
                        xt_lin = 10 ** ((xt * self.path.p_data.std + self.path.p_data.mean) / 20.0)
                        w = torch.clamp(1.0 / (xt_lin + 1e-6), max=10.0)
                    losses.append(torch.mean(torch.square(w * (ut_theta - ut_ref))))
                elif self.loss_type == 'freq_weighted':
                    F_bins = ut_theta.shape[1]
                    fw = torch.linspace(1.0, getattr(self, 'freq_weight_max', 3.0), F_bins,
                                        device=ut_theta.device).view(1, F_bins, 1, 1, 1)
                    losses.append(torch.mean(fw * torch.square(ut_theta - ut_ref)))
                else:
                    losses.append(torch.mean(torch.square(ut_theta - ut_ref)))
            return torch.stack(losses).mean()

        # 2. Create the sparse observation set (fast vectorised path)
        if _do_time: torch.cuda.synchronize(); _t0 = time.time()
        obs_coords_rel, obs_values, obs_mask = self.make_observation_set_fast(
            z_full, src_xyz, deterministic=False
        )
        if _do_time: torch.cuda.synchronize(); _t1 = time.time()

        # 3. Encode the observations into conditioning tokens
        y_tokens, pooled_context, freq_contexts = self.set_encoder(obs_coords_rel, obs_values, obs_mask)  # [B, M_max, d_model]
        if _do_time: torch.cuda.synchronize(); _t2 = time.time()

        # 4. Define the Flow Matching path from noise to data
        t = self._sample_timesteps(batch_size, x1.device)
        x0 = torch.randn_like(x1)

        # xt = (1 - t) * x0 + t * x1
        xt = (1 - (1 - self.sigma) * t) * x0 + t * x1
        # ut_ref = x1 - x0
        ut_ref = x1 - (1 - self.sigma) * x0

        # 5. Apply Classifier-Free Guidance during training
        # With probability eta, replace conditioning tokens with the null token
        is_conditional_mask = (torch.rand(batch_size, device=x1.device) > self.eta)

        # Broadcast y_null_token and select based on the mask
        null_tokens = self.set_encoder.y_null_token.expand(batch_size, y_tokens.shape[1], -1)
        null_context = self.set_encoder.y_null_token.squeeze(1).expand(batch_size, -1)

        final_tokens = torch.where(is_conditional_mask.view(-1, 1, 1), y_tokens, null_tokens)
        final_pooled_context = torch.where(is_conditional_mask.view(-1, 1), pooled_context, null_context)

        if freq_contexts is not None:
            null_freq_ctx = self.set_encoder.y_null_token.squeeze(1).expand(batch_size, freq_contexts.shape[1], -1)
            final_freq_ctx = torch.where(is_conditional_mask.view(-1, 1, 1), freq_contexts, null_freq_ctx)
        else:
            final_freq_ctx = None

        # The mask for the transformer (to ignore padding) is the same for both cases
        final_obs_mask = obs_mask

        # 6. Get the model's prediction for the velocity field
        model_kwargs = {
            'context': final_tokens,
            'context_mask': final_obs_mask,
            'pooled_context': final_pooled_context,
            'freq_contexts': final_freq_ctx
        }

        # The 3D U-Net's forward pass must accept `context` and `context_mask`
        if _do_time: torch.cuda.synchronize(); _t3 = time.time()
        ut_theta = self.model(xt, t, **model_kwargs)
        if _do_time: torch.cuda.synchronize(); _t4 = time.time()

        # 7. Compute the loss based on the selected type
        if self.loss_type == 'weighted':
            # --- Perceptually Weighted Loss ---
            with torch.no_grad():
                # Un-normalize to get approximate dB magnitudes
                xt_denorm = xt * self.path.p_data.std + self.path.p_data.mean
                # Convert from dB to a linear-like scale for weighting
                xt_linear = 10 ** (xt_denorm / 20.0)

                epsilon = 1e-6
                weights = 1.0 / (xt_linear + epsilon)
                weights = torch.clamp(weights, max=10.0)  # Prevent instability

            weighted_error = weights * (ut_theta - ut_ref)
            loss = torch.mean(torch.square(weighted_error))

        elif self.loss_type == 'freq_weighted':
            # --- Per-Frequency-Bin Weighted Loss ---
            # ut_theta / ut_ref shape: [B, F, D, H, W]  (F = num freq bins)
            # Linear ramp: bin 0 gets weight 1.0, bin F-1 gets freq_weight_max.
            # Diagnostic showed LSD grows monotonically from DC to Nyquist;
            # upweighting high bins forces the network to invest more capacity there.
            F_bins = ut_theta.shape[1]
            freq_weight_max = getattr(self, 'freq_weight_max', 3.0)
            freq_weights = torch.linspace(1.0, freq_weight_max, F_bins,
                                          device=ut_theta.device)  # [F]
            freq_weights = freq_weights.view(1, F_bins, 1, 1, 1)   # broadcast [B,F,D,H,W]
            loss = torch.mean(freq_weights * torch.square(ut_theta - ut_ref))

        elif self.loss_type == 'standard':  # Default to 'standard'
            # --- Standard Loss ---
            loss = torch.mean(torch.square(ut_theta - ut_ref))

        if _do_time:
            print(f"  [iter {iteration}] obs={_t1-_t0:.3f}s  encoder={_t2-_t1:.3f}s  unet={_t4-_t3:.3f}s")

        return loss

    def get_train_loss_ddpm(self, **kwargs):
        batch_size = kwargs.get('batch_size')
        z_full, src_xyz, _ = self.path.p_data.sample(batch_size)

        dev = self.dev
        z_full, src_xyz = z_full.to(dev), src_xyz.to(dev)

        x1 = z_full  # Clean data

        obs_coords_rel, obs_values, obs_mask = self.make_observation_set(z_full, src_xyz)
        y_tokens, pooled_context, freq_contexts = self.set_encoder(obs_coords_rel, obs_values, obs_mask)

        # 1. Sample discrete timesteps for DDPM
        timesteps = torch.randint(0, self.ddpm_scheduler.num_timesteps, (batch_size,), device=dev).long()

        # 2. Create the noise (our target) and the noised sample `xt`
        noise_target = torch.randn_like(x1)
        xt = self.ddpm_scheduler.add_noise(original_samples=x1, noise=noise_target, timesteps=timesteps)
        xt = xt.float()  # ensure float32

        # 3. Apply Classifier-Free Guidance during training
        is_conditional_mask = (torch.rand(batch_size, device=dev) > self.eta)
        null_tokens = self.set_encoder.y_null_token.expand(batch_size, y_tokens.shape[1], -1)
        null_context = self.set_encoder.y_null_token.squeeze(1).expand(batch_size, -1)
        final_tokens = torch.where(is_conditional_mask.view(-1, 1, 1), y_tokens, null_tokens)
        final_pooled_context = torch.where(is_conditional_mask.view(-1, 1), pooled_context, null_context)
        if freq_contexts is not None:
            null_freq_ctx = self.set_encoder.y_null_token.squeeze(1).expand(batch_size, freq_contexts.shape[1], -1)
            final_freq_ctx = torch.where(is_conditional_mask.view(-1, 1, 1), freq_contexts, null_freq_ctx)
        else:
            final_freq_ctx = None

        model_kwargs = {'context': final_tokens, 'context_mask': obs_mask, 'pooled_context': final_pooled_context,
                        'freq_contexts': final_freq_ctx}

        # 4. Pass continuous-time equivalent to the model
        continuous_time = timesteps.float() / self.ddpm_scheduler.num_timesteps
        continuous_time = continuous_time.view(-1, 1, 1, 1, 1)

        # The model's job is to predict the noise that was added
        predicted_noise = self.model(xt, continuous_time, **model_kwargs)

        # 5. The loss is the mean squared error between the predicted noise and the actual noise
        loss = torch.mean(torch.square(predicted_noise - noise_target))

        return loss

    def get_train_loss(self, **kwargs):
        if self.FM_vs_Diff == 'score_matching':
            return self.get_train_loss_ddpm(**kwargs)
        elif self.FM_vs_Diff == "flow_matching":  # Default to flow matching
            return self.get_train_loss_flow_matching(**kwargs)

    @torch.no_grad()
    @torch.no_grad()
    def get_val_loss_ddpm(self, valid_sampler: Sampleable, **kwargs) -> torch.Tensor:
        batch_size = kwargs.get('batch_size')
        z_full, src_xyz, indices = valid_sampler.sample(batch_size)

        dev = self.dev
        z_full, src_xyz = z_full.to(dev), src_xyz.to(dev)

        x1 = z_full

        # Get absolute source IDs for deterministic mic selection
        abs_src_ids = valid_sampler.sample_info[indices].flatten()

        obs_coords_rel, obs_values, obs_mask = self.make_observation_set(
            z_full, src_xyz, 
            sample_indices=abs_src_ids, 
            deterministic = self.perm_matrix is not None
        )
        y_tokens, pooled_context, freq_contexts = self.set_encoder(obs_coords_rel, obs_values, obs_mask)

        timesteps = torch.randint(0, self.ddpm_scheduler.num_timesteps, (batch_size,), device=dev).long()
        noise_target = torch.randn_like(x1)
        xt = self.ddpm_scheduler.add_noise(original_samples=x1, noise=noise_target, timesteps=timesteps)
        xt = xt.float()

        model_kwargs = {'context': y_tokens, 'context_mask': obs_mask, 'pooled_context': pooled_context,
                        'freq_contexts': freq_contexts}

        continuous_time = timesteps.float() / self.ddpm_scheduler.num_timesteps
        continuous_time = continuous_time.view(-1, 1, 1, 1, 1)

        predicted_noise = self.model(xt, continuous_time, **model_kwargs)

        loss = torch.mean(torch.square(predicted_noise - noise_target))

        # LSD monitor: denormalize x1 vs a rough x0-prediction, freq_dim=1 ([B,F,D,H,W])
        mean, std = valid_sampler.mean.to(dev), valid_sampler.std.to(dev)
        x1_db = x1 * std + mean
        # Predict x0 from noise prediction at a representative alpha_bar (midpoint t=0.5)
        alpha_bar = self.ddpm_scheduler.alphas_cumprod[499].to(dev)
        pred_x0 = (xt - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        pred_x0_db = pred_x0 * std + mean
        lsd = torch.sqrt(torch.mean((pred_x0_db - x1_db) ** 2, dim=1)).mean()

        return loss, lsd

    @torch.no_grad()
    def get_val_loss_flow_matching(self, valid_sampler: Sampleable, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the full validation set:
          - FM MSE : single forward pass per batch (cheap, tracks training objective)
          - LSD    : 10-step Euler reconstruction per batch (real metric, in dB)
        Returns (mean_fm_mse, mean_lsd) as scalar tensors.
        """
        batch_size = kwargs.get('batch_size')
        compute_lsd = kwargs.get('compute_lsd', True)
        dev = self.dev
        n_val = len(valid_sampler.cubes)
        mean_db = valid_sampler.mean.to(dev)
        std_db  = valid_sampler.std.to(dev)

        # --- Build the correct ODE wrapper once (no weight copies, just references) ---
        # Always use V2 wrapper (passes pooled_context); v1_legacy accepts it with pooled_context=None default
        ode = CFGVectorFieldODE_3D_V2(unet=self.model, set_encoder=self.set_encoder)
        simulator = EulerSimulator(ode=ode)

        # Pre-build timestep template for Euler integration (N_STEPS=10 good for monitoring)
        N_STEPS = 10
        ts_1d = torch.linspace(0, 1, N_STEPS + 1, device=dev)  # [N_STEPS+1]

        all_mse, all_lsd = [], []

        # Iterate over the full validation set in fixed-size batches
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            idx = torch.arange(start, end)
            B = idx.shape[0]

            z_full = valid_sampler.cubes[idx].to(dev)          # [B, F, D, H, W]
            src_xyz = valid_sampler.source_coords[idx].to(dev) # [B, 3]
            abs_src_ids = valid_sampler.sample_info[idx].flatten()

            x1 = z_full

            # ── Conditioning ────────────────────────────────────────────────
            obs_coords_rel, obs_values, obs_mask = self.make_observation_set_fast(
                z_full, src_xyz,
                sample_indices=abs_src_ids,
                deterministic=self.perm_matrix is not None
            )
            y_tokens, pooled_context, freq_contexts = self.set_encoder(obs_coords_rel, obs_values, obs_mask)

            # ── 1. FM MSE (single forward pass) ─────────────────────────────
            torch.manual_seed(start)  # deterministic per batch, reproducible across runs
            t  = torch.rand(B, device=dev).view(-1, 1, 1, 1, 1)
            x0 = torch.randn_like(x1)
            xt     = (1 - (1 - self.sigma) * t) * x0 + t * x1
            ut_ref =  x1 - (1 - self.sigma) * x0

            model_kwargs = {'context': y_tokens, 'context_mask': obs_mask, 'pooled_context': pooled_context,
                            'freq_contexts': freq_contexts}
            ut_theta = self.model(xt, t, **model_kwargs)
            batch_mse = torch.mean(torch.square(ut_theta - ut_ref))
            all_mse.append(batch_mse)

            if compute_lsd:
                # ── 2. Full ODE reconstruction → LSD ────────────────────────────
                torch.manual_seed(start)  # same seed → same x0 for reproducibility
                x0_recon = torch.randn_like(x1)

                # ts shape: [B, N_STEPS+1, 1, 1, 1, 1]
                ts = ts_1d.view(1, -1, 1, 1, 1, 1).expand(B, -1, -1, -1, -1, -1)

                sim_kwargs = dict(y_tokens=y_tokens, obs_mask=obs_mask, pooled_context=pooled_context,
                                  freq_contexts=freq_contexts, silent=True)

                x1_hat = simulator.simulate(x0_recon, ts, **sim_kwargs)  # [B, F, D, H, W]

                # Denormalize both prediction and ground truth
                x1_hat_db = x1_hat * std_db + mean_db
                x1_db     = x1     * std_db + mean_db

                # LSD: sqrt(mean_over_freq((pred-gt)^2)), then mean over all positions & batch
                lsd = torch.sqrt(torch.mean((x1_hat_db - x1_db) ** 2, dim=1)).mean()
                all_lsd.append(lsd)

        mean_fm_mse = torch.stack(all_mse).mean()
        mean_lsd    = torch.stack(all_lsd).mean() if all_lsd else None
        return mean_fm_mse, mean_lsd

    def get_valid_loss(self, **kwargs):
        if self.FM_vs_Diff == 'score_matching':
            return self.get_val_loss_ddpm(**kwargs)
        elif self.FM_vs_Diff == "flow_matching":  # Default to flow matching
            return self.get_val_loss_flow_matching(**kwargs)
        # Both branches return (loss, lsd)


# """ Part 3: An Architecture for Spectrograms: Building a U-Net """

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        # Converts (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels)
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone()  # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x)  # (bs, c, h, w)

        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding (conditional embedding)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + y_embed

        # Second conv block
        x = self.block2(x)  # (bs, c, h, w)

        # Add back residual
        x = x + res  # (bs, c, h, w)

        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, context_embed_dim, groups=8):
        super().__init__()

        # Ensure num_groups is valid
        if in_channels < groups: groups = 1
        if out_channels < groups: groups = 1

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # Adapters for time and context embeddings
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_channels))
        self.context_mlp = nn.Sequential(nn.SiLU(), nn.Linear(context_embed_dim, out_channels))

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # This layer ensures the residual connection has the same number of channels
        self.residual_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_embed, context_embed):
        res = self.residual_conv(x)
        h = self.block1(x)

        # Add time and context embeddings
        time_emb = self.time_mlp(t_embed).view(x.shape[0], -1, 1, 1, 1)
        context_emb = self.context_mlp(context_embed).view(x.shape[0], -1, 1, 1, 1)
        h = h + time_emb + context_emb

        h = self.block2(h)
        return h + res


class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int,
                 y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x


class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int,
                 y_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


class CrossAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, d_model, nhead=4):
        super().__init__()
        # The attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            kdim=d_model,  # Key dimension from context
            vdim=d_model,  # Value dimension from context
            num_heads=nhead,
            batch_first=True
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, context, context_mask):
        """
        Args:
            x (Tensor): The spatial feature map from the U-Net [B, C, D, H, W]
            context (Tensor): The conditioning tokens from SetEncoder [B, M, d_model]
            context_mask (Tensor): The padding mask for the context [B, M]
        """
        B, C, D, H, W = x.shape

        # 1. Reshape spatial features to a sequence for attention
        # Query: The pixels/voxels of our image
        query = x.view(B, C, -1).permute(0, 2, 1)  # [B, D*H*W, C]

        # 2. Perform cross-attention
        # The query (our image pixels) attends to the key/value (our observation tokens)
        attn_output, _ = self.attention(
            query=query,
            key=context,
            value=context,
            key_padding_mask=~context_mask  # Invert mask: True means "ignore"
        )

        # 3. Add and normalize (residual connection)
        x_flat = query + attn_output
        x_flat = self.norm(x_flat)

        # 4. Reshape back to the original 3D spatial format
        return x_flat.permute(0, 2, 1).view(B, C, D, H, W)


# A more stable 3D convolutional block using GroupNorm
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        # Ensure num_groups is valid
        if out_channels < groups:
            groups = 1 if out_channels == 1 else 2  # A simple fallback

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU()  # Using SiLU (Swish) as a modern alternative to ReLU
        )

    def forward(self, x):
        return self.block(x)


# Second version with dynamic parametric channel unet
class CrossAttentionUNet3D(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, channels=[32, 64, 128], d_model=256, nhead=4, input_size=11,
                 init_kernel_size=3, freq_channel_bias=False, freq_film=False, freq_ctx=False):
        super().__init__()
        # Ensure channel dimensions are divisible by the number of attention heads
        assert all(c % nhead == 0 for c in channels), "Channel dimensions must be divisible by nhead"
        assert init_kernel_size >= 1, "init_kernel_size must be >= 1"

        # Optional learnable per-frequency channel bias added before init_conv.
        # Acts like a positional encoding over the frequency axis, telling the network
        # explicitly which channel corresponds to which frequency (DC vs high freq).
        # ~in_channels params, negligible cost.
        if freq_channel_bias:
            self.freq_channel_bias = nn.Parameter(torch.zeros(in_channels, 1, 1, 1))
        else:
            self.freq_channel_bias = None

        # FiLM: observation-conditional per-frequency scale+shift derived from SetEncoder pooled_context.
        # Projects pooled_context [B, d_model] → scale [B, F] and shift [B, F], then modulates
        # the UNet input x [B, F, D, H, W] before the first conv: x = x*(1+scale) + shift.
        # Initialised to zero so it starts as identity. ~2 * F * d_model extra params.
        if freq_film:
            self.film_scale = nn.Linear(d_model, in_channels)
            self.film_shift = nn.Linear(d_model, in_channels)
            nn.init.zeros_(self.film_scale.weight); nn.init.zeros_(self.film_scale.bias)
            nn.init.zeros_(self.film_shift.weight); nn.init.zeros_(self.film_shift.bias)
        else:
            self.film_scale = None
            self.film_shift = None

        # freq_ctx FiLM: per-frequency context [B, F, d_model] → per-channel scale/shift.
        # Linear(d_model, 1) applied independently per freq channel; zero-init → identity start.
        if freq_ctx:
            self.freq_ctx_film_scale = nn.Linear(d_model, 1)
            self.freq_ctx_film_shift = nn.Linear(d_model, 1)
            nn.init.zeros_(self.freq_ctx_film_scale.weight); nn.init.zeros_(self.freq_ctx_film_scale.bias)
            nn.init.zeros_(self.freq_ctx_film_shift.weight); nn.init.zeros_(self.freq_ctx_film_shift.bias)
        else:
            self.freq_ctx_film_scale = None
            self.freq_ctx_film_shift = None

        self.pad = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0.0)
        self.time_embedder = FourierEncoder(d_model)

        num_levels = len(channels) - 1
        divisor = 2 ** num_levels

        # Calculate the smallest target size divisible by the divisor
        self.target_size = math.ceil(input_size / divisor) * divisor
        total_pad = self.target_size - input_size

        # Distribute padding (e.g., for 5 total, pad with 2 on left, 3 on right)
        pad_front = total_pad // 2
        pad_back = total_pad - pad_front
        self.padding_tuple = (pad_front, pad_back, pad_front, pad_back, pad_front, pad_back)

        # Store the crop indices
        self.crop_start = pad_front
        self.crop_end = pad_front + input_size

        # --- DYNAMICALLY BUILD THE U-NET ---

        # Initial convolution
        init_groups = 8 if channels[0] % 8 == 0 else (4 if channels[0] % 4 == 0 else (2 if channels[0] % 2 == 0 else 1))
        init_padding = 1 if init_kernel_size == 3 else 'same'
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=init_kernel_size, padding=init_padding),
            nn.GroupNorm(num_groups=init_groups, num_channels=channels[0]),
            nn.SiLU()
        )

        # --- Encoder Path ---
        self.encoders = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(nn.Sequential(ConvBlock3D(channels[i], channels[i + 1]), nn.MaxPool3d(2)))
            self.encoder_attns.append(CrossAttentionBlock3D(in_channels=channels[i + 1], d_model=d_model, nhead=nhead))

        # --- Bottleneck ---
        bottleneck_channels = channels[-1]
        self.bottleneck = ConvBlock3D(bottleneck_channels, bottleneck_channels)
        self.time_mlp = nn.Linear(d_model, bottleneck_channels)  # Projects time to the deepest channel dimension
        self.attn_mid = CrossAttentionBlock3D(in_channels=bottleneck_channels, d_model=d_model, nhead=nhead)

        # --- Decoder Path ---
        self.decoders = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(reversed_channels) - 1):
            # Upsampling transpose convolution
            up_conv = nn.ConvTranspose3d(reversed_channels[i], reversed_channels[i + 1], kernel_size=2, stride=2)
            # Convolutional block after concatenating with skip connection
            conv = ConvBlock3D(reversed_channels[i + 1] * 2, reversed_channels[i + 1])
            self.decoders.append(nn.ModuleDict({'up_conv': up_conv, 'conv': conv}))

        # --- Final Convolution ---
        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x, t, context, context_mask, pooled_context=None, freq_contexts=None):
        # x: [B, C, 11, 11, 11]
        B = x.size(0)
        x = F.pad(x, self.padding_tuple, mode='reflect')

        # Apply learnable per-frequency bias (if enabled)
        if self.freq_channel_bias is not None:
            x = x + self.freq_channel_bias

        # FiLM: observation-conditional per-frequency scale+shift from pooled_context
        if self.film_scale is not None and pooled_context is not None:
            scale = self.film_scale(pooled_context).view(B, -1, 1, 1, 1)  # [B, F, 1, 1, 1]
            shift = self.film_shift(pooled_context).view(B, -1, 1, 1, 1)
            x = x * (1 + scale) + shift

        # freq_ctx FiLM: per-frequency context [B, F, d_model] → per-channel scale/shift
        if self.freq_ctx_film_scale is not None and freq_contexts is not None:
            fc_scale = self.freq_ctx_film_scale(freq_contexts).squeeze(-1)  # [B, F]
            fc_shift = self.freq_ctx_film_shift(freq_contexts).squeeze(-1)  # [B, F]
            x = x * (1 + fc_scale[:, :, None, None, None]) + fc_shift[:, :, None, None, None]

        # Initial conv
        x = self.init_conv(x)

        # --- Encoder with Skip Connections ---
        skip_connections = [x]
        for encoder, attn in zip(self.encoders, self.encoder_attns):
            x = encoder(x)
            x = attn(x, context, context_mask)
            skip_connections.append(x)

        # --- Bottleneck ---
        bn = self.bottleneck(x)
        t_emb = self.time_mlp(self.time_embedder(t.unsqueeze(-1)))
        bn = bn + t_emb.view(B, -1, 1, 1, 1)  # Use -1 to be fully dynamic
        bn = self.attn_mid(bn, context, context_mask)

        # --- Decoder ---
        # We iterate through decoders and the *reversed* skip connections
        x = bn
        for i, decoder_module in enumerate(self.decoders):
            skip = skip_connections[-(i + 2)]  # Get corresponding skip connection
            x = decoder_module['up_conv'](x)
            x = torch.cat([x, skip], dim=1)
            x = decoder_module['conv'](x)

        # --- Final Output ---
        out = self.final_conv(x)
        s = self.crop_start
        e = self.crop_end
        # print(s, e, out.shape)
        return out[..., s:e, s:e, s:e]  # Crop back to original size


class CrossAttentionUNet3D_RED3d(nn.Module):
    """
    Parametric v2 UNet with residual blocks + cross-attention at every level.
    Supports arbitrary depth via len(channels):
      channels=[A, B, C]       → 2 down-steps, bottleneck at C  (original)
      channels=[A, B, C, D]    → 3 down-steps, bottleneck at D
      etc.

    Channel layout:
      Encoder level 0:  channels[0] → channels[0]  (identity, full resolution)
      Encoder level i≥1: channels[i-1] → channels[i]
      Bottleneck:       channels[-2] → channels[-1] → channels[-2]
      Decoder level k:  upsample + skip-concat → channels[idx-1]  (mirrors encoder)
    """
    def __init__(self, channels, d_model, nhead, in_channels=20, out_channels=20, input_size=11,
                 init_kernel_size=3, freq_channel_bias=False, freq_film=False, freq_ctx=False):
        super().__init__()
        assert len(channels) >= 3, "channels must have at least 3 values (2 encoder levels + bottleneck peak)"
        assert init_kernel_size >= 1, "init_kernel_size must be >= 1"

        # Optional learnable per-frequency channel bias (see CrossAttentionUNet3D for rationale)
        if freq_channel_bias:
            self.freq_channel_bias = nn.Parameter(torch.zeros(in_channels, 1, 1, 1))
        else:
            self.freq_channel_bias = None

        # FiLM: see CrossAttentionUNet3D for rationale
        if freq_film:
            self.film_scale = nn.Linear(d_model, in_channels)
            self.film_shift = nn.Linear(d_model, in_channels)
            nn.init.zeros_(self.film_scale.weight); nn.init.zeros_(self.film_scale.bias)
            nn.init.zeros_(self.film_shift.weight); nn.init.zeros_(self.film_shift.bias)
        else:
            self.film_scale = None
            self.film_shift = None

        # freq_ctx FiLM: per-frequency context [B, F, d_model] → per-channel scale/shift
        if freq_ctx:
            self.freq_ctx_film_scale = nn.Linear(d_model, 1)
            self.freq_ctx_film_shift = nn.Linear(d_model, 1)
            nn.init.zeros_(self.freq_ctx_film_scale.weight); nn.init.zeros_(self.freq_ctx_film_scale.bias)
            nn.init.zeros_(self.freq_ctx_film_shift.weight); nn.init.zeros_(self.freq_ctx_film_shift.bias)
        else:
            self.freq_ctx_film_scale = None
            self.freq_ctx_film_shift = None

        self.time_embedder = FourierEncoder(d_model)

        # --- Padding: pad input so spatial size is divisible by 2^n_downs ---
        # n_downs = len(channels) - 1  spatial downsampling steps (bottleneck peak is channels[-1])
        n_downs = len(channels) - 1
        divisor = 2 ** n_downs
        # Minimum target_size=16 ensures bottleneck >= 4³=64 voxels for global spatial capacity.
        # For n_downs=2: formula gives 12 (bottleneck 3³=27) but empirically 16 (bottleneck 4³=64)
        # achieves 1.80 dB vs 2.19 dB — larger bottleneck better captures global room acoustics.
        # For n_downs=3: formula gives 16, same result.
        target_size = max(math.ceil(input_size / divisor) * divisor, 16)
        total_pad = target_size - input_size
        pad_front = total_pad // 2
        pad_back = total_pad - pad_front
        self.padding_tuple = (pad_front, pad_back, pad_front, pad_back, pad_front, pad_back)
        self.crop_start = pad_front
        self.crop_end = pad_front + input_size
        self.n_downs = n_downs

        # --- Initial conv: in_channels → channels[0] ---
        init_padding = 1 if init_kernel_size == 3 else 'same'
        self.init_conv = nn.Conv3d(in_channels, channels[0], kernel_size=init_kernel_size, padding=init_padding)

        # --- Encoder: n_downs levels, each with ResBlock + CrossAttn + MaxPool ---
        # Level 0:  channels[0] → channels[0]  (no channel expansion at first level)
        # Level i≥1: channels[i-1] → channels[i]
        self.enc_res  = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        for i in range(n_downs):
            in_c  = channels[0] if i == 0 else channels[i - 1]
            out_c = channels[i]
            self.enc_res.append(ResidualBlock3D(in_c, out_c, d_model, d_model))
            self.enc_attn.append(CrossAttentionBlock3D(out_c, d_model, nhead))
        self.down = nn.MaxPool3d(2)  # shared; stateless

        # --- Bottleneck: channels[-2] → channels[-1] → channels[-2] ---
        self.bottle_res1 = ResidualBlock3D(channels[-2], channels[-1], d_model, d_model)
        self.bottle_attn = CrossAttentionBlock3D(channels[-1], d_model, nhead)
        self.bottle_res2 = ResidualBlock3D(channels[-1], channels[-2], d_model, d_model)

        # --- Decoder: n_downs levels (mirrors encoder in reverse) ---
        # Step k (k=0 = deepest):
        #   idx      = n_downs - 1 - k   (index into channels for current feature map)
        #   cur_c    = channels[idx]       (current channels, same as matching encoder skip)
        #   out_c    = channels[idx-1]     (output channels; channels[0] for the final step)
        #   upsample : ConvTranspose cur_c → cur_c
        #   concat   : cur_c + cur_c (skip) = 2*cur_c
        #   ResBlock : 2*cur_c → out_c
        #   CrossAttn: out_c
        self.dec_up   = nn.ModuleList()
        self.dec_res  = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        for k in range(n_downs):
            idx   = n_downs - 1 - k
            cur_c = channels[idx]
            out_c = channels[idx - 1] if idx > 0 else channels[0]
            self.dec_up.append(nn.ConvTranspose3d(cur_c, cur_c, kernel_size=2, stride=2))
            self.dec_res.append(ResidualBlock3D(cur_c * 2, out_c, d_model, d_model))
            self.dec_attn.append(CrossAttentionBlock3D(out_c, d_model, nhead))

        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x, t, context, context_mask, pooled_context=None, freq_contexts=None):
        B = x.size(0)
        x = F.pad(x, self.padding_tuple, mode='reflect')

        # Apply learnable per-frequency bias (if enabled)
        if self.freq_channel_bias is not None:
            x = x + self.freq_channel_bias

        # FiLM: observation-conditional per-frequency scale+shift from pooled_context
        if self.film_scale is not None and pooled_context is not None:
            scale = self.film_scale(pooled_context).view(B, -1, 1, 1, 1)
            shift = self.film_shift(pooled_context).view(B, -1, 1, 1, 1)
            x = x * (1 + scale) + shift

        # freq_ctx FiLM: per-frequency context [B, F, d_model] → per-channel scale/shift
        if self.freq_ctx_film_scale is not None and freq_contexts is not None:
            fc_scale = self.freq_ctx_film_scale(freq_contexts).squeeze(-1)  # [B, F]
            fc_shift = self.freq_ctx_film_shift(freq_contexts).squeeze(-1)  # [B, F]
            x = x * (1 + fc_scale[:, :, None, None, None]) + fc_shift[:, :, None, None, None]

        t_emb = self.time_embedder(t.squeeze())

        # --- Encoder: save skip connections before each downsampling ---
        x = self.init_conv(x)
        skips = []
        for res, attn in zip(self.enc_res, self.enc_attn):
            x = res(x, t_emb, pooled_context)
            x = attn(x, context, context_mask)
            skips.append(x)
            x = self.down(x)

        # --- Bottleneck ---
        x = self.bottle_res1(x, t_emb, pooled_context)
        x = self.bottle_attn(x, context, context_mask)
        x = self.bottle_res2(x, t_emb, pooled_context)

        # --- Decoder: upsample, concat skip, ResBlock, CrossAttn ---
        for up, res, attn, skip in zip(self.dec_up, self.dec_res, self.dec_attn, reversed(skips)):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
            x = res(x, t_emb, pooled_context)
            x = attn(x, context, context_mask)

        out = self.final_conv(x)
        s, e = self.crop_start, self.crop_end
        return out[..., s:e, s:e, s:e]


# V3 SELF ATTENTION UNET

class SelfAttentionBlock3D(nn.Module):
    """ Applies self-attention to a 3D spatial feature map. """

    def __init__(self, channels, nhead=4, groups=4):
        super().__init__()
        # Ensure num_groups is valid
        if channels < groups: groups = 1

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=nhead,
            batch_first=True
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_norm = self.norm(x)

        # Reshape for attention: [B, C, D*H*W] -> [B, D*H*W, C]
        seq = x_norm.view(B, C, -1).permute(0, 2, 1)

        # Self-attention: query, key, and value are all the same
        attn_output, _ = self.attention(seq, seq, seq)

        # Add residual and reshape back to 3D
        out = x + attn_output.permute(0, 2, 1).view(B, C, D, H, W)
        return out


class ResidualAttentionBlock3D(nn.Module):
    """ A combined residual and attention block for the U-Net v3. """

    def __init__(self, in_channels, out_channels, time_embed_dim, context_embed_dim, nhead=4, groups=8):
        super().__init__()
        self.res_block = ResidualBlock3D(in_channels, out_channels, time_embed_dim, context_embed_dim, groups)
        self.self_attn = SelfAttentionBlock3D(out_channels, nhead, groups)

    def forward(self, x, t_embed, context_embed):
        # First, pass through the residual block with conditioning injection
        x = self.res_block(x, t_embed, context_embed)
        # Then, apply self-attention for spatial reasoning
        x = self.self_attn(x)
        return x


class CrossAttentionUNet3D_v3(nn.Module):
    """
    U-Net v3: Combines Residual Blocks, Self-Attention, and Cross-Attention.
    """

    def __init__(self, in_channels=20, out_channels=20, channels=[32, 64, 128], d_model=256, nhead=4, input_size=11,
                 init_kernel_size=3, freq_channel_bias=False, freq_film=False, freq_ctx=False):
        super().__init__()
        assert init_kernel_size >= 1, "init_kernel_size must be >= 1"

        # Optional learnable per-frequency channel bias (see CrossAttentionUNet3D for rationale)
        if freq_channel_bias:
            self.freq_channel_bias = nn.Parameter(torch.zeros(in_channels, 1, 1, 1))
        else:
            self.freq_channel_bias = None

        # FiLM: see CrossAttentionUNet3D for rationale
        if freq_film:
            self.film_scale = nn.Linear(d_model, in_channels)
            self.film_shift = nn.Linear(d_model, in_channels)
            nn.init.zeros_(self.film_scale.weight); nn.init.zeros_(self.film_scale.bias)
            nn.init.zeros_(self.film_shift.weight); nn.init.zeros_(self.film_shift.bias)
        else:
            self.film_scale = None
            self.film_shift = None

        # freq_ctx FiLM: per-frequency context [B, F, d_model] → per-channel scale/shift
        if freq_ctx:
            self.freq_ctx_film_scale = nn.Linear(d_model, 1)
            self.freq_ctx_film_shift = nn.Linear(d_model, 1)
            nn.init.zeros_(self.freq_ctx_film_scale.weight); nn.init.zeros_(self.freq_ctx_film_scale.bias)
            nn.init.zeros_(self.freq_ctx_film_shift.weight); nn.init.zeros_(self.freq_ctx_film_shift.bias)
        else:
            self.freq_ctx_film_scale = None
            self.freq_ctx_film_shift = None

        self.time_embedder = FourierEncoder(d_model)

        # Padding to make input size divisible by 4 (for 2 downsampling stages)
        target_size = math.ceil(input_size / 4) * 4
        total_pad = target_size - input_size
        pad_front = total_pad // 2
        pad_back = total_pad - pad_front
        self.padding_tuple = (pad_front, pad_back, pad_front, pad_back, pad_front, pad_back)
        self.crop_start = pad_front
        self.crop_end = pad_front + input_size

        # --- ENCODER PATH ---
        init_padding = 1 if init_kernel_size == 3 else 'same'
        self.init_conv = nn.Conv3d(in_channels, channels[0], kernel_size=init_kernel_size, padding=init_padding)

        self.enc1_res_attn = ResidualAttentionBlock3D(channels[0], channels[0], d_model, d_model, nhead)
        self.enc1_cross_attn = CrossAttentionBlock3D(channels[0], d_model, nhead)
        self.down1 = nn.MaxPool3d(2)

        self.enc2_res_attn = ResidualAttentionBlock3D(channels[0], channels[1], d_model, d_model, nhead)
        self.enc2_cross_attn = CrossAttentionBlock3D(channels[1], d_model, nhead)
        self.down2 = nn.MaxPool3d(2)

        # --- BOTTLENECK ---
        self.bottle_res_attn = ResidualAttentionBlock3D(channels[1], channels[2], d_model, d_model, nhead)
        self.bottle_cross_attn = CrossAttentionBlock3D(channels[2], d_model, nhead)

        # --- DECODER PATH ---
        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec1_res_attn = ResidualAttentionBlock3D(channels[1] * 2, channels[1], d_model, d_model, nhead)
        self.dec1_cross_attn = CrossAttentionBlock3D(channels[1], d_model, nhead)

        self.up2 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec2_res_attn = ResidualAttentionBlock3D(channels[0] * 2, channels[0], d_model, d_model, nhead)
        self.dec2_cross_attn = CrossAttentionBlock3D(channels[0], d_model, nhead)

        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x, t, context, context_mask, pooled_context=None, freq_contexts=None):
        B = x.size(0)
        x = F.pad(x, self.padding_tuple, mode='reflect')
        x = x.float()

        # Apply learnable per-frequency bias (if enabled)
        if self.freq_channel_bias is not None:
            x = x + self.freq_channel_bias

        # FiLM: observation-conditional per-frequency scale+shift from pooled_context
        if self.film_scale is not None and pooled_context is not None:
            scale = self.film_scale(pooled_context).view(B, -1, 1, 1, 1)
            shift = self.film_shift(pooled_context).view(B, -1, 1, 1, 1)
            x = x * (1 + scale) + shift

        # freq_ctx FiLM: per-frequency context [B, F, d_model] → per-channel scale/shift
        if self.freq_ctx_film_scale is not None and freq_contexts is not None:
            fc_scale = self.freq_ctx_film_scale(freq_contexts).squeeze(-1)  # [B, F]
            fc_shift = self.freq_ctx_film_shift(freq_contexts).squeeze(-1)  # [B, F]
            x = x * (1 + fc_scale[:, :, None, None, None]) + fc_shift[:, :, None, None, None]

        t_emb = self.time_embedder(t.squeeze())

        # --- Encoder ---
        h1 = self.init_conv(x)
        h1 = self.enc1_res_attn(h1, t_emb, pooled_context)
        h1 = self.enc1_cross_attn(h1, context, context_mask)

        h2 = self.down1(h1)
        h2 = self.enc2_res_attn(h2, t_emb, pooled_context)
        h2 = self.enc2_cross_attn(h2, context, context_mask)

        # --- Bottleneck ---
        bn = self.down2(h2)
        bn = self.bottle_res_attn(bn, t_emb, pooled_context)
        bn = self.bottle_cross_attn(bn, context, context_mask)

        # --- Decoder ---
        d1 = self.up1(bn)
        d1 = torch.cat((d1, h2), dim=1)
        d1 = self.dec1_res_attn(d1, t_emb, pooled_context)
        d1 = self.dec1_cross_attn(d1, context, context_mask)

        d2 = self.up2(d1)
        d2 = torch.cat((d2, h1), dim=1)
        d2 = self.dec2_res_attn(d2, t_emb, pooled_context)
        d2 = self.dec2_cross_attn(d2, context, context_mask)

        out = self.final_conv(d2)
        s, e = self.crop_start, self.crop_end
        return out[..., s:e, s:e, s:e]

class LSD(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data, target, freq_dim=1, data_type='atf_mag', mean=True):
        '''
        :param data:   (B,2,L,S) complex (or float) tensor or 1D array
        :param target: (B,2,L,S) complex (or float) tensor or 1D array
        :return: a scalar or (B,2,S) tensor
        '''
        #print(data.shape, target.shape)
        # Convert to numpy arrays if they're tensors
        # if hasattr(data, 'cpu'):
        #     data = data.cpu().numpy()
        # if hasattr(target, 'cpu'):
        #     target = target.cpu().numpy()

        # Handle dimension bounds checking
        data_arr = np.asarray(data)
        target_arr = np.asarray(target)
        dim = freq_dim
        # If dim is out of bounds, use the last available dimension
        if dim >= data_arr.ndim:
            # print(f"dim is out of bounds: {dim}, data_arr.ndim: {data_arr.ndim}")
            dim = data_arr.ndim - 1 if data_arr.ndim > 0 else 0
            # print("Using dim:", dim)

        # LSD = torch.sqrt(mean((data - target).pow(2), dim=dim))
        LSD = np.sqrt(np.mean((data - target)**2, axis=dim))
        if mean:
            LSD = np.mean(LSD)
        return LSD