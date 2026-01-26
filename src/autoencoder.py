# ──────────────────────────────────────────────────────────────────────────────
#  physics_informed_autoencoder.py
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt          # pip install PyWavelets
import numpy as np

class PhysicsInformedAutoencoder(nn.Module):
    """
    
    The forward pass returns the reconstructed signal.  A helper
    `compute_losses` method returns the three loss components
    (reconstruction, wavelet sparsity, temporal coherence) and their
    weighted sum.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the bottleneck.  16/64 is a good compromise.
    signal_length : int
        Length of the 1D PD signal (e.g. 11).
    wavelet : str, optional
        Wavelet basis for the sparsity penalty.  Default: 'db4'.
    lambda_wavelet : float, optional
        Weight for the wavelet sparsity penalty.
    lambda_temporal : float, optional
        Weight for the temporal coherence penalty.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        signal_length: int = 11,
        wavelet: str = "db4",
        lambda_wavelet: float = 0.01,
        lambda_temporal: float = 0.01,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.signal_length = signal_length
        self.wavelet = wavelet
        self.lambda_wavelet = lambda_wavelet
        self.lambda_temporal = lambda_temporal

        # ------------------------------------------------------------------
        # ENCODER
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Global average pool over the *signal* dimension
            nn.AdaptiveAvgPool1d(1),          # → (B, 128, 1)
            nn.Flatten(start_dim=1),          # → (B, 128)

            nn.Linear(128, latent_dim),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # DECODER
        # ------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * signal_length),  # Expand to (B, 128*L)
            nn.ReLU(inplace=True),

            nn.Unflatten(1, (128, signal_length)),  # → (B, 128, L)

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # Final reconstruction – linear activation
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            # No bias/activation because the reconstruction may contain
            # positive & negative values.
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, L).

        Returns
        -------
        recon : torch.Tensor
            Reconstructed signal, same shape as *x*.
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent vector back to signal space."""
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _wavelet_coeffs(signal: torch.Tensor, wavelet: str) -> torch.Tensor:
        """
        Compute the wavelet detail coefficients for each sample in *signal*.
        PyWavelets operates on NumPy arrays, so we perform the conversion
        sample by sample.  The result is a 2D tensor of shape (B, C),
        where C is the total number of detail coefficients.
        """
        B, _, L = signal.shape
        coeffs = []

        # Convert to CPU numpy for the transform 
        for i in range(B):
            try:
                coeff = pywt.wavedec(signal[i, 0, :].detach().cpu().numpy(),
                                     wavelet=wavelet,
                                     level=None)  # all levels
                detail_list = [c for c in coeff[1:]]
                # Handle case where no detail coefficients exist (short signals)
                if detail_list:
                    detail = np.concatenate(detail_list)
                else:
                    # For very short signals, use the signal itself as fallback
                    detail = signal[i, 0, :].detach().cpu().numpy()
                    
                coeffs.append(detail)
            except Exception as e:
                # Fallback: use signal as-is if wavelet decomposition fails
                coeffs.append(signal[i, 0, :].detach().cpu().numpy())

        # Pad to the same length (if needed)
        max_len = max(len(c) for c in coeffs)
        coeffs_padded = np.array([np.pad(c, (0, max_len - len(c))) for c in coeffs])
        return torch.tensor(coeffs_padded, dtype=signal.dtype, device=signal.device)

    def compute_wavelet_loss(self, recon: torch.Tensor) -> torch.Tensor:
        """
        L1 norm of the wavelet detail coefficients of the reconstruction.
        Encourages sparsity in the wavelet domain.
        """
        coeffs = self._wavelet_coeffs(recon, self.wavelet)   # (B, C)
        return torch.mean(torch.abs(coeffs))

    def compute_temporal_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Temporal coherence penalty: average L1 distance between
        consecutive latent vectors *within the batch*.
        Useful when the batch is a sliding window over a time-series.
        """
        if latent.size(0) < 2:
            return torch.tensor(0.0, device=latent.device)
        diff = latent[1:] - latent[:-1]
        return torch.mean(torch.abs(diff))

    def compute_losses(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        latent: torch.Tensor,
    ) -> dict:
        """
        Compute all three loss components and return a dictionary.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth signal (B, 1, L).
        recon : torch.Tensor
            Reconstructed signal (B, 1, L).
        latent : torch.Tensor
            Latent vector (B, latent_dim).

        Returns
        -------
        dict
            Keys: 'recon', 'wavelet', 'temporal', 'total'.
        """
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        wavelet_loss = self.compute_wavelet_loss(recon)
        temporal_loss = self.compute_temporal_loss(latent)

        total_loss = (
            recon_loss
            + self.lambda_wavelet * wavelet_loss
            + self.lambda_temporal * temporal_loss
        )

        return {
            "recon": recon_loss,
            "wavelet": wavelet_loss,
            "temporal": temporal_loss,
            "total": total_loss,
        }

# ----------------------------------------------------------------------
# Simple training loop skeleton (illustrative – not meant for production)
# ----------------------------------------------------------------------
def train_pi_ae(
    model: PhysicsInformedAutoencoder,
    dataloader,
    optimizer,
    device: torch.device = torch.device("cpu"),
    epochs: int = 20,
):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # batch is expected to be a tensor of shape (B, 1, L)
            batch = batch.to(device)

            optimizer.zero_grad()

            recon = model(batch)
            latent = model.encode(batch)

            losses = model.compute_losses(batch, recon, latent)
            loss = losses["total"]

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d}  "
            f"Loss: {epoch_loss:.6f}  "
            f"Recon: {losses['recon']:.6f}  "
            f"Wavelet: {losses['wavelet']:.6f}  "
            f"Temporal: {losses['temporal']:.6f}"
        )


