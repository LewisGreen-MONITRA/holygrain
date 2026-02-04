# ──────────────────────────────────────────────────────────────────────────────
#  physics_informed_autoencoder.py
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt     
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
        return recon, latent 

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
        Vectorized implementation: applies wavelet transform across the batch axis.
        The result is a 2D tensor of shape (B, C), where C is the total number 
        of detail coefficients.
        """
        B, _, L = signal.shape
        
        # Convert entire batch to numpy at once (single transfer)
        signal_np = signal[:, 0, :].detach().cpu().numpy()  # (B, L)
        
        try:
            # Vectorized wavelet decomposition across batch (axis=1 applies per-sample)
            coeffs = pywt.wavedec(signal_np, wavelet=wavelet, level=None, axis=1)
            
            # Concatenate all detail coefficients (exclude approximation coeffs[0])
            if len(coeffs) > 1:
                detail_coeffs = np.concatenate(coeffs[1:], axis=1)  # (B, C)
            else:
                # Fallback for very short signals
                detail_coeffs = signal_np
                
        except Exception as e:
            # Fallback: use signal as-is if wavelet decomposition fails
            detail_coeffs = signal_np
        
        return torch.tensor(detail_coeffs, dtype=signal.dtype, device=signal.device)

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
    optimiser,
    scheduler,
    device: torch.device = torch.device("cpu"),
    epochs: int = 20,
    patience = 3,
    min_delta: float = 1e-4
):
    model.to(device)
    model.train()

    best_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # batch is expected to be a tensor of shape (B, 1, L)
            batch = batch.to(device)
            # clear gradients
            optimiser.zero_grad(set_to_none=True)

            recon, latent = model(batch)
            
            losses = model.compute_losses(batch, recon, latent)
            loss = losses["total"]

            loss.backward()
            optimiser.step()
            scheduler.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d}  "
            f"Loss: {epoch_loss:.6f}  "
            f"Recon: {losses['recon']:.6f}  "
            f"Wavelet: {losses['wavelet']:.6f}  "
            f"Temporal: {losses['temporal']:.6f}"
        )

        # Early stopping check
        if epoch_loss < (best_loss - min_delta):
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch+1:02d}. "
                f"Best loss: {best_loss:.6f}"
            )
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(device)
            break


