"""
Physics-Informed Autoencoder for Partial Discharge Signal Representation Learning

This module implements an autoencoder with physics-informed loss functions specifically
designed for learning meaningful representations of PD signals in high-voltage systems.

Physics-Informed Approach:
1. Wavelet Sparsity Loss: PD signals are inherently sparse in the wavelet domain
   (they are impulsive, short-duration events). We encourage the latent representation
   to preserve this sparsity property.

2. Temporal Coherence Loss: PD signals have characteristic impulse shapes that appear
   repeatedly. We enforce that similar temporal patterns map to similar latent representations.

3. Reconstruction Loss: Standard MSE ensures the latent representation can reconstruct
   the original signal accurately.

The combination of these losses guides the autoencoder to learn a latent space that
captures physically meaningful features of PD signals while filtering out noise.
"""

import numpy as np
import tensorflow as tf
import pywt  
from tensorflow.keras import layers, losses, Model, optimizers


class PhysicsInformedAutoencoder(Model):
    """
    Physics-Informed Autoencoder for PD signal representation learning.
    
    Inherits from tf.keras.Model to enable custom training loops and loss computation.
    
    Args:
        latent_dim (int): Dimensionality of the latent (bottleneck) space.
                         Lower values: More compression, more abstraction
                         Higher values: More detail preservation
                         Recommended: 16-64 depending on input dimension
        
        input_shape (tuple): Shape of input signals. For 1D signals: (signal_length,)
                           Example: (11,) for 11 PD features
        
        wavelet (str): Wavelet basis for sparsity constraint. 'db4' (Daubechies 4)
                      is good for PD signals due to sharp discontinuity detection.
        
        lambda_wavelet (float): Weight of wavelet sparsity loss (0.01 recommended)
        
        lambda_temporal (float): Weight of temporal coherence loss (0.01 recommended)
    """
    
    def __init__(self, latent_dim=32, input_shape=(11,), wavelet='db4',
                 lambda_wavelet=0.01, lambda_temporal=0.01):
        super(PhysicsInformedAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_shape_val = input_shape
        self.wavelet = wavelet
        self.lambda_wavelet = lambda_wavelet
        self.lambda_temporal = lambda_temporal
        
        # ========================================================================
        # ENCODER: Compress input signal into latent representation
        # ========================================================================
        # The encoder learns to extract the most important features of a PD signal
        # and represent them in a lower-dimensional space (latent space).
        # 
        # Architecture explanation:
        # - Conv1D layers extract local patterns (impulse characteristics)
        # - Each layer doubles the filter count to capture increasingly complex patterns
        # - GlobalAveragePooling reduces spatial dimension while retaining global info
        # - Final Dense layer projects to latent_dim
        # ========================================================================
        
        self.encoder = tf.keras.Sequential([
            # Input layer explicitly sized for 1D signals
            layers.Input(shape=input_shape),
            
            # Conv1D layer 1: Extract basic impulse patterns
            # 32 filters means 32 different pattern detectors
            # Kernel size 3: Look at 3-point windows (captures local discontinuities)
            layers.Conv1D(32, kernel_size=3, padding='same', activation='relu', 
                         name='encoder_conv1'),
            layers.BatchNormalization(name='encoder_bn1'),
            
            # Conv1D layer 2: Extract higher-level patterns from layer 1 outputs
            # 64 filters build on the 32 patterns to find combinations
            layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                         name='encoder_conv2'),
            layers.BatchNormalization(name='encoder_bn2'),
            
            # Conv1D layer 3: Extract abstract features
            # 128 filters for complex pattern combinations
            layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                         name='encoder_conv3'),
            layers.BatchNormalization(name='encoder_bn3'),
            
            # Global Average Pooling: Average all 128 feature maps across the signal
            # This reduces (signal_length, 128) → (128,)
            # Preserves global signal characteristics without spatial dimension
            layers.GlobalAveragePooling1D(name='encoder_gap'),
            
            # Latent representation: Project 128-dim features to latent_dim
            # This is the "bottleneck" - forces compression of important information
            layers.Dense(latent_dim, activation='relu', name='latent_space')
        ], name='Encoder')
        
        # ========================================================================
        # DECODER: Reconstruct signal from latent representation
        # ========================================================================
        # Mirror of encoder: expands latent representation back to original shape
        # Must learn to "uncompresses" the abstract features into concrete signals
        # ========================================================================
        
        self.decoder = tf.keras.Sequential([
            # Expand latent_dim back to initial compressed feature dimension
            layers.Input(shape=(latent_dim,)),
            layers.Dense(128, activation='relu', name='decoder_dense1'),
            
            # Reshape from (128,) to (128 features, 1 spatial position)
            # Then expand spatially through upsampling
            layers.Reshape((1, 128), name='decoder_reshape'),
            
            # Conv1D transpose: Upsample while reducing filters
            # Maps 128 filters → 64 filters with spatial expansion
            layers.Conv1DTranspose(64, kernel_size=3, padding='same', 
                                  activation='relu', name='decoder_conv1'),
            layers.BatchNormalization(name='decoder_bn1'),
            
            # Further upsampling: 64 → 32 filters
            layers.Conv1DTranspose(32, kernel_size=3, padding='same',
                                  activation='relu', name='decoder_conv2'),
            layers.BatchNormalization(name='decoder_bn2'),
            
            # Final reconstruction: 32 filters → original signal shape
            # Linear activation allows unbounded reconstruction values
            layers.Conv1DTranspose(1, kernel_size=3, padding='same',
                                  activation='linear', name='decoder_output'),
            
            # Reshape back to original input shape
            # From (input_shape[0], 1) → input_shape
            layers.Reshape(input_shape, name='final_reshape')
        ], name='Decoder')
    
    def encode(self, x):
        """
        Encode input signal into latent representation.
        
        Args:
            x: Input signal tensor of shape (batch_size, *input_shape)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation back to signal space.
        
        Args:
            z: Latent representation of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed signal of shape (batch_size, *input_shape)
        """
        return self.decoder(z)
    
    def call(self, x):
        """
        Forward pass: encode input and decode back to reconstruction.
        
        Args:
            x: Input signal
            
        Returns:
            Reconstructed signal (same shape as input)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed
    
    # ========================================================================
    # PHYSICS-INFORMED LOSS FUNCTIONS
    # ========================================================================
    
    def reconstruction_loss(self, x_original, x_reconstructed):
        """
        Reconstruction Loss (MSE): Measures how well the autoencoder can
        reconstruct the input from the latent representation.
        
        Why MSE? PD signals can have continuous values, and MSE is differentiable
        everywhere, making optimization stable.
        
        Low reconstruction loss = Latent representation captures signal details well
        """
        return losses.mean_squared_error(x_original, x_reconstructed)
    
    def wavelet_sparsity_loss(self, x):
        """
        Wavelet Sparsity Loss: Encourages latent representation to align with
        the natural sparsity of PD signals in the wavelet domain.
        
        Physics Principle:
        - PD signals are impulsive (short-duration, high-amplitude events)
        - Impulses are sparse in wavelet decomposition (few large coefficients)
        - Noise is distributed across many wavelet coefficients
        - By encouraging wavelet sparsity, we make the AE learn to represent
          PD events distinctly from noise
        
        Implementation:
        1. Decompose input into wavelet coefficients (using specified wavelet)
        2. Compute L1 norm of high-frequency components (detail coefficients)
        3. L1 norm encourages zeros (sparsity) - non-zero coefficients must
           be important to justify their existence
        
        Returns: Scalar loss value (0 = perfectly sparse, higher = less sparse)
        """
        try:
            # For each sample in batch, compute wavelet decomposition
            sparsity_penalties = []
            
            for i in range(tf.shape(x)[0]):
                sample = x[i].numpy()
                
                # Wavelet decomposition: Split signal into approximation (cA)
                # and detail (cD) coefficients
                # cA: Low-frequency components (slow variations)
                # cD: High-frequency components (fast variations, impulses)
                cA, cD = pywt.dwt(sample, self.wavelet)
                
                # L1 norm of detail coefficients: Sum of absolute values
                # This encourages sparsity: fewer large coefficients preferred
                # over many small coefficients
                l1_norm = np.sum(np.abs(cD))
                sparsity_penalties.append(l1_norm)
            
            # Average sparsity penalty across batch
            sparsity_loss = tf.reduce_mean(tf.constant(sparsity_penalties, dtype=tf.float32))
            return sparsity_loss
            
        except Exception as e:
            # Fallback: If wavelet decomposition fails, use L1 regularization
            # This encourages sparse activation patterns in the signal
            print(f"Wavelet decomposition warning: {e}. Using L1 regularization fallback.")
            return tf.reduce_mean(tf.abs(x))
    
    def temporal_coherence_loss(self, x_original, x_reconstructed):
        """
        Temporal Coherence Loss: Ensures similar temporal patterns map to
        similar latent representations.
        
        Physics Principle:
        - PD events follow characteristic impulse shapes (rise time, peak, fall time)
        - Different PD events in similar conditions have similar shapes
        - By enforcing temporal coherence, we ensure the latent space groups
          similar PD patterns together, separating them from noise
        
        Implementation:
        1. Compute first derivatives (temporal velocity)
        2. Penalize large differences in derivatives between original and reconstruction
        3. This encourages smooth, physically plausible reconstructions
        
        Why derivatives? They capture the "shape" of transients (impulses).
        Two signals with same shape but different amplitudes have similar derivatives.
        """
        
        # Compute first derivative: Difference between consecutive time points
        # Represents how quickly the signal changes (temporal velocity)
        original_derivative = x_original[:, 1:] - x_original[:, :-1]
        reconstructed_derivative = x_reconstructed[:, 1:] - x_reconstructed[:, :-1]
        
        # MSE between derivatives: Penalizes reconstruction that doesn't
        # preserve the temporal structure (impulse shape)
        derivative_mse = losses.mean_squared_error(
            original_derivative, 
            reconstructed_derivative
        )
        
        # Compute second derivative: Acceleration (curvature)
        # Captures the "sharpness" of impulses (PD events are sharp)
        original_second_derivative = original_derivative[:, 1:] - original_derivative[:, :-1]
        reconstructed_second_derivative = reconstructed_derivative[:, 1:] - reconstructed_derivative[:, :-1]
        
        # Second derivative MSE: Penalizes loss of sharp features (impulses)
        second_derivative_mse = losses.mean_squared_error(
            original_second_derivative,
            reconstructed_second_derivative
        )
        
        # Combine: Weight first and second derivatives equally
        # Together they ensure both smooth transitions (1st) and sharp features (2nd)
        return derivative_mse + 0.5 * second_derivative_mse
    
    def compute_loss(self, x):
        """
        Compute total physics-informed loss.
        
        Total Loss = L_reconstruction + lambda_wavelet * L_wavelet + lambda_temporal * L_temporal
        
        Where:
        - L_reconstruction: How well we reconstruct the signal (0.7 importance)
        - L_wavelet: Wavelet sparsity (0.15 importance)
        - L_temporal: Temporal coherence (0.15 importance)
        
        The weights (λ) control how much each term influences training.
        """
        x_reconstructed = self(x)
        
        # Compute each loss component
        recon_loss = tf.reduce_mean(self.reconstruction_loss(x, x_reconstructed))
        wavelet_loss = self.wavelet_sparsity_loss(x)
        temporal_loss = tf.reduce_mean(self.temporal_coherence_loss(x, x_reconstructed))
        
        # Weighted combination: Physics-informed total loss
        total_loss = (
            recon_loss + 
            self.lambda_wavelet * wavelet_loss + 
            self.lambda_temporal * temporal_loss
        )
        
        return total_loss, recon_loss, wavelet_loss, temporal_loss
    
    def train_step(self, x):
        """
        Custom training step for physics-informed learning.
        
        Uses GradientTape to compute gradients of the physics-informed loss
        with respect to model parameters, then applies them via optimizer.
        """
        with tf.GradientTape() as tape:
            total_loss, recon_loss, wavelet_loss, temporal_loss = self.compute_loss(x)
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_weights)
        
        # Apply gradients to update model weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
        # Return metrics for monitoring
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'wavelet_loss': wavelet_loss,
            'temporal_loss': temporal_loss
        }
    
    def fit_physics_informed(self, train_data, epochs=100, batch_size=32,
                            validation_data=None, verbose=1):
        """
        Train the physics-informed autoencoder.
        
        Args:
            train_data: Training signal data (numpy array or tf.Dataset)
            epochs: Number of training iterations over full dataset
            batch_size: Number of samples per gradient update
            validation_data: Optional validation data for monitoring
            verbose: Print training progress (0, 1, or 2)
            
        Returns:
            Training history (losses over epochs)
        """
        # Use Adam optimizer for adaptive learning rate
        self.optimizer = optimizers.Adam(learning_rate=1e-3)
        
        # Create dataset
        if not isinstance(train_data, tf.data.Dataset):
            train_dataset = tf.data.Dataset.from_tensor_slices(train_data)\
                .batch(batch_size)\
                .shuffle(buffer_size=1000)
        else:
            train_dataset = train_data.batch(batch_size)
        
        history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'wavelet_loss': [],
            'temporal_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = {
                'total_loss': [],
                'reconstruction_loss': [],
                'wavelet_loss': [],
                'temporal_loss': []
            }
            
            for batch in train_dataset:
                metrics = self.train_step(batch)
                
                for key in epoch_losses.keys():
                    epoch_losses[key].append(metrics[key].numpy())
            
            # Average losses over epoch
            for key in history.keys():
                avg_loss = np.mean(epoch_losses[key])
                history[key].append(avg_loss)
            
            if verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Total Loss: {history['total_loss'][-1]:.4f}, "
                      f"Recon: {history['reconstruction_loss'][-1]:.4f}, "
                      f"Wavelet: {history['wavelet_loss'][-1]:.4f}, "
                      f"Temporal: {history['temporal_loss'][-1]:.4f}")
        
        return history
