"""
AccEar: Accelerometer Acoustic Eavesdropping with Unconstrained Vocabulary
Complete PyTorch & Kaggle Implementation for Conditional GAN (cGAN) Speech Reconstruction
========================================================================================

Paper Implementation Specifications:
1. Signal Preprocessing:
   - Zero-mean normalization per axis: s_ij = (s_ij - mean(s_i)) / std(s_i)
   - 20 Hz High-pass Butterworth filter (removes low-freq body motion, preserves voice vibration)
   - Linear interpolation to 1 kHz constant sampling rate stream
   - 4-second fixed windowing + STFT computation (Hann window)
   - Square-root magnitude STFT (sqrt(|STFT|)) scaled & normalized to [0, 1] for IMU z-axis condition y
   - Audio Mel-spectrogram ground truth x using Mel formula: Mel(f) = 2595 * log10(1 + f / 700)

2. cGAN Architecture:
   - Generator (G): Symmetrical U-Net with skip connections (4x4 kernels, stride 2)
     Input: Concatenation of noise prior z and accelerometer spectrogram condition y
   - Discriminator (D): 30x30 PatchGAN Discriminator
     Input: Pair (x, y) real vs (x_hat, y) fake concatenated along channels

3. Loss Functions & Objective:
   - Spectrogram L1 Magnitude Loss: L_S = ||S(t, f) - S_p(t, f)||_1
   - Min-Max Adversarial cGAN Objective: V_cGAN(D, G)
   - Total Combined Loss: L* = lambda_L1 * L_S + L_adv

4. Training Loop & Constraints:
   - 200 Total Epochs
   - Epochs 1 to 100: Fixed lr = 0.0002 (Adam beta1=0.5, beta2=0.999)
   - Epochs 101 to 200: Linear decay of lr to 0
   - Algorithm 1 Updating Order: Step A (Update D given fixed G) -> Step B (Update G given fixed D)

5. Vocoder Audio Reconstruction:
   - Griffin-Lim algorithm (iSTFT phase estimation) to reconstruct human-audible .wav files
"""

import os
import sys
import math
import time
import glob
import random
import argparse
import numpy as np
import scipy.signal
import scipy.io.wavfile
import scipy.ndimage
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optional Torchaudio / Librosa imports with robust fallbacks
try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ==============================================================================
# 1. SIGNAL PREPROCESSING PIPELINE (ACCORDING TO ACCEAR PAPER)
# ==============================================================================

def zero_mean_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Step 1: Zero-Mean Normalization
    s_ij = (s_ij - mean(s_i)) / sigma
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal) + 1e-8
    return (signal - mean_val) / std_val


def highpass_filter_20hz(signal: np.ndarray, fs: float = 1000.0, order: int = 4) -> np.ndarray:
    """
    Step 2: 20 Hz High-Pass Butterworth Filter
    Eliminates low-frequency human movement artifacts (<20 Hz) while preserving
    voice-related high-frequency sensor vibrations.
    """
    nyquist = 0.5 * fs
    cutoff = 20.0 / nyquist
    if cutoff >= 1.0:
        cutoff = 0.99
    b, a = scipy.signal.butter(order, cutoff, btype='high', analog=False)
    # Zero-phase digital filtering via filtfilt
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    return filtered_signal


def linear_interpolate_1khz(raw_timestamps: np.ndarray, raw_signal: np.ndarray, duration_sec: float = 4.0) -> np.ndarray:
    """
    Step 3: Linear Interpolation
    Converts non-uniform raw sensor timestamps into a constant 1 kHz sampling rate stream.
    """
    target_num_samples = int(duration_sec * 1000) # 4000 samples for 4 seconds
    uniform_timestamps = np.linspace(0.0, duration_sec, target_num_samples, endpoint=False)
    
    # If timestamps missing, assume uniform range
    if raw_timestamps is None or len(raw_timestamps) != len(raw_signal):
        raw_timestamps = np.linspace(0.0, duration_sec, len(raw_signal), endpoint=False)

    interpolated_signal = np.interp(uniform_timestamps, raw_timestamps, raw_signal)
    return interpolated_signal


def compute_imu_stft_spectrogram(imu_signal_1khz: np.ndarray, 
                                 fs: int = 1000, 
                                 n_fft: int = 254, 
                                 hop_length: int = 31, 
                                 target_shape: tuple = (128, 128)) -> np.ndarray:
    """
    Step 4: Accelerometer STFT Spectrogram (y)
    - 4-second fixed windowing at 1 kHz (4000 samples)
    - Short-Time Fourier Transform (STFT) with Hann window
    - Square root of magnitude spectral features (sqrt(|STFT|))
    - Normalize to range [0, 1]
    - Resize/crop to target_shape (128, 128) for neural network input grid
    """
    # STFT using SciPy Hann window
    frequencies, times, Zxx = scipy.signal.stft(
        imu_signal_1khz, 
        fs=fs, 
        window='hann', 
        nperseg=n_fft, 
        noverlap=n_fft - hop_length, 
        boundary=None, 
        padded=False
    )
    
    # Square root magnitude feature extraction
    magnitude = np.abs(Zxx)
    sqrt_mag = np.sqrt(magnitude)
    
    # Min-max normalization to [0, 1]
    min_val = np.min(sqrt_mag)
    max_val = np.max(sqrt_mag) + 1e-8
    norm_mag = (sqrt_mag - min_val) / (max_val - min_val)
    
    # Resize to exact target 2D tensor shape (128, 128)
    h, w = norm_mag.shape
    target_h, target_w = target_shape
    
    # Zoom/interpolate to match exact spatial grid
    zoom_h = target_h / float(h)
    zoom_w = target_w / float(w)
    
    norm_mag_resized = scipy.ndimage.zoom(norm_mag, (zoom_h, zoom_w), order=1)
    # Clamp to [0, 1]
    norm_mag_resized = np.clip(norm_mag_resized[:target_h, :target_w], 0.0, 1.0)
    
    return norm_mag_resized.astype(np.float32)


def hz_to_mel(f_hz: np.ndarray) -> np.ndarray:
    """Paper formula: Mel(f) = 2595 * log10(1 + f / 700)"""
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Inverse Mel formula: f = 700 * (10^(mel / 2595) - 1)"""
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)


def compute_audio_mel_spectrogram(audio_pcm: np.ndarray, 
                                  fs: int = 16000, 
                                  n_fft: int = 1024, 
                                  hop_length: int = 500, 
                                  n_mels: int = 128, 
                                  target_shape: tuple = (128, 128)) -> tuple:
    """
    Step 5: Audio Mel-Spectrogram Ground Truth (x)
    Converts matching audio clips (16 kHz) into 2D Mel-spectrogram representations
    using standard Mel formula: Mel(f) = 2595 * log10(1 + f / 700)
    """
    if LIBROSA_AVAILABLE:
        # Standard librosa Mel spectrogram calculation
        mel_spec = librosa.feature.melspectrogram(
            y=audio_pcm, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fs//2
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize to [0, 1]
        norm_mel = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)
    else:
        # SciPy fallback implementation using explicit Mel filterbank matrix
        freqs, times, Zxx = scipy.signal.stft(audio_pcm, fs=fs, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length)
        power_spec = np.abs(Zxx) ** 2
        
        # Create Mel filterbank
        mel_min = hz_to_mel(0.0)
        mel_max = hz_to_mel(fs / 2.0)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)
        
        fbank = np.zeros((n_mels, int(n_fft // 2 + 1)))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (bin_points[m] - bin_points[m - 1]))
            for k in range(f_m, f_m_plus):
                if k < fbank.shape[1]:
                    fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (bin_points[m + 1] - bin_points[m]))
                    
        mel_spec = np.dot(fbank, power_spec)
        log_mel_spec = np.log10(mel_spec + 1e-8)
        norm_mel = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)

    # Resize to exact target shape (128, 128)
    h, w = norm_mel.shape
    target_h, target_w = target_shape
    norm_mel_resized = scipy.ndimage.zoom(norm_mel, (target_h / float(h), target_w / float(w)), order=1)
    norm_mel_resized = np.clip(norm_mel_resized[:target_h, :target_w], 0.0, 1.0)

    return norm_mel_resized.astype(np.float32)


# ==============================================================================
# 2. cGAN ARCHITECTURE SPECIFICATIONS (PYTORCH)
# ==============================================================================

class UNetGenerator(nn.Module):
    """
    Generator (G): Symmetrical U-Net with Skip Connections
    Input: Concatenation of noise prior z and accelerometer spectrogram condition y (2 channels total)
    Encoder: Convolutional layers with 4x4 square kernels and stride 2.
    Decoder: Transposed convolutional layers with skip connections.
    Output: Predicted audio Mel-spectrogram x_hat = G(z|y) with Sigmoid activation [0, 1].
    """
    def __init__(self, in_channels=2, out_channels=1, features=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder (Downsampling)
        # 128x128 -> 64x64
        self.enc1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)
        # 64x64 -> 32x32
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2)
        )
        # 32x32 -> 16x16
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4)
        )
        # 16x16 -> 8x8
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8)
        )
        # 8x8 -> 4x4 (Bottleneck)
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Upsampling with Skip Connections)
        # 4x4 -> 8x8
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        # 8x8 -> 16x16 (Concat skip enc4: features*8 + features*8 = features*16)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(features * 16, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        # 16x16 -> 32x32 (Concat skip enc3: features*4 + features*4 = features*8)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True)
        )
        # 32x32 -> 64x64 (Concat skip enc2: features*2 + features*2 = features*4)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        # 64x64 -> 128x128 (Concat skip enc1: features + features = features*2)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Bounded output in [0, 1] for Mel-spectrogram
        )

    def forward(self, z, y):
        # Concatenate noise prior z and IMU condition y along channel dimension
        x_in = torch.cat([z, y], dim=1) # Shape: (B, 2, 128, 128)
        
        # Encoder pass
        e1 = self.enc1(x_in)  # 64x64
        e2 = self.enc2(e1)    # 32x32
        e3 = self.enc3(e2)    # 16x16
        e4 = self.enc4(e3)    # 8x8
        b = self.bottleneck(e4) # 4x4
        
        # Decoder pass with skip connections
        d4 = self.dec4(b)
        d4_cat = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2_cat)
        d1_cat = torch.cat([d1, e1], dim=1)
        
        x_hat = self.final_up(d1_cat) # Shape: (B, 1, 128, 128)
        return x_hat


class PatchGANDiscriminator(nn.Module):
    """
    Discriminator (D): 30x30 PatchGAN Discriminator
    Input: Pair of (x, y) [Real Mel-spec + IMU condition] or (x_hat, y) [Generated Mel-spec + IMU condition]
    Architecture: Convolutional layers classifying 30x30 local image patches as Real (1) or Fake (0).
    """
    def __init__(self, in_channels=2, features=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # Layer 1: 128x128 -> 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 2: 64x64 -> 32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 3: 32x32 -> 31x31 (stride 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Patch Output Layer: 31x31 -> 30x30 patch logits
        self.final_patch = nn.Conv2d(features * 4, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x_or_xhat, y):
        # Concatenate audio Mel-spectrogram and IMU condition
        pair = torch.cat([x_or_xhat, y], dim=1) # Shape: (B, 2, 128, 128)
        
        c1 = self.conv1(pair)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        patch_logits = self.final_patch(c3) # Output shape: (B, 1, 30, 30)
        return patch_logits


# ==============================================================================
# 3. LOSS FUNCTIONS & OBJECTIVES
# ==============================================================================

class AccEarLoss(nn.Module):
    """
    AccEar Objective Function L*:
    1. Spectrogram L1 Magnitude Loss: L_S = ||S(t, f) - S_p(t, f)||_1
    2. Min-Max Adversarial cGAN Objective: V_cGAN(D, G)
    3. Combined Loss: L* = lambda_L1 * L_S + L_adv
    """
    def __init__(self, lambda_l1=100.0):
        super(AccEarLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, d_real_logits, d_fake_logits):
        # Real loss: target 1.0
        real_targets = torch.ones_like(d_real_logits)
        loss_real = self.bce_loss(d_real_logits, real_targets)
        
        # Fake loss: target 0.0
        fake_targets = torch.zeros_like(d_fake_logits)
        loss_fake = self.bce_loss(d_fake_logits, fake_targets)
        
        d_loss = (loss_real + loss_fake) * 0.5
        return d_loss

    def generator_loss(self, d_fake_logits, x_real, x_fake):
        # Adversarial loss: G wants D to classify x_fake as real (1.0)
        real_targets = torch.ones_like(d_fake_logits)
        loss_adv = self.bce_loss(d_fake_logits, real_targets)
        
        # Spectrogram L1 magnitude loss L_S
        loss_l1 = self.l1_loss(x_fake, x_real)
        
        # Combined L*
        total_g_loss = loss_adv + (self.lambda_l1 * loss_l1)
        return total_g_loss, loss_adv, loss_l1


# ==============================================================================
# 4. DATASET & DATA GENERATOR (IMU + SPEECH COMMANDS)
# ==============================================================================

class AccEarDataset(Dataset):
    """
    Dataset loader for paired Accelerometer z-axis IMU signals and Speech Audio Mel-spectrograms.
    Supports reading from local StealthyIMU dataset directory or fallback synthetic generation.
    Includes in-memory spectrogram caching for ultra-fast epoch iteration.
    """
    def __init__(self, imu_dir="./StealthyIMU_dataset", num_samples=500, mode="train", use_cache=True):
        super(AccEarDataset, self).__init__()
        self.mode = mode
        self.samples = []
        self.use_cache = use_cache
        self.cache = {}
        
        # Look for local IMU files if present
        data_path = os.path.join(imu_dir, "data")
        found_files = []
        if os.path.exists(data_path):
            found_files = glob.glob(os.path.join(data_path, "**", "*.accnpy"), recursive=True)
            
        if len(found_files) > 0:
            if num_samples is not None and num_samples > 0:
                found_files = found_files[:num_samples]
            print(f"[{mode.upper()}] Found {len(found_files)} real IMU files in {data_path}")
            self.samples = found_files
        else:
            print(f"[{mode.upper()}] Operating with Speech Commands dataset / IMU sensor dataset simulator.")
            self.samples = list(range(num_samples if num_samples else 1000))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        item = self.samples[idx]
        
        if isinstance(item, str) and os.path.exists(item):
            # Load real IMU accnpy file
            acc_data = np.load(item) # Shape (4, N) or (N, 3)
            if acc_data.ndim == 2 and acc_data.shape[0] >= 3:
                # Extract z-axis (index 2 or 3 as per paper analysis)
                z_axis = acc_data[2, :] if acc_data.shape[0] < acc_data.shape[1] else acc_data[:, 2]
            else:
                z_axis = acc_data.flatten()
                
            timestamps = np.linspace(0, 4.0, len(z_axis))
            
            # Load corresponding wav if available
            wav_path = item.replace(".accnpy", ".wav")
            if os.path.exists(wav_path):
                sr, audio_pcm = scipy.io.wavfile.read(wav_path)
                audio_pcm = audio_pcm.astype(np.float32)
                if np.max(np.abs(audio_pcm)) > 0:
                    audio_pcm = audio_pcm / np.max(np.abs(audio_pcm))
            else:
                # Synthetic audio match if wav missing
                t_audio = np.linspace(0, 4.0, 64000)
                audio_pcm = np.sin(2 * np.pi * 440 * t_audio) * np.exp(-t_audio)
        else:
            # Generate realistic synthetic Speech/IMU pairs for robust self-contained Kaggle execution
            np.random.seed(idx + (10000 if self.mode == "test" else 0))
            t_imu = np.linspace(0, 4.0, 3200) # Raw 800Hz sensor stream
            freq1 = np.random.uniform(100, 300)
            z_axis = np.sin(2 * np.pi * freq1 * t_imu) + 0.5 * np.random.randn(len(t_imu))
            timestamps = t_imu
            
            # Synthetic 16kHz audio matching segment
            t_audio = np.linspace(0, 4.0, 64000)
            audio_pcm = np.sin(2 * np.pi * freq1 * t_audio) * np.exp(-0.5 * t_audio)
            audio_pcm = audio_pcm + 0.1 * np.random.randn(len(t_audio))
            audio_pcm = audio_pcm / (np.max(np.abs(audio_pcm)) + 1e-8)

        # ----------------------------------------------------------------------
        # Execute Exact Paper Preprocessing Pipeline
        # ----------------------------------------------------------------------
        # 1. Zero-Mean Normalization
        s_norm = zero_mean_normalize(z_axis)
        
        # 2. 20 Hz High-Pass Filtering
        s_hp = highpass_filter_20hz(s_norm, fs=1000.0)
        
        # 3. Linear Interpolation to 1 kHz
        s_1khz = linear_interpolate_1khz(timestamps, s_hp, duration_sec=4.0)
        
        # 4. Accelerometer STFT Spectrogram (y)
        y_spec = compute_imu_stft_spectrogram(s_1khz, fs=1000, target_shape=(128, 128))
        
        # 5. Audio Mel-Spectrogram Ground Truth (x)
        x_spec = compute_audio_mel_spectrogram(audio_pcm, fs=16000, target_shape=(128, 128))

        # Return Tensors with shape (1, 128, 128)
        y_tensor = torch.from_numpy(y_spec).unsqueeze(0)
        x_tensor = torch.from_numpy(x_spec).unsqueeze(0)

        item_tuple = (y_tensor, x_tensor)
        if self.use_cache:
            self.cache[idx] = item_tuple

        return item_tuple


# ==============================================================================
# 5. INFERENCE & VOCODER RECONSTRUCTION (GRIFFIN-LIM)
# ==============================================================================

def reconstruct_audio_griffin_lim(mel_spec: np.ndarray, 
                                  sr: int = 16000, 
                                  n_fft: int = 1024, 
                                  hop_length: int = 500, 
                                  n_iter: int = 64) -> np.ndarray:
    """
    Synthesis function taking Generator output Mel-spectrogram [128, 128]
    and reconstructing human-audible .wav PCM audio files using Griffin-Lim algorithm.
    """
    if LIBROSA_AVAILABLE:
        # Denormalize dB power scale
        db_mel = mel_spec * 80.0 - 80.0
        power_mel = librosa.db_to_power(db_mel)
        # Convert Mel to STFT linear magnitude
        stft_mag = librosa.feature.inverse.mel_to_stft(power_mel, sr=sr, n_fft=n_fft)
        # Griffin-Lim phase retrieval
        audio_reconstructed = librosa.griffinlim(stft_mag, n_iter=n_iter, hop_length=hop_length)
    else:
        # Standalone Griffin-Lim phase estimation algorithm implementation
        # Map 128 mel bins back to linear STFT bins (n_fft//2 + 1 = 513)
        n_stft_bins = n_fft // 2 + 1
        mel_min = hz_to_mel(0.0)
        mel_max = hz_to_mel(sr / 2.0)
        mel_points = np.linspace(mel_min, mel_max, mel_spec.shape[0] + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        # Pseudo-inverse of filterbank
        stft_mag = np.zeros((n_stft_bins, mel_spec.shape[1]))
        for i in range(mel_spec.shape[0]):
            b_start = bin_points[i]
            b_end = bin_points[i+1]
            stft_mag[b_start:b_end, :] = mel_spec[i, :]

        # Iterative Griffin-Lim
        angles = np.exp(2j * np.pi * np.random.rand(*stft_mag.shape))
        complex_spec = stft_mag * angles
        
        for _ in range(n_iter):
            _, audio_reconstructed = scipy.signal.istft(complex_spec, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            _, _, complex_spec_new = scipy.signal.stft(audio_reconstructed, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            angles = np.exp(1j * np.angle(complex_spec_new[:n_stft_bins, :stft_mag.shape[1]]))
            complex_spec = stft_mag * angles
            
        _, audio_reconstructed = scipy.signal.istft(complex_spec, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Normalize amplitude
    max_amp = np.max(np.abs(audio_reconstructed)) + 1e-8
    audio_reconstructed = audio_reconstructed / max_amp
    return audio_reconstructed


# ==============================================================================
# 6. KAGGLE TRAINING LOOP & OPTIMIZATION SCHEDULER
# ==============================================================================

def get_learning_rate(epoch: int, total_epochs: int = 200, initial_lr: float = 0.0002) -> float:
    """
    Epochs 1 to 100: Fixed lr = 0.0002
    Epochs 101 to 200: Linear decay from 0.0002 to 0.0
    """
    if epoch <= 100:
        return initial_lr
    else:
        decay_factor = 1.0 - ((epoch - 100) / float(total_epochs - 100))
        return initial_lr * max(0.0, decay_factor)


def train_acc_ear_cgan(args):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Kaggle Multi-GPU T4 x2 Detection & Performance Optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"=== AccEar cGAN Training Execution ===")
    print(f"Active Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {num_gpus} x {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Output Checkpoints directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Multi-threaded Data Loader (pin_memory=False to prevent Kaggle IPC thread crashes)
    dataset = AccEarDataset(imu_dir=args.dataset_dir, num_samples=args.num_samples, mode="train")
    num_workers = 2 if os.name != 'nt' else 0
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

    # Initialize Neural Network Models
    generator_base = UNetGenerator(in_channels=2, out_channels=1, features=64).to(device)
    discriminator_base = PatchGANDiscriminator(in_channels=2, features=64).to(device)

    # Wrap models for Multi-GPU DataParallel execution if multiple GPUs are detected
    if num_gpus > 1:
        print(f"Enabling PyTorch DataParallel across {num_gpus} GPUs for maximum throughput...")
        generator = nn.DataParallel(generator_base)
        discriminator = nn.DataParallel(discriminator_base)
    else:
        generator = generator_base
        discriminator = discriminator_base

    # Initialize Optimizers (Adam with beta1=0.5, beta2=0.999 as per paper)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Loss Module
    criterion = AccEarLoss(lambda_l1=args.lambda_l1).to(device)

    # Mixed Precision AMP Scalers for NVIDIA T4 Tensor Core acceleration
    use_amp = torch.cuda.is_available()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler_G = torch.amp.GradScaler(device_type, enabled=use_amp)
        scaler_D = torch.amp.GradScaler(device_type, enabled=use_amp)
    else:
        scaler_G = torch.cuda.amp.GradScaler(enabled=use_amp)
        scaler_D = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_l1_loss = float('inf')
    start_epoch = 1

    # Checkpoint Resumption Logic
    if hasattr(args, "resume") and args.resume:
        resume_path = args.resume
        if resume_path.lower() == "auto":
            ckpts = glob.glob(os.path.join(args.checkpoint_dir, "accear_cgan_epoch_*.pt"))
            if not ckpts:
                ckpts = glob.glob("/kaggle/input/**/accear_cgan_epoch_*.pt", recursive=True)
            if not ckpts:
                ckpts = glob.glob("/kaggle/input/**/accear_cgan_best_model.pt", recursive=True)
            if ckpts:
                def get_ep_num(p):
                    b = os.path.basename(p).replace(".pt", "")
                    for item in b.split("_"):
                        if item.isdigit():
                            return int(item)
                    return 0
                ckpts = sorted(ckpts, key=get_ep_num)
                resume_path = ckpts[-1]
        
        if os.path.exists(resume_path):
            print(f"--- Resuming Training from Checkpoint: {resume_path} ---")
            ckpt = torch.load(resume_path, map_location=device)
            if "generator" in ckpt:
                generator_base.load_state_dict(ckpt["generator"])
            if "discriminator" in ckpt:
                discriminator_base.load_state_dict(ckpt["discriminator"])
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            if "best_l1_loss" in ckpt:
                best_l1_loss = ckpt["best_l1_loss"]
            print(f"Successfully loaded checkpoint weights. Continuing training from Epoch {start_epoch}...")
        else:
            print(f"Warning: Specified resume checkpoint not found: {resume_path}. Starting fresh from Epoch 1.")

    start_time = time.time()
    
    print(f"Starting Training for Epochs {start_epoch} -> {args.epochs} (AMP Enabled: {use_amp}, Batch Size: {args.batch_size})...")
    for epoch in range(start_epoch, args.epochs + 1):
        # Update Learning Rate according to schedule
        current_lr = get_learning_rate(epoch, total_epochs=args.epochs, initial_lr=args.lr)
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = current_lr
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = current_lr

        generator.train()
        discriminator.train()
        
        running_d_loss = 0.0
        running_g_loss = 0.0
        running_l1_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d}/{args.epochs:03d} (lr={current_lr:.6f})", leave=False)
        for y_condition, x_real in pbar:
            y_condition = y_condition.to(device, non_blocking=True) # Shape: (B, 1, 128, 128)
            x_real = x_real.to(device, non_blocking=True)           # Shape: (B, 1, 128, 128)
            batch_size = y_condition.size(0)

            # Sample Gaussian noise prior z matching spatial dimensions
            z_noise = torch.randn_like(y_condition).to(device, non_blocking=True)

            # ------------------------------------------------------------------
            # Step A: Update Discriminator parameters theta_D (Fix Generator theta_G)
            # ------------------------------------------------------------------
            optimizer_D.zero_grad(set_to_none=True)
            
            autocast_ctx = torch.amp.autocast(device_type, enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                with torch.no_grad():
                    x_fake = generator(z_noise, y_condition)

                d_real_logits = discriminator(x_real, y_condition)
                d_fake_logits = discriminator(x_fake, y_condition)
                d_loss = criterion.discriminator_loss(d_real_logits, d_fake_logits)

            scaler_D.scale(d_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            # ------------------------------------------------------------------
            # Step B: Update Generator parameters theta_G (Fix Discriminator theta_D)
            # ------------------------------------------------------------------
            optimizer_G.zero_grad(set_to_none=True)

            with autocast_ctx:
                x_fake = generator(z_noise, y_condition)
                d_fake_logits_for_g = discriminator(x_fake, y_condition)
                g_loss, g_adv, g_l1 = criterion.generator_loss(d_fake_logits_for_g, x_real, x_fake)

            scaler_G.scale(g_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # Accumulate metrics
            running_d_loss += d_loss.item() * batch_size
            running_g_loss += g_loss.item() * batch_size
            running_l1_loss += g_l1.item() * batch_size

            pbar.set_postfix({
                'D_Loss': f"{d_loss.item():.4f}",
                'G_Loss': f"{g_loss.item():.4f}",
                'L1_Spec': f"{g_l1.item():.4f}"
            })

        epoch_d_loss = running_d_loss / len(dataset)
        epoch_g_loss = running_g_loss / len(dataset)
        epoch_l1_loss = running_l1_loss / len(dataset)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] - D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f} | Spectrogram L1: {epoch_l1_loss:.4f}")

        # ----------------------------------------------------------------------
        # Checkpoint Saving: Every Epoch + Best Model Tracking
        # ----------------------------------------------------------------------
        gen_state = generator.module.state_dict() if hasattr(generator, "module") else generator.state_dict()
        disc_state = discriminator.module.state_dict() if hasattr(discriminator, "module") else discriminator.state_dict()

        # Save checkpoint after EVERY epoch
        ckpt_path = os.path.join(args.checkpoint_dir, f"accear_cgan_epoch_{epoch:03d}.pt")
        torch.save({
            'epoch': epoch,
            'generator_state_dict': gen_state,
            'discriminator_state_dict': disc_state,
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'g_loss': epoch_d_loss,
            'd_loss': epoch_g_loss,
            'l1_spec_loss': epoch_l1_loss
        }, ckpt_path)

        # Track and Save Best Model based on Lowest Spectrogram L1 Loss
        if epoch_l1_loss < best_l1_loss:
            best_l1_loss = epoch_l1_loss
            best_ckpt_path = os.path.join(args.checkpoint_dir, "accear_cgan_best_model.pt")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': gen_state,
                'discriminator_state_dict': disc_state,
                'best_l1_loss': best_l1_loss,
                'g_loss': epoch_g_loss,
                'd_loss': epoch_d_loss
            }, best_ckpt_path)
            print(f" [BEST MODEL UPDATED] Epoch {epoch:03d} -> Best Spectrogram L1: {best_l1_loss:.4f} Saved: {best_ckpt_path}")

            # Generate Best Reconstructed Audio Sample
            generator.eval()
            with torch.no_grad():
                sample_y, sample_x = dataset[0]
                sample_y = sample_y.unsqueeze(0).to(device)
                sample_z = torch.randn_like(sample_y).to(device)
                pred_mel_tensor = generator_base(sample_z, sample_y).squeeze(0).squeeze(0).cpu().numpy()
                rec_wav = reconstruct_audio_griffin_lim(pred_mel_tensor, sr=16000, n_iter=32)
                best_wav_path = os.path.join(args.checkpoint_dir, "accear_cgan_best_reconstructed_sample.wav")
                scipy.io.wavfile.write(best_wav_path, 16000, (rec_wav * 32767).astype(np.int16))

        # Periodic Audio Synthesis Every 20 Epochs
        if epoch % 20 == 0 or epoch == args.epochs:
            generator.eval()
            with torch.no_grad():
                sample_y, sample_x = dataset[0]
                sample_y = sample_y.unsqueeze(0).to(device)
                sample_z = torch.randn_like(sample_y).to(device)
                pred_mel_tensor = generator_base(sample_z, sample_y).squeeze(0).squeeze(0).cpu().numpy()
                rec_wav = reconstruct_audio_griffin_lim(pred_mel_tensor, sr=16000, n_iter=32)
                wav_out_path = os.path.join(args.checkpoint_dir, f"sample_reconstruction_epoch_{epoch:03d}.wav")
                scipy.io.wavfile.write(wav_out_path, 16000, (rec_wav * 32767).astype(np.int16))
                print(f" Reconstructed audio sample saved: {wav_out_path}")

    total_time = time.time() - start_time
    print(f"\nTraining Complete! Total Time Elapsed: {total_time/60.0:.2f} minutes.")
    print(f"Best Spectrogram L1 Loss Achieved: {best_l1_loss:.4f} (Saved to {os.path.join(args.checkpoint_dir, 'accear_cgan_best_model.pt')})")


def evaluate_best_model(checkpoint_dir="./checkpoints", dataset_dir="./StealthyIMU_dataset"):
    """
    Evaluates all saved checkpoints in checkpoint_dir, finds the best model checkpoint,
    and synthesizes human-audible audio waveform .wav samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = os.path.join(checkpoint_dir, "accear_cgan_best_model.pt")
    if not os.path.exists(best_path):
        all_ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if not all_ckpts:
            print(f"No checkpoint files found in {checkpoint_dir}")
            return
        best_path = sorted(all_ckpts)[-1]
        
    print(f"=== AccEar Model Evaluation ===")
    print(f"Loading Best Model Checkpoint: {best_path}")
    checkpoint = torch.load(best_path, map_location=device)
    
    generator = UNetGenerator(in_channels=2, out_channels=1, features=64).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    dataset = AccEarDataset(imu_dir=dataset_dir, num_samples=10, mode="test")
    sample_y, sample_x = dataset[0]
    sample_y = sample_y.unsqueeze(0).to(device)
    sample_z = torch.randn_like(sample_y).to(device)
    
    with torch.no_grad():
        pred_mel = generator(sample_z, sample_y).squeeze().cpu().numpy()
        
    rec_wav = reconstruct_audio_griffin_lim(pred_mel, sr=16000, n_iter=64)
    eval_wav_path = os.path.join(checkpoint_dir, "evaluated_best_reconstruction.wav")
    scipy.io.wavfile.write(eval_wav_path, 16000, (rec_wav * 32767).astype(np.int16))
    
    print(f"Evaluation Complete!")
    print(f"Best Epoch: {checkpoint.get('epoch', 'N/A')}")
    loss_val = checkpoint.get('best_l1_loss', checkpoint.get('l1_spec_loss', None))
    loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else "N/A"
    print(f"Best L1 Loss: {loss_str}")
    print(f"Reconstructed Wav Saved: {eval_wav_path}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AccEar cGAN Speech Reconstruction Training & Evaluation Script")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (Default: 200)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (Default: 32)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Initial learning rate (Default: 0.0002)")
    parser.add_argument("--lambda-l1", type=float, default=100.0, help="L1 Spectrogram loss weight (Default: 100.0)")
    parser.add_argument("--dataset-dir", type=str, default="./StealthyIMU_dataset", help="IMU dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Output directory for checkpoints and wav samples")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to load (Default: all)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file or 'auto' to resume training")
    parser.add_argument("--test-run", action="store_true", help="Quick verification run with 2 epochs")
    parser.add_argument("--eval-best", action="store_true", help="Evaluate best saved model checkpoint and reconstruct audio")

    args = parser.parse_args()

    if args.eval_best:
        evaluate_best_model(checkpoint_dir=args.checkpoint_dir, dataset_dir=args.dataset_dir)
    else:
        if args.test_run:
            print("--- Executing Quick Verification Test Run ---")
            args.epochs = 2
            args.batch_size = 4
            args.num_samples = 20

        train_acc_ear_cgan(args)

