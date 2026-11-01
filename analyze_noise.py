#!/usr/bin/env python3
"""Analyze background noise characteristics to help determine if audio is real or deepfake."""
import numpy as np
import soundfile as sf
import glob
import os


def analyze_background_noise(audio_path):
    """
    Analyze background noise characteristics to help determine if audio is real or deepfake.
    
    Returns:
        dict with noise features and a noise-based verdict
    """
    # Read audio
    data, rate = sf.read(audio_path, dtype='float32', always_2d=True)
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    
    n = len(data)
    
    # 1. Calculate noise floor (low-energy portion)
    fft_spec = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    sorted_magnitudes = np.sort(fft_spec)
    noise_floor = np.mean(sorted_magnitudes[:len(sorted_magnitudes)//10])
    
    # 2. Calculate signal-to-noise ratio
    signal_power = np.mean(data**2)
    noise_power = noise_floor**2 + 1e-12
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    # 3. Spectral centroid (brightness indicator)
    spectral_centroid = np.sum(freqs * fft_spec) / (np.sum(fft_spec) + 1e-12)
    
    # 4. Spectral rolloff (edge frequency - 85% of energy)
    total_energy = np.sum(fft_spec**2)
    cum_energy = np.cumsum(fft_spec**2)
    rolloff_idx = np.searchsorted(cum_energy, 0.85 * total_energy)
    spectral_rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]
    
    # 5. Zero crossing rate (rough noise indicator)
    zcr = np.mean(np.abs(np.diff(np.sign(data))) ) / 2.0
    
    # 6. High-frequency content ratio (important for TTS detection)
    high_freq_mask = freqs > 4000
    high_energy = np.sum(fft_spec[high_freq_mask]**2)
    total_energy_all = np.sum(fft_spec**2)
    high_freq_ratio = high_energy / (total_energy_all + 1e-12)
    
    # Noise-based verdict
    verdict = "unknown"
    confidence = 0.0
    
    # Heuristic rules based on noise characteristics
    if snr_db > 0 and snr_db < 30 and high_freq_ratio < 0.3:
        # Moderate SNR, limited high frequencies - could be real speech captured via phone
        verdict = "likely_real"
        confidence = min(0.9, 0.5 + abs(snr_db) / 60.0)
    elif snr_db > 40 or high_freq_ratio > 0.5:
        # High SNR or lots of high-frequency content - could be synthetic
        verdict = "likely_synthetic"
        confidence = min(0.9, 0.5 + (snr_db - 20) / 40.0 if snr_db > 20 else (high_freq_ratio - 0.3) / 0.3 * 0.4)
    else:
        # Ambiguous - rely on primary model
        verdict = "uncertain"
        confidence = 0.3
    
    return {
        'snr_db': round(snr_db, 2),
        'noise_floor': round(noise_floor, 6),
        'spectral_centroid': round(spectral_centroid, 2),
        'spectral_rolloff': round(spectral_rolloff, 2),
        'high_freq_ratio': round(high_freq_ratio, 4),
        'zcr': round(zcr, 4),
        'noise_verdict': verdict,
        'noise_confidence': round(confidence, 3),
        'n_samples': n,
        'sample_rate': rate
    }


# Quick test on available files
wav_files = glob.glob(os.path.join('data', 'asvspoof2019_la', '**', '*.flac'), recursive=True)
if wav_files:
    # Test first 3 files
    for wav_file in wav_files[:3]:
        result = analyze_background_noise(wav_file)
        print(f"\n=== {os.path.basename(wav_file)} ===")
        for key, value in result.items():
            if key not in ['n_samples', 'sample_rate']:
                print(f"  {key}: {value}")
else:
    print("No WAV files found in data directory")

# Also test with a synthetic test signal
print("\n\n=== Test with synthetic signal ===")
t = np.linspace(0, 1, 16000, endpoint=False)
# Real-like signal: voice-like with noise
real_signal = np.sin(2 * np.pi * 200 * t) * 0.5 + np.random.normal(0, 0.01, 16000)
result = analyze_background_noise_from_array(real_signal, 16000)
for key, value in result.items():
    if key not in ['n_samples']:
        print(f"  {key}: {value}")

def analyze_background_noise_from_array(audio_data, sample_rate):
    """Helper to analyze audio from numpy array."""
    data = np.asarray(audio_data, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data)
    
    fft_spec = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    sorted_magnitudes = np.sort(fft_spec)
    noise_floor = np.mean(sorted_magnitudes[:len(sorted_magnitudes)//10])
    
    signal_power = np.mean(data**2)
    noise_power = noise_floor**2 + 1e-12
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    spectral_centroid = np.sum(freqs * fft_spec) / (np.sum(fft_spec) + 1e-12)
    
    total_energy = np.sum(fft_spec**2)
    cum_energy = np.cumsum(fft_spec**2)
    rolloff_idx = np.searchsorted(cum_energy, 0.85 * total_energy)
    spectral_rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]
    
    zcr = np.mean(np.abs(np.diff(np.sign(data))) ) / 2.0
    
    high_freq_mask = freqs > 4000
    high_energy = np.sum(fft_spec[high_freq_mask]**2)
    total_energy_all = np.sum(fft_spec**2)
    high_freq_ratio = high_energy / (total_energy_all + 1e-12)
    
    verdict = "unknown"
    confidence = 0.0
    
    if snr_db > 0 and snr_db < 30 and high_freq_ratio < 0.3:
        verdict = "likely_real"
        confidence = min(0.9, 0.5 + abs(snr_db) / 60.0)
    elif snr_db > 40 or high_freq_ratio > 0.5:
        verdict = "likely_synthetic"
        confidence = min(0.9, 0.5 + (snr_db - 20) / 40.0 if snr_db > 20 else (high_freq_ratio - 0.3) / 0.3 * 0.4)
    else:
        verdict = "uncertain"
        confidence = 0.3
    
    return {
        'snr_db': round(snr_db, 2),
        'noise_floor': round(noise_floor, 6),
        'spectral_centroid': round(spectral_centroid, 2),
        'spectral_rolloff': round(spectral_rolloff, 2),
        'high_freq_ratio': round(high_freq_ratio, 4),
        'zcr': round(zcr, 4),
        'noise_verdict': verdict,
        'noise_confidence': round(confidence, 3),
        'n_samples': n,
        'sample_rate': sample_rate
    }