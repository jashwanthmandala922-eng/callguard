#!/usr/bin/env python3
"""Combine CNN deepfake detection with background noise analysis for real/deepfake labeling."""
import numpy as np
import soundfile as sf
import os
import glob


def analyze_background_noise(audio_path):
    """Analyze background noise characteristics to determine likely real vs deepfake."""
    data, rate = sf.read(audio_path, dtype='float32', always_2d=True)
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    n = len(data)
    
    fft_spec = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
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
        'noise_verdict': verdict,
        'noise_confidence': round(confidence, 3),
        'high_freq_ratio': round(high_freq_ratio, 4),
    }


# Find test WAV files
wav_files = glob.glob(os.path.join('data', 'asvspoof2019_la', '**', '*.flac'), recursive=True)[:5]

print("="*70)
print("COMBINED CNN + BACKGROUND NOISE ANALYSIS")
print("="*70)
print(f"{'File':<30} {'SNR (dB)':>10} {'Noise Verdict':>15} {'Combined Label':>15}")
print("="*70)

for wav_file in wav_files:
    noise = analyze_background_noise(wav_file)
    
    # Simulate CNN model verdict (in real app, this is TFLite inference)
    fname = os.path.basename(wav_file).lower()
    if 'real' in fname or 'bonafide' in fname:
        cnn_verdict = "real"
        cnn_confidence = 0.85
    elif 'fake' in fname or 'spoof' in fname:
        cnn_verdict = "deepfake"
        cnn_confidence = 0.82
    else:
        cnn_verdict = "uncertain"
        cnn_confidence = 0.55
    
    # Combine CNN + Noise analysis
    if noise['noise_verdict'] == 'likely_real' and cnn_verdict == 'real':
        combined = "real (confirmed)"
        combined_conf = min(0.95, cnn_confidence + 0.1)
    elif noise['noise_verdict'] == 'likely_synthetic' and cnn_verdict == 'deepfake':
        combined = "deepfake (confirmed)"
        combined_conf = min(0.95, cnn_confidence + 0.1)
    elif noise['noise_verdict'] == 'likely_real' and cnn_verdict == 'deepfake':
        combined = "conflict - verify conditions"
        combined_conf = 0.5
    elif noise['noise_verdict'] == 'likely_synthetic' and cnn_verdict == 'real':
        combined = "conflict - verify source"
        combined_conf = 0.5
    else:
        combined = "uncertain"
        combined_conf = 0.5
    
    print(f"{os.path.basename(wav_file):<30} {noise['snr_db']:>10.1f} {noise['noise_verdict']:>15} {combined:>15} (CNN:{cnn_verdict}, P(fake):{cnn_confidence:.2f})")

print()
print("="*70)
print("FEATURE SUMMARY")
print("="*70)
print("""
1. BACKGROUND NOISE ANALYSIS:
   - Computes SNR, spectral centroid, rolloff, high-frequency ratio
   - Provides independent verdict: likely_real / likely_synthetic / uncertain
   - O(1) computation: just FFT on the audio clip

2. COMBINATION STRATEGY:
   - AND: Both CNN + noise agree → higher confidence
   - CONFLICT: Disagreement → flag for manual review
   - EITHER uncertain → flag for manual review

3. ANDROID APP INTEGRATION:
   - Add noise analysis as lightweight pre/post-processing
   - Display combined verdict: "Real", "Deepfake", "Uncertain", "Conflict"
   - Show SNR and key noise features as diagnostic info
   - No additional model retraining needed

4. DIAGNOSTIC VALUE:
   - Helps explain why a clip was classified certain way
   - Detects far-field, noisy, or artifacted recordings
   - Identifies codec-induced artifacts vs. synthetic patterns
""")