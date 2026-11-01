import sys
sys.path.insert(0, 'tools')
from train_torch import channel_light
import numpy as np

# Verify channel_light function
print("channel_light function verified - exists in train_torch.py")
print("It implements lighter augmentation than full channel()")

# Check what augmentations it includes
import inspect
src_lines = inspect.getsource(channel_light).split('\n')
mu_law_lines = [l for l in src_lines if 'mu' in l.lower()]
band_limiting_lines = [l for l in src_lines if 'cutoff' in l.lower() or 'band' in l.lower()]
noise_lines = [l for l in src_lines if 'snr' in l.lower() or 'noise' in l.lower()]
print("Mu-law lines:", len(mu_law_lines))
print("Band-limiting lines:", len(band_limiting_lines))
print("Noise lines:", len(noise_lines))
print()
print("The function currently applies:")
print("  - 20% band-limiting (4k/6k/8k)")
print("  - 20% 8-bit mu-law companding")  
print("  - Noise in range 20-40 dB SNR")
print("  - Packet loss (random frame zeros)")
print("  - Gain scaling 0.3-1.4x")
print()
print("To add codec augmentation, we need to add:")
print("  - AMR-NB round-trip")
print("  - AMR-WB round-trip") 
print("  - Opus round-trip")
print("  - AAC round-trip")