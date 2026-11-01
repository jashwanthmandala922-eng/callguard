import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simplified AASIST-inspired architecture
# Key: attendance pooling, no problematic skip connections with stride mismatches
class AASISTLite(nn.Module):
    def __init__(self, input_channels=1, emb_dim=128, n_frequencies=64, n_frames=198):
        super(AASISTLite, self).__init__()

        # Stem conv
        self.stem = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Conv blocks (no stride in first two, stride in later ones)
        # With padding=1 and kernel=3, time dimension preserved when stride=1
        self.block1 = self._make_block(16, 16, stride=1)   # time: 198 -> 198
        self.block2 = self._make_block(16, 32, stride=2)   # time: 198 -> 99
        self.block3 = self._make_block(32, 32, stride=1)  # time: 99 -> 99
        self.block4 = self._make_block(32, 64, stride=2)  # time: 99 -> 49 (but with avgpool later)

        # Attendance pooling
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Classifier
        # After 2 strided blocks: 198 -> 99 -> 49, but with adaptive pooling we don't need exact size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 1)
        )

    def _make_block(self, in_ch, out_ch, stride=1):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if stride == 1:
            layers.append(nn.LeakyReLU(0.3))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, 64, 198)
        x = F.leaky_relu(self.bn1(self.stem(x)), 0.3)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Attendance pooling
        attention_map = self.attention(x)
        x = x * attention_map
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)
        return x, attention_map


# Quick test
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AASISTLite(input_channels=1, emb_dim=128).to(device)
model.eval()

# Create dummy input (1, 1, 64, 198) matching CallGuard: (batch, channel, freq, time)
dummy_mel = np.random.randn(64, 198).astype(np.float32)
dummy_tensor = torch.from_numpy(dummy_mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,198)

with torch.no_grad():
    output, attn = model(dummy_tensor)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"Output range: {output.min().item():.4f} to {output.max().item():.4f}")
    print("AASIST-lite: PASSED")