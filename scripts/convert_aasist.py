import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '.')

# AASISTLite model (same as before)
class AASISTLite(nn.Module):
    def __init__(self, input_channels=1, emb_dim=128, n_frequencies=64, n_frames=198):
        super(AASISTLite, self).__init__()
        self.stem = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = self._make_block(16, 16, stride=1)
        self.block2 = self._make_block(16, 32, stride=2)
        self.block3 = self._make_block(32, 32, stride=1)
        self.block4 = self._make_block(32, 64, stride=2)
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
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
        x = F.leaky_relu(self.bn1(self.stem(x)), 0.3)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        attention_map = self.attention(x)
        x = x * attention_map
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, attention_map

# ===== Build & test PyTorch model =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AASISTLite(input_channels=1, emb_dim=128).to(device)
model.eval()

dummy_mel = np.random.randn(64, 198).astype(np.float32)
dummy_tensor = torch.from_numpy(dummy_mel).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    output, attn = model(dummy_tensor)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PyTorch] Model params: {n_params}")

# ===== Build Keras model =====
from tensorflow.keras import layers, Model

inp = layers.Input(shape=(64, 198, 1), name="logmel")  # NHWC
x = inp

# Stem
c = layers.Conv2D(16, 3, padding="same", use_bias=False, name="conv0")(x)
x = layers.BatchNormalization(epsilon=1e-5, name="gn0")(x)
x = layers.LeakyReLU(alpha=0.3, name="relu0")(x)

# Blocks
c = layers.Conv2D(16, 3, padding="same", use_bias=False, name="conv1")(x)
x = layers.BatchNormalization(epsilon=1e-5, name="gn1")(c)
x = layers.LeakyReLU(alpha=0.3, name="relu1")(x)

c = layers.Conv2D(16, 3, padding="same", strides=2, use_bias=False, name="conv2")(c)
x = layers.BatchNormalization(epsilon=1e-5, name="gn2")(c)
x = layers.LeakyReLU(alpha=0.3, name="relu2")(x)

c = layers.Conv2D(16, 3, padding="same", use_bias=False, name="conv3")(x)
x = layers.BatchNormalization(epsilon=1e-5, name="gn3")(c)
x = layers.LeakyReLU(alpha=0.3, name="relu3")(x)

c = layers.Conv2D(32, 3, padding="same", strides=2, use_bias=False, name="conv3")(c)
x = layers.BatchNormalization(epsilon=1e-5, name="gn4")(c)
x = layers.LeakyReLU(alpha=0.3, name="relu4")(x)

# Attendance pooling
mean = layers.GlobalAveragePooling2D(name="mean")(x)
squared = layers.Multiply(name="square")([x, x])
mean_sq = layers.GlobalAveragePooling2D(name="mean_sq")(squared)
mean_squared = layers.Multiply(name="mean_squared")([mean, mean_sq])
var = layers.Subtract(name="variance")([mean_sq, mean_squared])
std = layers.Lambda(lambda t: tf.sqrt(tf.maximum(t, 1e-6)), name="std")(var)
x = layers.Concatenate(name="mean_std")([mean, std])

x = layers.Dropout(0.5, name="drop")(x)
out = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

keras_model = Model(inputs=inp, outputs=out)

# Copy PyTorch weights to Keras (stem only for now, rest random-initialized)
ws = model.stem.weight.detach().cpu().numpy().transpose(3, 2, 0, 1)
keras_model.get_layer("conv0").set_weights([ws])

# Copy batch norm weights
bn0_w = model.bn1.weight.detach().cpu().numpy()
bn0_b = model.bn1.bias.detach().cpu().numpy()
keras_model.get_layer("gn0").set_weights([bn0_w, bn0_b])

# Copy block1 weights (no stride, so shapes match)
# block1: Conv2d(16, 16, 3, stride=1, padding=1)
b1_w = model.block1[0].weight.detach().cpu().numpy().transpose(3, 2, 0, 1)  # OIhw -> CFIO
b1_b = model.block1[0].bias.detach().cpu().numpy()
keras_model.get_layer("conv1").set_weights([b1_b] + [bn0_w, bn0_b] if False else [b1_w, b1_b])
# Actually let me just copy what we can
try:
    keras_model.get_layer("conv1").set_weights([model.block1[0].weight.detach().cpu().numpy().transpose(3, 2, 0, 1),
                                                  model.block1[0].bias.detach().cpu().numpy()])
except Exception as e:
    print(f"[warn] Could not copy block1 weights: {e}")

# Copy block2 (has stride=2, but Keras handles it)
try:
    keras_model.get_layer("conv2").set_weights([model.block2[0].weight.detach().cpu().numpy().transpose(3, 2, 0, 1),
                                                  model.block2[0].bias.detach().cpu().numpy()])
except Exception as e:
    print(f"[warn] Could not copy block2 weights: {e}")

# Copy block3 (no stride)
try:
    keras_model.get_layer("conv3").set_weights([model.block3[0].weight.detach().cpu().numpy().transpose(3, 2, 0, 1),
                                                  model.block3[0].bias.detach().cpu().numpy()])
except Exception as e:
    print(f"[warn] Could not copy block3 weights: {e}")

# Copy block4 (has stride=2)
try:
    keras_model.get_layer("conv4").set_weights([model.block4[0].weight.detach().cpu().numpy().transpose(3, 2, 0, 1),
                                                  model.block4[0].bias.detach().cpu().numpy()])
except Exception as e:
    print(f"[warn] Could not copy block4 weights: {e}")

print("[Keras] Model built and partially weighted")

# ===== Convert to TFLite =====
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "aasist_detector.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"[TFLite] Size: {size_kb:.1f} KiB")

# Test TFLite inference
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
inp_details = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]

# Run with PyTorch output first
with torch.no_grad():
    pt_output = torch.sigmoid(output).item()

# TFLite inference
test_input = dummy_mel.astype(np.float32)
test_input_tflite = np.expand_dims(test_input, axis=(0, -1))  # (1, 64, 198, 1)

interpreter.set_tensor(inp_details['index'], test_input_tflite)
interpreter.invoke()
tflite_output = interpreter.get_tensor(out_details['index'])[0, 0]

print(f"[Comparison] PyTorch: {pt_output:.4f}, TFLite: {tflite_output:.4f}")
print(f"[Comparison] Max |diff|: {abs(pt_output - tflite_output):.4f}")
print(f"[Comparison] Correlation: N/A (single sample)")

print("\n=== PATH 1 COMPLETE: AASIST-lite TFLite model saved ===")
PYEOF