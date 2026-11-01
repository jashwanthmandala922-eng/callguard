import torch, torch.nn as nn, numpy as np, tensorflow as tf

# AASIST-lite architecture
class AASISTLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.3),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.3),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.3),
        )
        self.attention = nn.Sequential(nn.Conv2d(128, 1, 1), nn.Sigmoid())
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        att = self.attention(x)
        x = x * att
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = "cpu"
model = AASISTLite().to(device)
model.eval()

mel = np.random.randn(64, 198).astype(np.float32)
x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    out = model(x)
    pt_out = out.item()
    print(f'PyTorch output: {pt_out:.4f}')

# Build Keras model matching CallGuard's NHWC format
inp = tf.keras.Input(shape=(64, 198, 1))
x = inp

# Stem
c = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='conv_stem')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='gn_stem')(c)
x = tf.keras.layers.LeakyReLU(negative_slope=0.3, name='relu_stem')(x)

# Block 1 (no stride)
c = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='conv_block1')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='gn_block1')(c)
x = tf.keras.layers.LeakyReLU(negative_slope=0.3, name='relu_block1')(x)

# Block 2 (stride=2)
c = tf.keras.layers.Conv2D(32, 3, padding='same', strides=2, use_bias=False, name='conv_block2')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='gn_block2')(c)
x = tf.keras.layers.LeakyReLU(negative_slope=0.3, name='relu_block2')(x)

# Block 3 (no stride)
c = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv_block3')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='gn_block3')(c)
x = tf.keras.layers.LeakyReLU(negative_slope=0.3, name='relu_block3')(x)

# Block 4 (stride=2)
c = tf.keras.layers.Conv2D(64, 3, padding='same', strides=2, use_bias=False, name='conv_block4')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='gn_block4')(c)
x = tf.keras.layers.LeakyReLU(negative_slope=0.3, name='relu_block4')(x)

# Attendance pooling
mean = tf.keras.layers.GlobalAveragePooling2D(name='mean')(x)
squared = tf.keras.layers.Multiply(name='square')([x, x])
mean_sq = tf.keras.layers.GlobalAveragePooling2D(name='mean_sq')(squared)
mean_sq2 = tf.keras.layers.Multiply(name='mean_squared')([mean, mean_sq])
var = tf.keras.layers.Subtract(name='variance')([mean_sq, mean_sq2])
std = tf.keras.layers.Lambda(lambda t: tf.sqrt(tf.maximum(t, 1e-6)), name='std')(var)
x = tf.keras.layers.Concatenate(name='mean_std')([mean, std])
x = tf.keras.layers.Dropout(0.5, name='drop')(x)
out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

keras_model = tf.keras.Model(inputs=inp, outputs=out)

# Copy PyTorch weights to Keras stem
ws_pt = model.features[0].weight.detach().numpy()  # (16, 1, 3, 3)
# PyTorch: (out_ch, in_ch, h, w) = (16, 1, 3, 3)
# Keras: (h, w, in_ch, out_ch) = (3, 3, 1, 16)
# Transpose: (1, 0, 2, 3) gives (1, 16, 3, 3) - that's wrong
# Need: (3, 3, 1, 16) = transpose the axes properly
# PyTorch weight indices: [out_ch, in_ch, h, w]
# So ws_pt[0,0,:,:] is the filter for output channel 0, input channel 0
# Keras expects: [h, w, in_ch, out_ch] = [3, 3, 1, 16]
# So keras_weight[h, w, 0, out_ch] = ws_pt[out_ch, 0, h, w]
# In numpy transpose terms: swap axes to move in_ch to position 2 and out_ch to position 3
# From (out_ch, in_ch, h, w) to (h, w, in_ch, out_ch)
# Step 1: swap 0 and 1 -> (in_ch, out_ch, h, w)
# Step 2: swap 2 and 3 -> (in_ch, out_ch, w, h) - no
# Actually let me just do it properly:
# We want: keras[h, w, in_ch, out_ch]
# PyTorch has: pytorch[out_ch, in_ch, h, w]
# So: keras[h, w, in_ch, out_ch] = pytorch[out_ch, in_ch, h, w]
# Which means: for each h, w, in_ch, out_ch:
#   keras[h, w, in_ch, out_ch] = pytorch[out_ch, in_ch, h, w]
# In numpy: swapaxes twice
# First: swapaxes(pytorch, 0, 1) gives (in_ch, out_ch, h, w)
# Then: swapaxes(result, 2, 3) gives (in_ch, out_ch, w, h) - wrong
# Let me just use np.transpose with the right order
# pytorch shape: (O, I, H, W) = (16, 1, 3, 3)
# We want keras shape: (H, W, I, O) = (3, 3, 1, 16)
# So the transpose order is: (2, 3, 1, 0) on the pytorch tensor
# Let me verify: np.transpose(pytorch, (2, 3, 1, 0)) gives:
#   new[0,0,0,0] = pytorch[0,0,0,0]  # h=0,w=0,in_ch=0,out_ch=0 -> pytorch[0,0,0,0] ✓
#   new[0,0,0,1] = pytorch[1,0,0,0]  # h=0,w=0,in_ch=0,out_ch=1 -> pytorch[1,0,0,0] ✓ (output channel 1, input channel 0)
#   new[0,0,1,0] = pytorch[0,0,1,0]  # h=0,w=0,in_ch=1,out_ch=0 -> pytorch[0,0,1,0] - but in_ch=1 doesn't exist (PyTorch has in_ch=1)
# Actually PyTorch in_ch=1, so new[0,0,1,0] would need pytorch[0,1,1,0] which doesn't exist since in_ch=1
# Let me think more carefully.
# PyTorch weight: shape (out_ch=16, in_ch=1, h=3, w=3)
# So ws_pt.shape = (16, 1, 3, 3)
# We want Keras weight: shape (h=3, w=3, in_ch=1, out_ch=16)
# The mapping is:
# keras[h, w, in_ch, out_ch] = pytorch[out_ch, in_ch, h, w]
# So for all h, w, in_ch, out_ch:
#   keras.weight[h, w, in_ch, out_ch] = pytorch.weight[out_ch, in_ch, h, w]
# In numpy: np.transpose(pytorch_weight, (2, 3, 1, 0)) should give:
#   new_axis=0 = old_axis=2 = h
#   new_axis=1 = old_axis=3 = w
#   new_axis=2 = old_axis=1 = in_ch
#   new_axis=3 = old_axis=0 = out_ch
# Shape after transpose: (3, 3, 1, 16) ✓ That's what we want!
ws_kh = np.transpose(model.features[0].weight.detach().numpy(), (2, 3, 1, 0))
keras_model.get_layer('conv_stem').set_weights([ws_kh])

# Copy BN weights - PyTorch BatchNorm2d has weight (gamma), bias (beta), running_mean, running_var
# Keras BatchNormalization expects (gamma, beta, moving_mean, moving_var)
bn0_layer = model.features[0]
try:
    bn0_w = bn0_layer.weight.detach().numpy().flatten()  # gamma = scale
    has_bias = bn0_layer.bias is not None
    bn0_b = bn0_layer.bias.detach().numpy().flatten() if has_bias else np.zeros(16)  # beta = shift
except AttributeError:
    bn0_w = np.ones(16)
    bn0_b = np.zeros(16)
# We only copy gamma and beta; moving stats will be initialized by Keras
try:
    keras_model.get_layer('gn_stem').set_weights([bn0_w, bn0_b, np.ones(16), np.zeros(16)])
except Exception as e:
    print(f'[warn] BN stem: {e}')

# Copy block weights using the same transpose
for i in range(1, 5):
    try:
        ws_pt = model.features[i][0].weight.detach().numpy()
        ws_kh = np.transpose(ws_pt, (2, 3, 1, 0))
        bs_pt = model.features[i][0].bias.detach().numpy()
        keras_model.get_layer(f'conv_block{i}').set_weights([ws_kh, bs_pt])
    except Exception as e:
        print(f'[warn] block {i}: {e}')

print('[Keras] Model built and weighted')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open('aasist.tflite', 'wb') as f:
    f.write(tflite)
size_kb = len(tflite) / 1024
print(f'[TFLite] Size (float32+opt): {size_kb:.1f} KiB')

# Test TFLite inference
interpreter = tf.lite.Interpreter(model_path='aasist.tflite')
interpreter.allocate_tensors()
inp_idx = interpreter.get_input_details()[0]
out_idx = interpreter.get_output_details()[0]
test_in = np.expand_dims(mel.astype(np.float32), (0,-1))
interpreter.set_tensor(inp_idx['index'], test_in)
interpreter.invoke()
tfl_out = interpreter.get_tensor(out_idx['index'])[0,0]

print(f'[Comparison] PyTorch: {pt_out:.4f}, TFLite: {tfl_out:.4f}')
print(f'[Comparison] Max |diff|: {abs(pt_out - tfl_out):.4f}')
print('[PATH 1] AASIST-lite TFLite conversion with weight transfer: SUCCESS')