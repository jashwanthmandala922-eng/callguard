#!/usr/bin/env python3
"""Fix export_keras to support AASIST-lite architecture."""
import re

with open('tools/train_torch.py', 'r') as f:
    content = f.read()

# Find the export_keras function and replace it entirely
# We need to find the function definition and its body until the next top-level function

# Strategy: find "def export_keras" and "def tflite_scores" and replace everything in between
start_marker = "def export_keras"
end_marker = "def tflite_scores"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx >= 0 and end_idx > start_idx:
    # Get the function including the next function's def line
    new_function = """def export_keras(torch_model, out_path: str, pooling: str, width: tuple) -> None:
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = layers.Input(shape=(N_MEL, N_FRAMES, 1), name="logmel")
    x = inp
    conv_layers = []
    gn_layers = []
    # Detect architecture from model state dict keys
    is_aasist = "features.0.weight" in torch_model.state_dict()

    if is_aasist:
        # AASIST-lite architecture (fixed width, LeakyReLU, stride-2 at blocks 1&3)
        fixed_width = (16, 32, 64, 128)
        for i, out_ch in enumerate(fixed_width):
            c = layers.Conv2D(out_ch, 3, padding="same", use_bias=False,
                              name=f"conv{i}")
            x = c(x)
            g = layers.GroupNormalization(groups=8, epsilon=1e-5, name=f"gn{i}")
            x = g(x)
            x = layers.LeakyReLU(negative_slope=0.3, name=f"relu{i}")(x)
            if i < 3:  # stride-2 after blocks 1 and 2 (0-indexed)
                x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
            conv_layers.append(c)
            gn_layers.append(g)

        if pooling == "meanstd":
            mean = layers.GlobalAveragePooling2D(name="mean")(x)
            squared = layers.Multiply(name="square")([x, x])
            mean_sq = layers.GlobalAveragePooling2D(name="mean_sq")(squared)
            mean_squared = layers.Multiply(name="mean_squared")([mean, mean])
            var = layers.Subtract(name="variance")([mean_sq, mean_squared])
            std = layers.Lambda(lambda t: tf.sqrt(tf.maximum(t, 1e-6)), name="std")(var)
            x = layers.Concatenate(name="mean_std")([mean, std])
        else:
            x = layers.GlobalAveragePooling2D(name="mean")(x)

        x = layers.Dense(64, activation="relu", name="fc1")(x)
        x = layers.Dropout(0.5, name="drop")(x)
        out = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    else:
        # CNN architecture (original path)
        for i, out_ch in enumerate(width):
            c = layers.Conv2D(out_ch, 3, padding="same", use_bias=False,
                              name=f"conv{i}")
            x = c(x)
            g = layers.GroupNormalization(groups=8, epsilon=1e-5, name=f"gn{i}")
            x = g(x)
            x = layers.ReLU(name=f"relu{i}")(x)
            x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
            conv_layers.append(c)
            gn_layers.append(g)

        if pooling == "meanstd":
            mean = layers.GlobalAveragePooling2D(name="mean")(x)
            squared = layers.Multiply(name="square")([x, x])
            mean_sq = layers.GlobalAveragePooling2D(name="mean_sq")(squared)
            mean_squared = layers.Multiply(name="mean_squared")([mean, mean])
            var = layers.Subtract(name="variance")([mean_sq, mean_squared])
            std = layers.Lambda(lambda t: tf.sqrt(tf.maximum(t, 1e-6)), name="std")(var)
            x = layers.Concatenate(name="mean_std")([mean, std])
        else:
            x = layers.GlobalAveragePooling2D(name="mean")(x)

        x = layers.Dense(64, activation="relu", name="fc1")(x)
        x = layers.Dropout(0.3, name="drop")(x)
        out = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

    keras_model = tf.keras.Model(inp, out)

    state = torch_model.state_dict()

    if is_aasist:
        # === AASIST-lite weight transfer ===
        # Stem conv weight: features.0.weight -> conv0
        ws = state["features.0.weight"].detach().cpu().numpy()
        ws_kh = np.transpose(ws, (2, 3, 1, 0))
        conv_layers[0].set_weights([ws_kh])

        # Stem BN: features.1.weight (gamma), features.1.bias (beta)
        bn0_w = state["features.1.weight"].detach().cpu().numpy().flatten()
        bn0_b = state["features.1.bias"].detach().cpu().numpy().flatten()
        gn_layers[0].set_weights([bn0_w, bn0_b])

        # Block weights: features.2.weight/ bias, features.4.weight/ bias, features.6.weight/ bias
        for block_idx in range(1, 4):
            conv_w_key = f"features.{block_idx * 2}.weight"
            bn_w_key = f"features.{block_idx * 2 + 1}.weight"
            bn_b_key = f"features.{block_idx * 2 + 1}.bias"
            if conv_w_key in state:
                w = state[conv_w_key].detach().cpu().numpy()
                w_kh = np.transpose(w, (2, 3, 1, 0))
                conv_layers[block_idx].set_weights([w_kh])
            if bn_w_key in state:
                gn_layers[block_idx].set_weights([
                    state[bn_w_key].detach().cpu().numpy().flatten(),
                    state[bn_b_key].detach().cpu().numpy().flatten(),
                ])

        # Attention: attention.0.weight -> attention Conv2D(1,1)
        if "attention.0.weight" in state:
            att_w = state["attention.0.weight"].detach().cpu().numpy()
            try:
                keras_model.get_layer("attention").set_weights([att_w.reshape(1, 1, 128, 1)])
            except Exception:
                pass

        # Head: classifier.2.weight/bias -> fc1; classifier.3.weight/bias -> fake_probability
        if "classifier.2.weight" in state:
            fc_w = state["classifier.2.weight"].detach().cpu().numpy().T
            fc_b = state["classifier.3.bias"].detach().cpu().numpy()
            keras_model.get_layer("fc1").set_weights([fc_w, fc_b])

        if "classifier.3.weight" in state:
            out_w = state["classifier.3.weight"].detach().cpu().numpy().T
            out_b = state["classifier.3.bias"].detach().cpu().numpy()
            out_layer = keras_model.get_layer("fake_probability")
            out_layer.set_weights([out_w, out_b])
    else:
        # CNN weight transfer (existing logic, unchanged)
        for i, _ in enumerate(width):
            w = state[f"body.{i * 4}.weight"].detach().cpu().numpy()
            conv_layers[i].set_weights([w.transpose(2, 3, 1, 0)])
            gamma = state[f"body.{i * 4 + 1}.weight"].detach().cpu().numpy()
            beta = state[f"body.{i * 4 + 1}.bias"].detach().cpu().numpy()
            gn_layers[i].set_weights([gamma, beta])

        keras_model.get_layer("fc1").set_weights([
            state["head.0.weight"].detach().cpu().numpy().T,
            state["head.0.bias"].detach().cpu().numpy(),
        ])
        keras_model.get_layer("fake_probability").set_weights([
            state["head.3.weight"].detach().cpu().numpy().T,
            state["head.3.bias"].detach().cpu().numpy(),
        ])


def tflite_scores"""

    # Replace
    new_content = content[:start_idx] + new_function + content[end_idx:]
    
    with open('tools/train_torch.py', 'w') as f:
        f.write(new_content)
    print(f"Replaced export_keras function successfully (positions {start_idx}-{end_idx})")
else:
    print(f"Could not find markers: start={start_idx}, end={end_idx}")