#!/usr/bin/env python3
"""Fix AASIST-lite weight transfer in export_keras."""
with open('tools/train_torch.py', 'r') as f:
    content = f.read()

old = """    if is_aasist:
        # === AASIST-lite weight transfer ===
        # features blocks: Conv2d-BatchNorm-LeakyReLU, stride at blocks 1 and 3
        # params: features.0.weight, features.1.weight/ bias, features.3.weight/ bias, etc.
        # attention: attention.0.weight
        # classifier: classifier.2.weight, classifier.3.bias

        # Copy stem conv weights: PyTorch (O,I,H,W) -> Keras (H,W,I,O)
        ws = state["features.0.weight"].detach().cpu().numpy()
        ws_kh = np.transpose(ws, (2, 3, 1, 0))  # (3,3,1,16)
        conv_layers[0].set_weights([ws_kh])

        # Copy stem BN weights: gamma (weight), beta (bias)
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
            out_layer.set_weights([out_w, out_b])"""

new = """    if is_aasist:
        # === AASIST-lite weight transfer ===
        # Stem conv: features.0.weight -> conv0
        if "features.0.weight" in state:
            ws = state["features.0.weight"].detach().cpu().numpy()
            ws_kh = np.transpose(ws, (2, 3, 1, 0))
            conv_layers[0].set_weights([ws_kh])
        
        # Stem BN: copy weight and bias if present
        if "features.1.weight" in state:
            bn0_w = state["features.1.weight"].detach().cpu().numpy().flatten()
            bn0_b = state.get("features.1.bias", None)
            if bn0_b is not None:
                bn0_b = bn0_b.detach().cpu().numpy().flatten()
            else:
                bn0_b = np.zeros(16, dtype=np.float32)
            gn_layers[0].set_weights([bn0_w, bn0_b])
        
        # Block weights - check key existence before accessing
        for block_idx in range(1, 4):
            conv_w_key = f"features.{block_idx * 2}.weight"
            if conv_w_key in state and block_idx < len(conv_layers):
                w = state[conv_w_key].detach().cpu().numpy()
                w_kh = np.transpose(w, (2, 3, 1, 0))
                conv_layers[block_idx].set_weights([w_kh])
            # BN weight and bias
            bn_key = f"features.{block_idx * 2 + 1}"
            if bn_key in state:
                bn_state = state[bn_key]
                bn_w = bn_state.get("weight", bn_state.get("w", None))
                bn_b = bn_state.get("bias", bn_state.get("b", None))
                if bn_w is not None:
                    bn_w = bn_w.detach().cpu().numpy().flatten()
                else:
                    n_filters = 16 * (2 ** block_idx)
                    bn_w = np.ones(n_filters, dtype=np.float32)
                if bn_b is not None:
                    bn_b = bn_b.detach().cpu().numpy().flatten()
                else:
                    bn_b = np.zeros(n_filters, dtype=np.float32)
                if block_idx < len(gn_layers):
                    gn_layers[block_idx].set_weights([bn_w, bn_b])
        
        # Attention: attention.0.weight -> attention Conv2D(1,1)
        if "attention.0.weight" in state:
            att_w = state["attention.0.weight"].detach().cpu().numpy()
            try:
                keras_model.get_layer("attention").set_weights([att_w.reshape(1, 1, 128, 1)])
            except Exception:
                pass
        
        # Head classifier weights
        if "classifier.2.weight" in state:
            fc_w = state["classifier.2.weight"].detach().cpu().numpy().T
            fc_b = state.get("classifier.3.bias", None)
            if fc_b is not None:
                fc_b = fc_b.detach().cpu().numpy()
            else:
                fc_b = np.zeros(1, dtype=np.float32)
            keras_model.get_layer("fc1").set_weights([fc_w, fc_b])
        
        if "classifier.3.weight" in state:
            out_w = state["classifier.3.weight"].detach().cpu().numpy().T
            out_b = state.get("classifier.3.bias", None)
            if out_b is not None:
                out_b = out_b.detach().cpu().numpy()
            else:
                out_b = np.zeros(1, dtype=np.float32)
            out_layer = keras_model.get_layer("fake_probability")
            out_layer.set_weights([out_w, out_b])"""

if old in content:
    content = content.replace(old, new)
    with open('tools/train_torch.py', 'w') as f:
        f.write(content)
    print('Fixed AASIST weight transfer successfully')
else:
    print('Old code not found, checking current state...')
    idx = content.find('if is_aasist:')
    if idx >= 0:
        print(f'AASIST block at position {idx}')