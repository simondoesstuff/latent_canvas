import math

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.onnx

# -----------------------------------------------
# 1. Configuration
# -----------------------------------------------
GRID_SIZE = 16
DIFF_RANGE = 2 * GRID_SIZE - 1
DIFF_VOCAB_SIZE = DIFF_RANGE * DIFF_RANGE
D_MODEL = 64
D_DIFF_EMB = 32
D_ROW_EMB = 16
D_COL_EMB = 16
N_HEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 128
DROPOUT = 0.1


# -----------------------------------------------
# 2. Model Definition
# -----------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PixelTransformer(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dim_feedforward, dropout):
        super(PixelTransformer, self).__init__()
        self.d_model = d_model
        self.diff_embedding = nn.Embedding(DIFF_VOCAB_SIZE, D_DIFF_EMB)
        self.row_embedding = nn.Embedding(GRID_SIZE, D_ROW_EMB)
        self.col_embedding = nn.Embedding(GRID_SIZE, D_COL_EMB)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.output_head = nn.Linear(d_model, DIFF_VOCAB_SIZE)

    def _generate_causal_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

    def forward(self, src, tgt_mask):
        diff_tokens, start_rows, start_cols = src[:, :, 0], src[:, :, 1], src[:, :, 2]
        diff_emb = self.diff_embedding(diff_tokens)
        row_emb = self.row_embedding(start_rows)
        col_emb = self.col_embedding(start_cols)
        src_emb = torch.cat([diff_emb, row_emb, col_emb], dim=-1)
        src_pos = self.pos_encoder(src_emb * math.sqrt(self.d_model))
        output = self.transformer_decoder(
            tgt=src_pos, memory=src_pos, tgt_mask=tgt_mask, memory_mask=tgt_mask
        )
        logits = self.output_head(output)
        return logits


# -----------------------------------------------
# 3. ONNX Export
# -----------------------------------------------
def export_and_verify_onnx():
    """
    Instantiates the model, exports it to ONNX, and verifies the output.
    """
    print("🚀 Starting ONNX export process...")

    # --- Model Instantiation ---
    # Instantiate the model and set it to evaluation mode for consistent output.
    model = PixelTransformer(
        d_model=D_MODEL,
        n_head=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    )
    model.eval()
    print("✅ Model instantiated and set to evaluation mode.")

    # --- Dummy Input Creation ---
    # Define a sample batch size and sequence length for the dummy input.
    batch_size = 4
    sequence_length = 256  # An example sequence length
    onnx_file_path = "pixel_transformer.onnx"

    # Create dummy inputs that match the model's forward pass signature.
    # src: (batch_size, sequence_length, 3)
    dummy_src = torch.randint(
        0, GRID_SIZE, (batch_size, sequence_length, 3), dtype=torch.long
    )

    # tgt_mask: (sequence_length, sequence_length)
    dummy_mask = model._generate_causal_mask(sz=sequence_length, device="cpu")

    print(
        f"✅ Dummy inputs created: src shape={dummy_src.shape}, mask shape={dummy_mask.shape}"
    )

    # --- Export to ONNX ---
    # The `dynamic_axes` argument allows the exported model to handle variable
    # batch sizes and sequence lengths, which is essential for many applications.
    torch.onnx.export(
        model,
        (dummy_src, dummy_mask),
        onnx_file_path,
        input_names=["src", "tgt_mask"],
        output_names=["logits"],
        dynamic_axes={
            "src": {0: "batch_size", 1: "sequence_length"},
            "tgt_mask": {0: "sequence_length", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"✅ Model successfully exported to '{onnx_file_path}'")

    # --- Verification Step ---
    print("\n🔍 Verifying the exported ONNX model...")
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model integrity check passed.")

    # Compare PyTorch and ONNX Runtime outputs
    ort_session = onnxruntime.InferenceSession(onnx_file_path)

    with torch.no_grad():
        torch_logits = model(dummy_src, dummy_mask)

    ort_inputs = {
        ort_session.get_inputs()[0].name: dummy_src.numpy(),
        ort_session.get_inputs()[1].name: dummy_mask.numpy(),
    }
    ort_logits = ort_session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(torch_logits.numpy(), ort_logits, rtol=1e-3, atol=1e-5)
    print("✅ ONNX Runtime output matches PyTorch output.")
    print("\n🎉 Export and verification complete!")


# --- Execute Export ---
if __name__ == "__main__":
    export_and_verify_onnx()
