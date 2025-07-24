import math

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------
# 1. Configuration & Hyperparameters
# -----------------------------------------------
GRID_SIZE = 16
# The range of a diff, e.g., for a 16x16 grid, a diff can be from -15 to +15 (31 values)
DIFF_RANGE = 2 * GRID_SIZE - 1
DIFF_VOCAB_SIZE = DIFF_RANGE * DIFF_RANGE

# Model Hyperparameters
D_MODEL = 64
# We'll split the model's dimension among the three input types
D_DIFF_EMB = 32
D_ROW_EMB = 16
D_COL_EMB = 16

N_HEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

# Training Hyperparameters
LEARNING_RATE = 0.002
EPOCHS = 100

# -----------------------------------------------
# 2. Helper Functions for Diffs
# -----------------------------------------------


def diff_to_token(diff):
    """Converts a (dr, dc) diff tuple to a single token."""
    dr, dc = diff
    # Shift dr and dc to be non-negative for tokenization
    dr_shifted = dr + GRID_SIZE - 1
    dc_shifted = dc + GRID_SIZE - 1
    return dr_shifted * DIFF_RANGE + dc_shifted


def token_to_diff(token):
    """Converts a token back to a (dr, dc) diff tuple."""
    dr_shifted = token // DIFF_RANGE
    dc_shifted = token % DIFF_RANGE
    dr = dr_shifted - (GRID_SIZE - 1)
    dc = dc_shifted - (GRID_SIZE - 1)
    return (dr, dc)


# -----------------------------------------------
# 3. Model Definition (Updated for Diffs)
# -----------------------------------------------


class PositionalEncoding(nn.Module):
    # This class remains unchanged.
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
    """
    A Transformer that predicts the next diff based on a history of
    diffs and their absolute starting positions.
    """

    def __init__(self, d_model, n_head, num_layers, dim_feedforward, dropout):
        super(PixelTransformer, self).__init__()
        self.d_model = d_model

        # Embeddings for diffs, and the absolute row/col where the diff started
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
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # A single output head to predict the next diff token
        self.output_head = nn.Linear(d_model, DIFF_VOCAB_SIZE)

    def _generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, 3] where the last dim is
                 (diff_token, absolute_start_row, absolute_start_col)
        """
        device = src.device
        causal_mask = self._generate_causal_mask(src.size(1)).to(device)

        # Create a combined embedding from the three input parts
        diff_tokens, start_rows, start_cols = src[:, :, 0], src[:, :, 1], src[:, :, 2]
        diff_emb = self.diff_embedding(diff_tokens)
        row_emb = self.row_embedding(start_rows)
        col_emb = a = self.col_embedding(start_cols)

        # Concatenate embeddings to form the full input vector for each sequence step
        src_emb = torch.cat([diff_emb, row_emb, col_emb], dim=-1)

        src_pos = self.pos_encoder(src_emb * math.sqrt(self.d_model))

        output = self.transformer_decoder(
            tgt=src_pos, memory=src_pos, tgt_mask=causal_mask, memory_mask=causal_mask
        )

        logits = self.output_head(output)
        return logits


# -----------------------------------------------
# 4. On-Demand Training and Inference (Updated)
# -----------------------------------------------


def train_and_infer(drawing_coords, device, k=5):
    """
    Trains a model on a single drawing's diffs and returns the top k predictions.
    """
    if len(drawing_coords) < 2:
        print("⚠️ Drawing is too short to train. Needs at least 2 points.")
        return []

    model = PixelTransformer(
        d_model=D_MODEL,
        n_head=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Convert the coordinate sequence into a training sequence of
    # (diff_token, start_row, start_col)
    model_inputs = []
    target_diffs = []
    for i in range(len(drawing_coords) - 1):
        p1 = drawing_coords[i]
        p2 = drawing_coords[i + 1]
        diff = (p2[0] - p1[0], p2[1] - p1[1])

        model_inputs.append([diff_to_token(diff), p1[0], p1[1]])
        target_diffs.append(diff_to_token(diff))

    model_inputs_tensor = torch.tensor(model_inputs, dtype=torch.long, device=device)
    target_diffs_tensor = torch.tensor(
        target_diffs[1:], dtype=torch.long, device=device
    )

    print(f"🚀 Starting on-demand training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        # We only train if there's at least one target diff to predict
        if len(target_diffs_tensor) > 0:
            for i in range(1, len(model_inputs_tensor)):
                input_sub_seq = model_inputs_tensor[:i].unsqueeze(0)
                target_token = target_diffs_tensor[i - 1].unsqueeze(0)

                optimizer.zero_grad()
                output_logits = model(input_sub_seq)
                last_token_logits = output_logits[:, -1, :]

                loss = criterion(last_token_logits, target_token)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(target_diffs_tensor)
                print(f"  Epoch [{epoch + 1}/{EPOCHS}], Avg Loss: {avg_loss:.4f}")

    print("✅ Training complete.")
    print(f"\n🔮 Running inference to find top {k} predictions...")
    model.eval()
    with torch.no_grad():
        inference_input = model_inputs_tensor.unsqueeze(0)
        output_logits = model(inference_input)
        last_token_logits = output_logits[:, -1, :]

        # Use torch.topk to get the best k predictions
        topk_probs, topk_indices = torch.topk(
            torch.softmax(last_token_logits, dim=-1), k, dim=1
        )

        last_coord = np.array(drawing_coords[-1])
        predictions = []
        for i in range(k):
            prob = topk_probs[0, i].item()
            pred_token = topk_indices[0, i].item()
            pred_diff = np.array(token_to_diff(pred_token))

            # Calculate the final predicted coordinate
            next_coord = tuple(last_coord + pred_diff)

            # Ensure the prediction is within the grid bounds
            if 0 <= next_coord[0] < GRID_SIZE and 0 <= next_coord[1] < GRID_SIZE:
                predictions.append({"coord": next_coord, "prob": prob})

    return predictions


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # A straight line. The model should overwhelmingly predict continuation.
    # user_drawing = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

    # An 'L' shape. Predictions should show possibilities like continuing the line or turning a corner.
    user_drawing = [(2, 2), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4)]

    print(f"Input drawing (coords): {user_drawing}")

    top_predictions = train_and_infer(user_drawing, device, k=5)

    if top_predictions:
        print("\n--- Top 5 Predictions ---")
        for pred in top_predictions:
            print(
                f"  Coordinate: {str(pred['coord']):<10} | Probability: {pred['prob']:.2%}"
            )
