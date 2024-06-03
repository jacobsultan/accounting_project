# Import necessary libraries and modules
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, cfg):
        super(Block, self).__init__()
        self.attention = nn.MultiheadAttention(
            cfg.embeds_size, cfg.num_heads, batch_first=True
        )

        # FFN  consisting of two linear layers and a LeakyReLU activation in between
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embeds_size, 2 * cfg.embeds_size),  # First linear layer to expand dimension
            nn.LeakyReLU(),  # Non-linear activation function
            nn.Linear(2 * cfg.embeds_size, cfg.embeds_size),  # Second linear layer to contract dimension back
        )

        # Dropout layers to prevent overfitting
        self.drop1 = nn.Dropout(cfg.drop_prob)
        self.drop2 = nn.Dropout(cfg.drop_prob)

        # Layer normalization to stabilize learning process
        self.ln1 = nn.LayerNorm(cfg.embeds_size)
        self.ln2 = nn.LayerNorm(cfg.embeds_size)

    # DForward pass function that takes in the hidden state and returns the output of the block
    def forward(self, hidden_state):
        attn, _ = self.attention(
            hidden_state, hidden_state, hidden_state, need_weights=False
        )
        attn = self.drop1(attn)
        out = self.ln1(hidden_state + attn)
        observed = self.ffn(out)
        observed = self.drop2(observed)
        return self.ln2(out + observed)