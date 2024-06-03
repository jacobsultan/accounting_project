# Import necessary libraries and modules
import torch.nn as nn
import torch as t
import sys
sys.path.append(
    "llm_classes"
)

# Import custom classes 
from block import Block

# Check for GPU availability and set the device accordingly
device = "cuda" if t.cuda.is_available() else "cpu"

# Define the Transformer class which inherits from nn.Module
class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()  # Initializing the parent class
        self.cfg = cfg  # Storing the configuration object
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embeds_size)  # Token embedding layer
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.embeds_size)  # Positional embedding layer
        self.block = Block(cfg)  # Instantiating the Block class with the configuration
        self.ln1 = nn.LayerNorm(cfg.embeds_size)  # Layer normalization
        self.ln2 = nn.LayerNorm(cfg.embeds_size)  # Another layer normalization (unused in forward)

        # Classifier head consisting of linear layers, activations, dropout, and softmax
        self.classifier_head = nn.Sequential(
            nn.Linear(cfg.embeds_size, cfg.embeds_size),
            nn.LeakyReLU(),  # LeakyReLU activation function
            nn.Dropout(cfg.drop_prob),  # Dropout for regularization
            nn.Linear(cfg.embeds_size, cfg.embeds_size),
            nn.LeakyReLU(),  # Another LeakyReLU activation function
            nn.Linear(cfg.embeds_size, cfg.num_classes),  # Final linear layer for class scores
            nn.Softmax(dim=1),  # Softmax to get probabilities for each class
        )

    # Method to calculate the number of parameters in the model
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())  # Summing the number of elements in each parameter
        return n_params

    # Forward pass of the model
    def forward(self, seq):
        B, T = seq.shape  # B is batch size, T is sequence length
        embedded = self.tok_emb(seq)  # Apply token embedding
        embedded = embedded + self.pos_emb(t.arange(T, device=device))  # Add positional embedding
        output = self.block(embedded)  # Pass through the Block
        output = output.mean(dim=1)  # Pooling operation (mean over the sequence)
        output = self.classifier_head(output)  # Pass through the classifier head
        return output  # Return the final output
