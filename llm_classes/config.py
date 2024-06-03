# Import necessary libraries and modules
from dataclasses import dataclass


# Defining model parameters
@dataclass
class Config:
    vocab_size: int
    num_classes: int
    block_size: int = 15
    embeds_size: int = 100
    drop_prob: int = 0.15
    batch_size: int = 16
    epochs: int = 16
    num_heads: int = 4