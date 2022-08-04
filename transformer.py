"""
This file contains the architecture for transformer.  The transformer, originally from the paper
"Attention is all you need," finds long term relationships through the use of dot product matrix multiplication
of encoded representations of the data.  This "attention" mechanisms compare the similarities in the
encoded data. Focusing the attention of each point in the sequence to the points that influence them the most.  The
encodings are then passed through an MLP to produce the encodings for the next layer.
"""
import torch
import torch.nn as nn


class Transformer(nn.Module):
    pass
