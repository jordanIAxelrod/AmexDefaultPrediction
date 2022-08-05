"""
This file contains the architecture for transformer.  The transformer, originally from the paper
"Attention is all you need," finds long term relationships through the use of dot product matrix multiplication
of encoded representations of the data.  This "attention" mechanisms compare the similarities in the
encoded data. Focusing the attention of each point in the sequence to the points that influence them the most.  The
encodings are then passed through an MLP to produce the encodings for the next layer.

We are looking only for a classification of an entire series not a sequence to sequence application.
Therefore we do not need masking. We produce an encoder only Transformer with an additional classification
token which is attached to the head.
"""
import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Attention Mechanism

    Parameters
    ----------
    dim : int
        The input and the output dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability appl;ied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    ----------
    scale : float
        Normalizing constant for the dot product
    qkv: nn.linear
        Linear projection for the query, key, value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all
        the attention heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p: float = 0., proj_p: float = 0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass
        :param x: torch.Tensor
             Shape `(n_smaples, n_patches + 1, dim)`.
        :return: torch.Tensor
             Shape `(n_smaples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError('Incorrect dimension')
        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)

        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ).permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)

        dp = (
                     q @ k_t
             ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)

        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches + 1, head_dim)

        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)

        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron

    Parameters
    ---------
    in_features : int
        Number of input features
    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
         Number of output features

    p : float
        Dropout probability

    n_layers : int
        number of hidden layers

    Attribute
    ---------
    fc : nn.Sequential
        the hidden layers

    act : nn.GELU
        GELU activation function

    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, p: float = 0.,
                 n_layers: int = 1) -> None:
        super().__init__()
        mod_list = [nn.Linear(in_features, hidden_features), nn.GELU()]
        for _ in range(n_layers - 1):
            mod_list.append(nn.Linear(hidden_features, hidden_features))
            mod_list.append(nn.GELU())
        self.output = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        self.mlp = nn.Sequential(*mod_list)

    def forward(self, x):
        """
        Rund forward pass.
        :param x:  torch.Tensor
            Shape: `(n_samples, n_patches + 1, in_features)`.
        :return: torch.Tensor
            Shape: `(n_samples, n_patches + 1, out_features'.
        """

        x = self.mlp(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)
        x = self.output(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)
        return x


class Block(nn.Module):
    """
    Transformer block.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the 'MLP' module with respect to 'dim'
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    n_layers : int
        number of layers in the MLP
    out_features : int
        Number of output features.
    p, attn_p: float
        Dropout probability

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attnetion Attention module.

    mlp: MLP
        MLP module.
    """

    def __init__(
            self,
            dim: int,
            n_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            n_layers: int = 1,
            out_features: int = None,
            p: float = 0.,
            attn_p: float = .0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        if not out_features:
            out_features = dim
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=out_features,
            n_layers=n_layers,
        )

    def forward(self, x) -> torch.Tensor:
        """
        Run forward pass
        :param x: torch.Tensor
            Shape: `(n_samples, n_patches + 1, dim)`.
        :return: torch.Tensor
            Shape: `(n_samples, n_patches  + 1, out_features)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class Embed(nn.Module):
    def __init__(self, in_features, enc_features, mlp_ratio, depth):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features)
        self.enc = nn.ModuleList()
        self.enc.append(nn.Linear(in_features, enc_features * mlp_ratio))
        for i in range(depth):
            self.enc.append(nn.ReLU())
            self.enc.append(nn.Linear(enc_features * mlp_ratio, enc_features * mlp_ratio))
        self.enc.append(nn.ReLU())
        self.enc.append(nn.Linear(enc_features * mlp_ratio, enc_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        for layer in self.enc:
            x = layer(x)
        return x


class Head(nn.Module):
    def __init__(self, in_features, enc_features, mlp_ratio, depth):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features)
        self.enc = nn.ModuleList()
        self.enc.append(nn.Linear(in_features, enc_features * mlp_ratio))
        for i in range(depth):
            self.enc.append(nn.ReLU())
            self.enc.append(nn.Linear(enc_features * mlp_ratio, enc_features * mlp_ratio))
        self.enc.append(nn.ReLU())
        self.enc.append(nn.Linear(enc_features * mlp_ratio, enc_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        for layer in self.enc:
            x = layer(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, enc_features, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, enc_features)  # (max_len, enc_features)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, enc_features, 2).float() * (-math.log(10e4)) / enc_features)

        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pos_encoding[:x.size(0), :])


class Transformer(nn.Module):
    def __init__(
            self,
            in_features,
            enc_features,
            n_heads,
            qkv_bias,
            mlp_ratio,
            p,
            attn_p,
            max_len,
            n_classes,
            depth
    ):
        super().__init__()

        self.embed = Embed(in_features, enc_features, mlp_ratio, depth)
        self.pos_embed = PositionalEncoding(enc_features, p, max_len)
        self.cls = nn.Parameter(torch.zeros(1, 1, enc_features))

        self.head = Head(enc_features, n_classes, mlp_ratio, depth)

        self.blocks = nn.ModuleList([
            Block(
                dim=enc_features,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                n_layers=depth,
                p=p,
                attn_p=attn_p
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embed(self.embed(x))
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x)

        final_cls = x[:, 0]

        return self.head(final_cls)
