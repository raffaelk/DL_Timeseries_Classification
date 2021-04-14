import torch
import torch.nn as nn

import numpy as np

import torch.nn.functional as F

"""
both, the SelfAttention module and the TransformerBlock, have been adapted from
exercise 10 of the DeepLearning course 2020 at ETHZ
    http://da.inf.ethz.ch/teaching/2020/DeepLearning/
    https://colab.research.google.com/github/leox1v/dl20/blob/master/Transformers_Solution.ipynb

the code for the positional embeddings has been copied from:
    https://www.tensorflow.org/tutorials/text/transformer

"""


class SelfAttention(nn.Module):
    """
    A SelfAttention model.
    Args:
        d: The embedding dimension.
        heads: The number of attention heads.
    """

    def __init__(self, d: int, heads: int = 8):
        super().__init__()

        # gpu or cpu
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        self.dev = torch.device(dev)

        self.k, self.h = d, heads

        self.Wq = nn.Linear(d, d * heads, bias=False).to(self.dev)
        self.Wk = nn.Linear(d, d * heads, bias=False).to(self.dev)
        self.Wv = nn.Linear(d, d * heads, bias=False).to(self.dev)

        # This unifies the outputs of the different heads into
        # a single k-dimensional vector
        self.unifyheads = nn.Linear(heads * d, d).to(self.dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d].

        Returns:
            Self attention tensor of shape [b, l, d].
        """
        b, l, d = x.size()
        h = self.h

        # Transform the input embeddings x of shape [b, l, d] to queries, keys, values.
        # The output shape is [b, l, d, d*h] which we transform into [b, l, h, d]. Then,
        # we fold the heads into the batch dimenstion to arrive at [b*h, l, d]
        queries = self.Wq(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)
        keys = self.Wk(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)
        values = self.Wv(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)

        # Compute the product of queries and keys and scale with sqrt(d).
        # The tensor w' has shape (b*h, l, l) containing raw weights.
        w_prime = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)

        # Compute w by normalizing w' over the last dimension.
        w = F.softmax(w_prime, dim=-1)

        # store w for visualization
        self.attmat = w.detach().view(b, h, l, l).clone()

        # Apply the self attention to the values.
        out = torch.bmm(w, values).view(b, h, l, d)

        # Swap h, l back.
        out = out.transpose(1, 2).contiguous().view(b, l, h * d)

        # Unify heads to arrive at shape [b, l, d].
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    """
    A Transformer block consisting of self attention and ff-layer.
    Args:
        d (int): The embedding dimension.
        heads (int): The number of attention heads.
        n_mlp (int): The number of mlp 'blocks'.
    """

    def __init__(self, d: int, heads: int = 8, n_mlp: int = 4, layer_order: str = 'postLN'):
        super().__init__()

        # gpu or cpu
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        print(f'device: {dev}')

        self.dev = torch.device(dev)
        
        # layyer norm order
        self.layer_order = layer_order

        # The self attention layer.
        self.attention = SelfAttention(d, heads=heads)

        # The two layer norms.
        self.norm1 = nn.LayerNorm(d).to(self.dev)
        self.norm2 = nn.LayerNorm(d).to(self.dev)

        # The feed-forward layer.
        self.ff = nn.Sequential(
            nn.Linear(d, n_mlp*d).to(self.dev),
            nn.ReLU().to(self.dev),
            nn.Linear(n_mlp*d, d).to(self.dev)
        ).to(self.dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d].

        Returns:
            Transformer output tensor of shape [b, l, d].
        """
        if self.layer_order == 'postLN':
            x_prime = self.attention(x)
            x = self.norm1(x_prime + x)
    
            x_prime = self.ff(x)
            return self.norm2(x_prime + x)
        
        elif self.layer_order == 'preLN':
            x_prime = self.attention(self.norm1(x))
            x = x_prime + x
    
            x_prime = self.ff(self.norm2(x))
            return x_prime + x
            

    def get_attmatt(self):
        return self.attention.attmat


# %% architecture of the transformer model


class UnivarTransformer(nn.Module):
    """
    Time series Transformer model.
    This implementation only works for univariate time series.
    """

    def __init__(self, siglen, n_targets, heads=8, depth=3, emb_dim=50, layer_order = 'postLN'):
        super(UnivarTransformer, self).__init__()

        # gpu or cpu
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        self.dev = torch.device(dev)

        # number Transformer blocks
        self.depth = depth

        # dimension of embeddings
        self.emb_dim = emb_dim

        # length of signal (time dimension)
        self.siglen = siglen

        # number of target classes
        self.n_targets = n_targets

        # positional embeddings
        self.pos_emb = self._positional_encoding(self.siglen, self.emb_dim).to(self.dev)

        # pseudo word embedding
        self.wrd_emb = nn.Conv1d(1, self.emb_dim, 1).to(self.dev)

        # The stacked transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d=self.emb_dim, heads=heads, layer_order=layer_order).to(self.dev) for _ in range(depth)]
        ).to(self.dev)

        # pooling - time dimension
        self.pooling = nn.MaxPool1d(self.siglen).to(self.dev)

        # Mapping of final output sequence to class logits.
        self.classification = nn.Linear(self.emb_dim, self.n_targets).to(self.dev)

    def forward(self, x):

        # b: batch size
        # l: signal length (time dimension)
        # d: embedding dimension

        # transform input to dimension [b, l, 1]
        x = torch.transpose(x, 1, 2)
        b, l, _ = x.size()

        # generate pseudo word embeddings. shape [b, l, d]
        words = self.wrd_emb(x.transpose(1, 2)).transpose(1, 2)

        # expand positional embeddings to batch size. shape [b, l, d]
        positions = self.pos_emb.expand(b, l, self.emb_dim)

        # add signal to positional embeddings. shape [b, l, d]
        embeddings = words + positions

        # apply transformer blocks to embedded signals. shape [b, l, d]
        x = self.transformer_blocks(embeddings)

        # global average pooling in time dimension. shape [b, d]
        x = self.pooling(x.transpose(1, 2)).view(b, self.emb_dim)

        # classify. shape [b, n_targets]
        x = self.classification(x)

        return x

    def comp_attention(self, x):
        """
        x: signal of shape [1, 1, l]

        Returns:
            list of attention matrices of the shape [1, h, l, l]
            prediction of forward pass
        """
        # activate evaluation mode
        self.eval()

        # forward pass
        out = self.forward(x)

        # get attention
        attention = [self.transformer_blocks[i].get_attmatt() for i in range(self.depth)]

        # prediction of forward pass
        pred = np.argmax(out.squeeze().detach().numpy())

        # set back to train mode
        self.train()

        return attention, pred

    # Code from https://www.tensorflow.org/tutorials/text/transformer
    def _get_angles(self, pos, i, d_model):
        """
        helper function for positional encoding
        """
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    # Code adapted from https://www.tensorflow.org/tutorials/text/transformer
    def _positional_encoding(self, position, d_model):
        """
        get sinusoidial encoding
        """
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return torch.from_numpy(pos_encoding).float()
