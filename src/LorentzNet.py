import torch
from torch import nn
from typing import Tuple

from LGEB import LGEB

class LorentzNet(nn.Module):
    r''' Implimentation of LorentzNet.

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of LGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    '''
    def __init__(self, n_scalar : int, n_hidden : int, n_class : int = 2, n_layers : int = 6, c_weight : float = 1e-3, dropout : float = 0.) -> None:
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([LGEB(self.n_hidden, self.n_hidden, self.n_hidden, 
                                    n_node_attr=n_scalar, dropout=dropout,
                                    c_weight=c_weight, last_layer=(i==n_layers-1))
                                    for i in range(n_layers)])

        self.graph_dec = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(self.n_hidden, n_class)) # classification

    def forward(self, scalars : torch.Tensor, x : torch.Tensor) -> torch.Tensor:
        #print("<LorentzNet::forward>")

        h = self.embedding(scalars)

        batchsize = x.size(dim=0)
        n_particles = x.size(dim=1)

        edges = torch.ones(n_particles, n_particles, dtype=torch.long, device=h.device)
        edges_above_diag = torch.triu(edges, diagonal=1)
        edges_below_diag = torch.tril(edges, diagonal=-1)
        edges = torch.add(edges_above_diag, edges_below_diag)
        edges = torch.nonzero(edges)
        edges = torch.swapaxes(edges, 0, 1)
        edgei = torch.unsqueeze(edges[0], dim=0)
        edgei = edgei.expand(h.size(dim=0), -1)
        edgej = torch.unsqueeze(edges[1], dim=0)
        edgej = edgej.expand(h.size(dim=0), -1)

        node_mask = torch.ones(n_particles, dtype=torch.bool, device=h.device).unsqueeze(dim=1)
        node_mask = node_mask.expand(batchsize, -1, -1)

        for i in range(self.n_layers):
            h, x, _ = self.LGEBs[i].forward(h, x, edgei, edgej, node_attr=scalars)

        h = h * node_mask
        h = h.view(-1, n_particles, self.n_hidden)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        result = pred.squeeze(0)
        #print("shape(result) = ", result.shape)
        #print("result = ", result)
        return result



