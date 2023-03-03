import torch
from torch import nn

from LGEB import LGEB


class LorentzNet(nn.Module):
    r"""Implementation of LorentzNet.

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of LGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    """

    def __init__(
        self,
        n_scalar: int,
        n_hidden: int,
        n_class: int = 2,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
        verbosity: int = 0,
    ) -> None:
        if verbosity >= 2:
            print("<LorentzNet::LorentzNet>:")
            print(" n_scalar = %i" % n_scalar)
            print(" n_hidden = %i" % n_hidden)
            print(" n_class = %i" % n_class)
            print(" n_layers = %i" % n_layers)
            print(" c_weight = %1.3f" % c_weight)
            print(" dropout = %1.2f" % dropout)
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList(
            [
                LGEB(
                    self.n_hidden,
                    self.n_hidden,
                    self.n_hidden,
                    n_node_attr=n_scalar,
                    dropout=dropout,
                    c_weight=c_weight,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.graph_dec = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.n_hidden, n_class)
        )  # classification

        self.verbosity = verbosity

    def forward(self, x: torch.Tensor, scalars: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # print("<LorentzNet::forward>")
        # print("shape(x) = ", x.shape)
        # print("shape(scalars) = ", scalars.shape)
        # print("shape(node_mask) = ", node_mask.shape)

        h = self.embedding(scalars)
        # print("shape(h) = ", h.shape)

        # batchsize = x.size(dim=0)
        # print("batchsize = %i" % batchsize)
        n_particles = x.size(dim=1)
        # print("n_particles = %i" % n_particles)

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
        # print("shape(edgei) = ", edgei.shape)
        # print("shape(edgej) = ", edgej.shape)

        for i in range(self.n_layers):
            h, x, _ = self.LGEBs[i].forward(h, x, edgei, edgej, node_attr=scalars)

        h = h * node_mask
        h = h.view(-1, n_particles, self.n_hidden)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        result = pred.squeeze(0)
        # print("shape(result) = ", result.shape)
        # print("result = ", result)
        return result
