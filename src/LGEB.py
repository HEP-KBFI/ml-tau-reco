import torch
from torch import nn
from typing import Tuple

class LGEB(nn.Module):
    def __init__(self, n_input : int, n_output : int, n_hidden : int, n_node_attr : int = 0, dropout : float = 0., c_weight : float = 1.0, last_layer : bool = False) -> None:
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2 # dims for Minkowski norm & inner product

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU())

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output))

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        if not last_layer:
            self.phi_x = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                layer)

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid())
        
        self.last_layer = last_layer

    def m_model(self, hi : torch.Tensor, hj : torch.Tensor, norms : torch.Tensor, dots : torch.Tensor) -> torch.Tensor:
        print("<LGEB::m_model>:")
        print("shape(hi) = ", hi.shape)
        print("shape(hj) = ", hj.shape)
        print("shape(norms) = ", norms.shape)
        print("shape(dots) = ", dots.shape)
        out = torch.cat([hi, hj, norms, dots], dim=-1)
        print("shape(out@1) = ", out.shape)
        out = out.view(-1, out.size(dim=-1))
        print("shape(out@2) = ", out.shape)
        out = self.phi_e(out)
        print("shape(out@3) = ", out.shape) 
        out = out.reshape(self.batchsize, -1, out.size(dim=-1))
        print("shape(out@4) = ", out.shape)
        w = self.phi_m(out)
        print("shape(w) = ", w.shape)
        out = out * w
        print("shape(out@4) = ", out.shape)
        ##out = out.reshape(self.batchsize, -1, out.size(dim=-1))
        ##print("shape(out@5) = ", out.shape)
        return out

    def h_model(self, h : torch.Tensor, segment_ids : torch.Tensor, m : torch.Tensor, node_attr : torch.Tensor) -> torch.Tensor:
        print("<LGEB::h_model>:")
        print("shape(h) = ", h.shape)
        print("shape(segment_ids) = ", segment_ids.shape)
        ##print("shape(edgej) = ", edgej.shape)
        print("shape(m) = ", m.shape)
        print("shape(node_attr) = ", node_attr.shape)
        ##x = x.view(-1, x.size(dim=-1))
        ##edgei = torch.flatten(edgei)
        ##m = m.view(-1, m.size(dim=-1))
        ##node_attr = node_attr.view(-1, node_attr.size(dim=-1))
        agg = unsorted_segment_sum(m, segment_ids, num_segments=self.n_particles) ### !!! this does not work: the edgesi need to be converted to segment_ids by adding (n_cols)*(irow-1) to each element when flattening the first dimension (batch_index). If this is not done, particles from different jets (batch_index values) get mixed up !! num_segments needs to be batchsize*n_particles
        print("shape(agg) = ", agg.shape) ### !!! the shape of agg needs to be [ batchsize*n_particles, 72 ]
        agg = torch.cat([h, agg, node_attr], dim=-1)
        print("shape(agg@1) = ", agg.shape)
        agg = agg.view(-1, agg.size(dim=-1))
        print("shape(agg@2) = ", agg.shape)
        agg = self.phi_h(agg)
        print("shape(agg@3) = ", agg.shape)
        agg = agg.reshape(self.batchsize, -1, agg.size(dim=-1))
        print("shape(agg@4) = ", agg.shape)
        out = h + agg
        print("shape(out) = ", out.shape)
        ##out = out.reshape(self.batchsize, -1, out.size(dim=-1))
        ##print("shape(out@2) = ", out.shape)
        return out

    def x_model(self, x : torch.Tensor, segment_ids : torch.Tensor, x_diff : torch.Tensor, m : torch.Tensor) -> torch.Tensor:
        print("<LGEB::x_model>:")
        print("shape(x@1) = ", x.shape)
        print("shape(segment_ids@1) = ", segment_ids.shape)
        print("shape(x_diff) = ", x_diff.shape)
        print("shape(m) = ", m.shape)
        assert hasattr(self, "phi_x")
        ##segment_ids = torch.flatten(segment_ids) ### !!! this does not work: the edgesi need to be converted to segment_ids by adding (n_cols)*(irow-1) to each element when flattening the first dimension (batch_index). If this is not done, particles from different jets (batch_index values) get mixed up !! num_segments needs to be batchsize*n_particles
        ##print("shape(segment_ids.@2) = ", segment_ids.shape)
        ##x_diff = x_diff.view(-1, x_diff.size(dim=-1))
        ##m = m.view(-1, m.size(dim=-1))
        trans = x_diff * self.phi_x(m)
        # From https://github.com/vgsatorras/egnn
        # This is never activated but just in case it explosed it may save the train
        trans = torch.clamp(trans, min=-100., max=+100.)
        print("shape(trans) = ", trans.shape)
        agg = unsorted_segment_mean(trans, segment_ids, num_segments=self.n_particles)
        print("shape(agg) = ", agg.shape) ### !!! the shape of agg needs to be [ batchsize*n_particles, 4 ]
        print("shape(self.c_weight) = ", self.c_weight)
        print("shape(agg * self.c_weight) = ", (agg * self.c_weight).shape)
        x = x + agg * self.c_weight
        print("shape(x@2) = ", x.shape)
        ##x = x.reshape(self.batchsize, -1, x.size(dim=-1))
        ##print("shape(x@3) = ", x.shape)
        return x

    def minkowski_feats(self, edgei : torch.Tensor, edgej : torch.Tensor, x : torch.Tensor) -> Tuple[ torch.Tensor, torch.Tensor, torch.Tensor ]:
        print("<LGEB::minkowski_feats>:")
        print("shape(edgei@1) = ", edgei.shape)
        print("shape(edgej@1) = ", edgej.shape)
        print("shape(x) = ", x.shape)
        ##x_diff = x[edgei] - x[edgej]
        edgei = edgei.unsqueeze(dim=2).expand(-1, -1, x.size(dim=2))
        xi = torch.gather(x, 1, edgei)
        edgej = edgej.unsqueeze(dim=2).expand(-1, -1, x.size(dim=2))
        xj = torch.gather(x, 1, edgej)
        print("shape(edgei@2) = ", edgei.shape)
        print("shape(edgej@2) = ", edgej.shape)
        x_diff = xi - xj
        print("shape(x_diff) = ", x_diff.shape)
        norms = normsq4(x_diff).unsqueeze(dim=-1)
        print("shape(norms) = ", norms.shape)
        dots = dotsq4(xi, xj).unsqueeze(dim=-1)
        print("shape(dots) = ", dots.shape)
        norms, dots = psi(norms), psi(dots)
        return norms, dots, x_diff

    def forward(self, h : torch.Tensor, x : torch.Tensor, edgei : torch.Tensor, edgej : torch.Tensor, 
                node_attr : torch.Tensor = None) -> Tuple[ torch.Tensor, torch.Tensor, torch.Tensor ]:
        print("<LGEB::forward>:")

        self.batchsize = h.size(dim=0)
        assert x.size(dim=0) == self.batchsize
        assert edgei.size(dim=0) == self.batchsize
        assert edgej.size(dim=0) == self.batchsize
        segment_ids = edgei

        self.n_particles = h.size(dim=-2)

        print("shape(h) = ", h.shape)
        print("shape(x) = ", x.shape)
        print("shape(edgei@1) = ", edgei.shape)
        print("shape(edgej@1) = ", edgej.shape)
        norms, dots, x_diff = self.minkowski_feats(edgei, edgej, x)

        #edgei = edgei.unsqueeze(dim=2).expand(-1, -1, h.size(dim=2))
        #edgej = edgej.unsqueeze(dim=2).expand(-1, -1, h.size(dim=2))
        #print("shape(edgei@2) = ", edgei.shape)
        #print("shape(edgej@2) = ", edgej.shape) 

        #edgei_squeezed = edgei.squeeze(dim=2).squeeze(dim=0)
        #print("shape(edgei_squeezed) = ", edgei_squeezed.shape)
        #hi = h[:edgei_squeezed]
        #print("shape(hi) = ", hi.shape)
        #raise ValueError("STOP.")

        #edgei_expanded = edgei.expand(h.size(dim=0), -1, h.size(dim=2))
        #print("shape(edgei_expanded) = ", edgei_expanded.shape)
        #hi = torch.gather(h, 1, edgei_expanded)
        #print("shape(hi) = ", hi.shape)
        #raise ValueError("STOP.")

        ##m = self.m_model(h[edgei], h[edgej], norms, dots) # [B*N, hidden]
        #dim = ( h.size(dim=0), edgei.size(dim=1), h.size(dim=2) )
        #print("dim = ", dim)
        ##hi = torch.tensor(dim, dtype=torch.float32, device=h.device).gather(h, 1, edgei)
        hi = torch.gather(h, 1, edgei.unsqueeze(dim=2).expand(-1, -1, h.size(dim=2)))
        print("shape(hi) = ", hi.shape)
        ##hj = torch.tensor(dim, dtype=torch.float32, device=h.device).gather(h, 1, edgej)
        hj = torch.gather(h, 1, edgej.unsqueeze(dim=2).expand(-1, -1, h.size(dim=2)))
        print("shape(hj) = ", hj.shape)
        m = self.m_model(hi, hj, norms, dots) # [B*N, hidden]

        if not self.last_layer:
            x = self.x_model(x, segment_ids, x_diff, m)
        h = self.h_model(h, segment_ids, m, node_attr)
        return h, x, m

def unsorted_segment_sum(data : torch.Tensor, segment_ids : torch.Tensor, num_segments : int) -> torch.Tensor:
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    '''
    print("<unsorted_segment_sum>:")
    print(" shape(data) = ", data.shape)
    print(" shape(segment_ids@1) = ", segment_ids.shape)
    print(" num_segments = %i" % num_segments)
    segment_ids = torch.nn.functional.one_hot(segment_ids, num_segments)
    print(" shape(segment_ids@2) = ", segment_ids.shape)
    segment_ids = torch.transpose(segment_ids, -2, -1).float()
    print(" shape(segment_ids@3) = ", segment_ids.shape)
    # CV: the following einsum operator gives the same as the operation
    #       'result = segment_ids @ data'
    #     but makes it more explicit how to multiply tensors in three dimensions
    result = torch.einsum('ijk,ikl->ijl', segment_ids, data)
    return result

def unsorted_segment_mean(data : torch.Tensor, segment_ids : torch.Tensor, num_segments : int) -> torch.Tensor:
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    '''
    print("<unsorted_segment_mean>:")
    print(" shape(data) = ", data.shape)
    print(" shape(segment_ids@1) = ", segment_ids.shape)
    print(" num_segments = %i" % num_segments)
    segment_ids = torch.nn.functional.one_hot(segment_ids, num_segments)
    print(" shape(segment_ids@2) = ", segment_ids.shape)
    segment_ids = torch.transpose(segment_ids, -2, -1).float()
    print(" shape(segment_ids@3) = ", segment_ids.shape)
    # CV: the following einsum operators give the same as the operations
    #       'result = segment_ids @ data' and 'count = segment_ids @ torch.ones_like(data)',
    #     but make it more explicit how to multiply tensors in three dimensions
    result = torch.einsum('ijk,ikl->ijl', segment_ids, data)
    count = torch.einsum('ijk,ikl->ijl', segment_ids, torch.ones_like(data))
    result = result / count.clamp(min=1)
    return result

def normsq4(p : torch.Tensor) -> torch.Tensor:
    r''' Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    '''
    psq = torch.pow(p, 2)
    result = 2 * psq[..., 0] - psq.sum(dim=-1)
    return result
    
def dotsq4(p : torch.Tensor, q : torch.Tensor) -> torch.Tensor:
    r''' Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    '''
    psq = p*q
    result = 2 * psq[..., 0] - psq.sum(dim=-1)
    return result
    
def psi(p : torch.Tensor) -> torch.Tensor:
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    '''
    result = torch.sign(p) * torch.log(torch.abs(p) + 1)
    return result

