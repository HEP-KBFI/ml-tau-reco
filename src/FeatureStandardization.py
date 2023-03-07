import json

import torch

class FeatureStandardization():
    def __init__(self, features, dim, verbosity = -1):
        self.features = features
        self.dim = dim
        self.verbosity = verbosity       
        self.mean = {}
        self.one_over_sigma = {}

    def compute_params(self, dataloader):
        print("<FeatureStandardization::compute_params>:")        

        dims = {}
        shapes = {}
        xs = {}
        x2s = {}
        num_particles = 0
        for (X, y, weight) in dataloader:
            for feature in self.features:
                x = X[feature]
                print("shape(%s) = " % feature, x.shape)

                dims[feature] = x.dim()
                shapes[feature] = x.shape

                if self.dim != (x.dim() - 1):
                    x = torch.swapaxes(x, self.dim, -1)
                x = x.reshape(-1, x.size(dim=-1))
                sum_x = torch.sum(x, dim=0)
                x2 = x * x
                sum_x2 = torch.sum(x2, dim=0)
                if feature not in xs.keys() or feature not in x2s.keys():
                    xs[feature] = sum_x
                    x2s[feature] = sum_x2
                else:
                    xs[feature] = torch.add(xs[feature], sum_x)
                    x2s[feature] = torch.add(x2s[feature], sum_x2)

            mask = X["mask"]
            num_particles += mask.sum().item()

        print("num_particles = %i" % num_particles)
        if num_particles == 0:
            raise RuntimeError("Dataset given as function argument is empty !!")

        for feature in self.features:
            x = torch.mul(xs[feature], 1./num_particles)
            x2 = torch.mul(x2s[feature], 1./num_particles)

            mean = x
            print("mean = ", mean)
            var = torch.sub(x2, x * x)
            sigma = torch.sqrt(var)
            print("sigma = ", sigma)
            one_over_sigma = torch.div(torch.tensor(1), torch.sqrt(var))
            if torch.isnan(one_over_sigma).sum().item() > 0.:
                raise RuntimeError("Failed to compute standard deviation, because <x^2> - <x>^2 is negative !!")

            self.mean[feature] = mean
            self.one_over_sigma[feature] = one_over_sigma
            for dim in range(dims[feature]):
                if dim < self.dim:
                     self.mean[feature] = torch.unsqueeze(self.mean[feature], 0)
                     self.one_over_sigma[feature] = torch.unsqueeze(self.one_over_sigma[feature], 0)
                elif dim > self.dim:
                     self.mean[feature] = torch.unsqueeze(self.mean[feature], -1)
                     self.one_over_sigma[feature] = torch.unsqueeze(self.one_over_sigma[feature], -1)

        if self.verbosity >= 1:
            self.print()

    def __call__(self, X):
        X_transformed = {}
        for feature in self.features:
            x = X[feature]
            print("before transformation: %s = " % feature, x)
            x_transformed = torch.sub(x, self.mean[feature])
            x_transformed = torch.mul(x_transformed, self.one_over_sigma[feature])
            X_transformed[feature] = x_transformed
            print("after transformation: %s = " % feature, X_transformed[feature])
        raise ValueError("STOP.")
        return X_transformed

    def load_params(self, filename):
        print("<FeatureStandardization::load_params>:")
        print(" filename = %s" % filename)
        cfg = json.load(filename)
        for feature in self.features:
            self.mean[feature] = torch.tensor(cfg[feature]['mean'])
            self.one_over_sigma[feature] = torch.tensor(cfg[feature]['one_over_sigma'])

        if self.verbosity >= 1:
            self.print()

    def save_params(self, filename):
        print("<FeatureStandardization::save_params>:")
        print(" filename = %s" % filename)
        file = open(filename, "w")
        cfg = {}
        for feature in self.features:
            cfg[feature] = {}
            cfg[feature]['mean'] = list(self.mean[feature].squeeze().detach().cpu().numpy())
            cfg[feature]['one_over_sigma'] = list(self.one_over_sigma[feature].squeeze().detach().cpu().numpy())
        print("cfg = %s" % cfg)
        file.write("%s" % cfg)
        file.close()

    def print(self):
        for feature in self.features:
            print("%s:" % feature)
            print("shape(mean) = ", self.mean[feature].shape)
            print("mean = ", self.mean[feature].squeeze())
            print("shape(sigma) = ", self.one_over_sigma[feature].shape)
            print("sigma = ", torch.div(torch.tensor(1), self.one_over_sigma[feature].squeeze()))
