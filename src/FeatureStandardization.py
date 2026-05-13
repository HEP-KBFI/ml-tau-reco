import json

import torch
from tqdm import tqdm


class FeatureStandardization:
    def __init__(self, method, features, feature_dim, verbosity=-1):
        if verbosity >= 1:
            print("<FeatureStandardization>:")
            print(" method = %s" % method)
            print(" features = %s" % features)
            print(" feature_dim = %i" % feature_dim)

        self.method = method
        self.features = features
        self.feature_dim = feature_dim
        self.verbosity = verbosity

        self.mean = {}
        self.one_over_sigma = {}
        self.dims = {}

        self.is_one_hot_encoded = {}

    def compute_params(self, dataloader, max_jets=-1):
        if self.verbosity >= 1:
            print("<FeatureStandardization::compute_params>:")

        if self.method == "mean_rms":
            self.compute_params_mean_rms(dataloader, max_jets)
        elif self.method == "median_quantile":
            self.compute_params_median_quantile(dataloader, max_jets)
        else:
            raise RuntimeError("Invalid configuration parameter 'method' !!")

        # CV: do not transform one-hot-embedded features
        for feature in self.features:
            for idx in range(len(self.is_one_hot_encoded[feature])):
                if self.is_one_hot_encoded[feature][idx].item():
                    self.mean[feature][idx] = 0.0
                    self.one_over_sigma[feature][idx] = 1.0

        self.reshape_params()

        if self.verbosity >= 1:
            self.print()

    def compute_params_mean_rms(self, dataloader, max_jets=-1):
        xs = {}
        x2s = {}
        num_particles = 0
        num_jets_processed = 0
        if self.verbosity >= 1:
            dataloader = tqdm(dataloader, total=len(dataloader), desc="compute_params_mean_rms")
        for (X, y, weight) in dataloader:
            if max_jets != -1 and num_jets_processed >= max_jets:
                break
            
            batch_size = X["mask"].size(0)
            
            mask_for_particles = X["mask"]
            # Truncate batch if we only need a few more jets
            if max_jets != -1 and (num_jets_processed + batch_size) > max_jets:
                keep = max_jets - num_jets_processed
                mask_for_particles = mask_for_particles[:keep]

            for feature in self.features:
                x = X[feature]
                if max_jets != -1 and (num_jets_processed + batch_size) > max_jets:
                    keep = max_jets - num_jets_processed
                    x = x[:keep]

                self.dims[feature] = x.dim()

                if self.feature_dim != (x.dim() - 1):
                    x = torch.swapaxes(x, self.feature_dim, -1)
                x = x.reshape(-1, x.size(dim=-1))
                
                # Apply mask to get only real particles
                mask = mask_for_particles
                if self.feature_dim != (mask.dim() - 1):
                    mask = torch.swapaxes(mask, self.feature_dim, -1)
                mask = mask.reshape(-1, mask.size(dim=-1))
                mask = mask.ge(0.5)
                
                x_masked = torch.masked_select(x, mask)
                x_masked = x_masked.reshape(-1, x.size(dim=-1))
                
                sum_x = torch.sum(x_masked, dim=0)
                x2 = x_masked * x_masked
                sum_x2 = torch.sum(x2, dim=0)
                if feature not in xs.keys() or feature not in x2s.keys():
                    xs[feature] = sum_x
                    x2s[feature] = sum_x2
                else:
                    xs[feature] = torch.add(xs[feature], sum_x)
                    x2s[feature] = torch.add(x2s[feature], sum_x2)

                if feature not in self.is_one_hot_encoded.keys():
                    self.is_one_hot_encoded[feature] = X["%s_is_one_hot_encoded" % feature][0]

            num_particles += mask_for_particles.sum().item()
            num_jets_processed += batch_size

        # print("num_particles = %i" % num_particles)
        if num_particles == 0:
            raise RuntimeError("Dataset given as function argument is empty !!")

        for feature in self.features:
            x = torch.mul(xs[feature], 1.0 / num_particles)
            x2 = torch.mul(x2s[feature], 1.0 / num_particles)

            mean = x
            var = torch.sub(x2, x * x)
            sigma = torch.sqrt(var)
            torch.clamp(sigma, min=1.0e-8, max=None)
            one_over_sigma = torch.div(torch.tensor(1), sigma)
            if torch.isnan(one_over_sigma).sum().item() > 0.0:
                raise RuntimeError("Failed to compute standard deviation, because <x^2> - <x>^2 is negative !!")

            self.mean[feature] = mean
            self.one_over_sigma[feature] = one_over_sigma

    def compute_params_median_quantile(self, dataloader, max_jets=-1):
        feature_values = {}
        for feature in self.features:
            feature_values[feature] = []
        num_jets_processed = 0
        if self.verbosity >= 1:
            dataloader = tqdm(dataloader, total=len(dataloader), desc="compute_params_median_quantile")
        for (X, y, weight) in dataloader:
            if max_jets != -1 and num_jets_processed >= max_jets:
                break
            
            batch_size = X["mask"].size(0)
            
            for feature in self.features:
                x = X[feature]

                # Truncate batch if we only need a few more jets
                if max_jets != -1 and (num_jets_processed + batch_size) > max_jets:
                    keep = max_jets - num_jets_processed
                    x = x[:keep]
                    mask = X["mask"][:keep]
                else:
                    mask = X["mask"]

                self.dims[feature] = x.dim()

                if self.feature_dim != (x.dim() - 1):
                    x = torch.swapaxes(x, self.feature_dim, -1)
                x = x.reshape(-1, x.size(dim=-1))

                if self.feature_dim != (mask.dim() - 1):
                    mask = torch.swapaxes(mask, self.feature_dim, -1)
                mask = mask.reshape(-1, mask.size(dim=-1))

                mask = mask.ge(0.5)

                x_masked = torch.masked_select(x, mask)
                x_masked = x_masked.reshape(-1, x.size(dim=-1))

                feature_values[feature].append(x_masked.cpu())

                if feature not in self.is_one_hot_encoded.keys():
                    self.is_one_hot_encoded[feature] = X["%s_is_one_hot_encoded" % feature][0]
            
            num_jets_processed += batch_size

        for feature in self.features:
            feature_combined = torch.cat(feature_values[feature])
            median, _ = torch.median(feature_combined, dim=0)
            quantile_hi = torch.quantile(feature_combined, 0.975, dim=0)
            quantile_lo = torch.quantile(feature_combined, 0.025, dim=0)
            width = quantile_hi - quantile_lo

            self.mean[feature] = median
            sigma = 0.5 * width
            torch.clamp(sigma, min=1.0e-8, max=None)
            one_over_sigma = torch.div(torch.tensor(1), sigma)
            self.one_over_sigma[feature] = one_over_sigma

    def reshape_params(self):
        # CV: add dimensions of size one before and after the "feature dimension"
        for feature in self.features:
            for dim in range(self.dims[feature]):
                if dim < self.feature_dim:
                    self.mean[feature] = torch.unsqueeze(self.mean[feature], 0)
                    self.one_over_sigma[feature] = torch.unsqueeze(self.one_over_sigma[feature], 0)
                elif dim > self.feature_dim:
                    self.mean[feature] = torch.unsqueeze(self.mean[feature], -1)
                    self.one_over_sigma[feature] = torch.unsqueeze(self.one_over_sigma[feature], -1)

    def __call__(self, X):
        if self.verbosity >= 4:
            print("<FeatureStandardization::operator()>:")
        X_transformed = {}
        # apply transformation to requested features
        for feature in self.features:
            x = X[feature]
            if self.verbosity >= 4:
                print("before transformation: %s = " % feature, x[0])
            x_transformed = torch.sub(x, self.mean[feature])
            x_transformed = torch.mul(x_transformed, self.one_over_sigma[feature])
            X_transformed[feature] = x_transformed
            if self.verbosity >= 4:
                print("after transformation: %s = " % feature, X_transformed[feature][0])
        # add features for which no transformation is requested
        for feature in X.keys():
            if feature not in self.features:
                X_transformed[feature] = X[feature]
        return X_transformed

    def load_params(self, filename):
        if self.verbosity >= 1:
            print("<FeatureStandardization::load_params>:")
            print(" filename = %s" % filename)
        file = open(filename, "r")
        cfg = json.load(file)
        file.close()
        if self.verbosity >= 1:
            print("cfg = %s" % cfg)
        for feature in self.features:
            self.mean[feature] = torch.tensor(cfg[feature]["mean"])
            self.one_over_sigma[feature] = torch.tensor(cfg[feature]["one_over_sigma"])
            self.dims[feature] = int(cfg[feature]["dims"])

        self.reshape_params()

        if self.verbosity >= 1:
            self.print()

    def save_params(self, filename):
        if self.verbosity >= 1:
            print("<FeatureStandardization::save_params>:")
            print(" filename = %s" % filename)
        cfg = {}
        for feature in self.features:
            cfg[feature] = {}
            cfg[feature]["mean"] = [float(mean) for mean in self.mean[feature].squeeze().detach().cpu().numpy()]
            cfg[feature]["one_over_sigma"] = [
                float(one_over_sigma) for one_over_sigma in self.one_over_sigma[feature].squeeze().detach().cpu().numpy()
            ]
            cfg[feature]["dims"] = self.dims[feature]
        cfg["method"] = self.method
        if self.verbosity >= 1:
            print("cfg = %s" % cfg)
        file = open(filename, "w")
        json.dump(cfg, file)
        file.close()

    def print(self):
        print("<FeatureStandardization::print>:")
        print(" method = %s" % self.method)
        for feature in self.features:
            print("%s:" % feature)
            print(" shape(mean) = ", self.mean[feature].shape)
            print(" mean = ", self.mean[feature].squeeze())
            print(" shape(sigma) = ", self.one_over_sigma[feature].shape)
            print(" sigma = ", torch.div(torch.tensor(1), self.one_over_sigma[feature].squeeze()))
            print(" dims = ", self.dims[feature])
