import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from .distnet import DistributionNet, BernoulliNet, DiagNormalNet, DiagBernoulliNet

class Model(PyroModule):
    """
    Generative model for a causal model with latent confounder ``z`` and binary
    treatment ``t``::

        z ~ p(z)      # latent confounder
        zt ~ p(zt)
        zy ~ p（zy)
        x ~ p(x|z,zt,zy)
        t ~ p(t|z,zt)
        y ~ p(y|t,z,zy)

    Each of these distributions is defined by a neural network.  The ``y``
    distribution is defined by a disjoint pair of neural networks defining
    ``p(y|t=0,z,zy)`` and ``p(y|t=1,z,zy)``; this allows highly imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """
    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        self.latent_dim_t = config["latent_dim_t"]
        self.latent_dim_y = config["latent_dim_y"]
        self.binfeats = config["binary_dim"]
        self.contfeats = config["continuous_dim"]

        super().__init__()
        self.x_nn = DiagNormalNet([config["latent_dim"]+config["latent_dim_t"]+config["latent_dim_y"]] +
                                  [config["hidden_dim"]] * config["num_layers"] +
                                  [len(config["continuous_dim"])])
        self.x2_nn = DiagBernoulliNet([config["latent_dim"]+config["latent_dim_t"]+config["latent_dim_y"]] +
                                  [config["hidden_dim"]] * config["num_layers"] +
                                  [len(config["binary_dim"])])
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        # The y network is split between the two t values.
        self.y0_nn = OutcomeNet([config["latent_dim"]+config["latent_dim_y"]] +
                                [config["hidden_dim"]] * config["num_layers"])
        self.y1_nn = OutcomeNet([config["latent_dim"]+config["latent_dim_y"]] +
                                [config["hidden_dim"]] * config["num_layers"])
        self.t_nn = BernoulliNet([config["latent_dim"]+config["latent_dim_t"]])

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            z = pyro.sample("z", self.z_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())

            # x = pyro.sample("x", self.x_dist(z, zt, zy), obs=x)
            x_binary = pyro.sample("x_bin", self.x_dist_binary(z, zt, zy), obs=x[:,self.binfeats])
            x_continuous = pyro.sample("x_cont", self.x_dist_continuous(z, zt, zy), obs=x[:,self.contfeats])
            x = torch.cat((x_binary,x_continuous), -1)
            # x = pyro.sample("x", self.x_dist_binary(z, zt, zy), obs=x)
            t = pyro.sample("t", self.t_dist(z, zt), obs=t)
            y = pyro.sample("y", self.y_dist(t, z, zy), obs=y)

        return y

    def y_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            z = pyro.sample("z", self.z_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())
            x_binary = pyro.sample("x_bin", self.x_dist_binary(z, zt, zy), obs=x[:,self.binfeats])
            x_continuous = pyro.sample("x_cont", self.x_dist_continuous(z, zt, zy), obs=x[:,self.contfeats])
            x = torch.cat((x_binary,x_continuous), -1)
            t = pyro.sample("t", self.t_dist(z, zt), obs=t)
        return self.y_dist(t, z, zy).mean

    def z_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim]).to_event(1)
    
    def zt_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_t]).to_event(1)

    def zy_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_y]).to_event(1)
    
    def x_dist_continuous(self, z, zt, zy):
        z_concat = torch.cat((z,zt,zy), -1)
        loc, scale = self.x_nn(z_concat)
        return dist.Normal(loc, scale).to_event(1)
    
    def x_dist_binary(self, z, zt, zy):
        z_concat = torch.cat((z,zt,zy), -1)
        logits = self.x2_nn(z_concat)
        return dist.Bernoulli(logits=logits).to_event(1)


    def y_dist(self, t, z, zy):
        # Parameters are not shared among t values.
        z_concat = torch.cat((z, zy), -1)
        params0 = self.y0_nn(z_concat)
        params1 = self.y1_nn(z_concat)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def t_dist(self, z,zt):
        z_concat = torch.cat((z,zt), -1)
        logits, = self.t_nn(z_concat)
        return dist.Bernoulli(logits=logits)
