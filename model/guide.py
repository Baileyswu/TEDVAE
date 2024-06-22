import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from .distnet import DistributionNet, BernoulliNet, DiagNormalNet
from .base import FullyConnected


class Guide(PyroModule):
    """
    Inference model for causal effect estimation with latent confounder ``z``
    and binary treatment ``t``::
        z ~ p(z|x)  # latent confounder, an embedding
        zt ~ p(zt|x)
        zy ~ p(zy|x)
        t ~ p(t|z,zt) # treatment
        y ~ p(y|t,z,zy)    # outcome

    Each of these distributions is defined by a neural network.  The ``y`` and
    ``z`` distributions are defined by disjoint pairs of neural networks
    defining ``p(-|t=0,...)`` and ``p(-|t=1,...)``; this allows highly
    imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """
    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        self.latent_dim_t = config["latent_dim_t"]
        self.latent_dim_y = config["latent_dim_y"]
        
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        super().__init__()
        # self.t_nn = BernoulliNet([config["feature_dim"]])
        self.t_nn = BernoulliNet([config["latent_dim"]+config["latent_dim_t"]])

        # The y and z networks both follow an architecture where the first few
        # layers are shared for t in {0,1}, but the final layer is split
        # between the two t values.
        self.y_nn = FullyConnected([config["latent_dim"] + config["latent_dim_y"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        self.y0_nn = OutcomeNet([config["hidden_dim"]])
        self.y1_nn = OutcomeNet([config["hidden_dim"]])
        
        self.z_nn = FullyConnected([config["feature_dim"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        

        self.z_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        
        
        self.zt_nn = FullyConnected([config["feature_dim"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        
        self.zt_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim_t"]])
        
        
        self.zy_nn = FullyConnected([config["feature_dim"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        self.zy_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim_y"]])

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            # The t and y sites are needed for prediction, and participate in
            # the auxiliary CEVAE loss. We mark them auxiliary to indicate they
            # do not correspond to latent variables during training.
            z=pyro.sample("z", self.z_dist(x))
            zt=pyro.sample("zt", self.zt_dist(x))
            zy=pyro.sample("zy", self.zy_dist(x))
            
            t = pyro.sample("t", self.t_dist(z,zt), obs=t, infer={"is_auxiliary": True})

            y = pyro.sample("y", self.y_dist(t,z,zy), obs=y, infer={"is_auxiliary": True})
            # The z site participates only in the usual ELBO loss.

    def z_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            z = pyro.sample("z", self.z_dist(x))
            zt = pyro.sample("zt", self.zt_dist(x))
            zy = pyro.sample("zy", self.zy_dist(x))
        return z,zt,zy

    def t_dist(self, z, zt):
        input_concat = torch.cat((z,zt),-1)
        logits, = self.t_nn(input_concat)
        return dist.Bernoulli(logits=logits)

    def y_dist(self, t, z, zy):
        # The first n-1 layers are identical for all t values.
        x = torch.cat((z,zy),-1)
        hidden = self.y_nn(x)
        # In the final layer params are not shared among t values.
        params0 = self.y0_nn(hidden)
        params1 = self.y1_nn(hidden)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def z_dist(self, x):
        # hidden = self.z_nn(x)
        hidden = self.z_nn(x.float())
        params = self.z_out_nn(hidden)
        return dist.Normal(*params).to_event(1)

    def zt_dist(self, x):
        hidden = self.zt_nn(x.float())
        params = self.zt_out_nn(hidden)
        return dist.Normal(*params).to_event(1)
    
    def zy_dist(self, x):
        hidden = self.zy_nn(x.float())
        params = self.zy_out_nn(hidden)
        return dist.Normal(*params).to_event(1)
