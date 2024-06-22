import torch
import torch.nn as nn
import pyro.distributions as dist
from .base import FullyConnected

class DistributionNet(nn.Module):
    """
    Base class for distribution nets.
    """
    @staticmethod
    def get_class(dtype):
        """
        Get a subclass by a prefix of its name, e.g.::

            assert DistributionNet.get_class("bernoulli") is BernoulliNet
        """
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))


class BernoulliNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = BernoulliNet([3, 4])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=-10, max=10)
        return logits,

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class ExponentialNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``rate``.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = ExponentialNet([3, 4])
        x = torch.randn(3)
        rate, = net(x)
        y = net.make_dist(rate).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        scale = nn.functional.softplus(self.fc(x).squeeze(-1)).clamp(min=1e-3, max=1e6)
        rate = scale.reciprocal()
        return rate,

    @staticmethod
    def make_dist(rate):
        return dist.Exponential(rate)


class LaplaceNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Laplace random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = LaplaceNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Laplace(loc, scale)


class NormalNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = NormalNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)


class StudentTNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``df,loc,scale``
    triple, with shared ``df > 1``.

    This is used to represent a conditional probability distribution of a
    single Student's t random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = StudentTNet([3, 4])
        x = torch.randn(3)
        df, loc, scale = net(x)
        y = net.make_dist(df, loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])
        self.df_unconstrained = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        df = nn.functional.softplus(self.df_unconstrained).add(1).expand_as(loc)
        return df, loc, scale

    @staticmethod
    def make_dist(df, loc, scale):
        return dist.StudentT(df, loc, scale)
    

class DiagNormalNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    ``sizes[-1]``-sized diagonal Normal random variable conditioned on a
    ``sizes[0]``-size real value, for example::

        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()

    This is intended for the latent ``z`` distribution and the prewhitened
    ``x`` features, and conservatively clips ``loc`` and ``scale`` values.
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        return loc, scale


class DiagBernoulliNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = DiagBernoulliNet([3, 4, 5])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=0, max=11)
        return logits

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)
