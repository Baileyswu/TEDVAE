
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pyro
from pyro import poutine
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan
from pyro.infer import SVI
from .pmodel import Model
from .guide import Guide
from .loss import TraceCausalEffect_ELBO

class TEDVAE(nn.Module):
    def __init__(self, feature_dim, continuous_dim, binary_dim, outcome_dist="normal", 
                 latent_dim=20, latent_dim_t=20, latent_dim_y=20 , hidden_dim=200, num_layers=3, num_samples=100):
        config = dict(feature_dim=feature_dim, latent_dim=latent_dim,
                      latent_dim_t = latent_dim_t, latent_dim_y = latent_dim_y,
                      hidden_dim=hidden_dim, num_layers=num_layers, continuous_dim = continuous_dim, binary_dim = binary_dim,
                      num_samples=num_samples)
        # for name, size in config.items():
        #     if not (isinstance(size, int) and size > 0):
        #         raise ValueError("Expected {} > 0 but got {}".format(name, size))
        config["outcome_dist"] = outcome_dist
        self.feature_dim = feature_dim
        self.num_samples = num_samples

        super().__init__()
        self.model = Model(config)
        self.guide = Guide(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def fit(self, x, t, y,
            num_epochs=100,
            batch_size=100,
            learning_rate=1e-3,
            learning_rate_decay=0.1,
            weight_decay=1e-4):
        """
        Train using :class:`~pyro.infer.svi.SVI` with the
        :class:`TraceCausalEffect_ELBO` loss.

        :param ~torch.Tensor x:
        :param ~torch.Tensor t:
        :param ~torch.Tensor y:
        :param int num_epochs: Number of training epochs. Defaults to 100.
        :param int batch_size: Batch size. Defaults to 100.
        :param float learning_rate: Learning rate. Defaults to 1e-3.
        :param float learning_rate_decay: Learning rate decay over all epochs;
            the per-step decay rate will depend on batch size and number of epochs
            such that the initial learning rate will be ``learning_rate`` and the final
            learning rate will be ``learning_rate * learning_rate_decay``.
            Defaults to 0.1.
        :param float weight_decay: Weight decay. Defaults to 1e-4.
        :return: list of epoch losses
        """
        assert x.dim() == 2 and x.size(-1) == self.feature_dim
        assert t.shape == x.shape[:1]
        assert y.shape == y.shape[:1]
        # self.whiten = PreWhitener(x)

        dataset = TensorDataset(x, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logging.info("Training with {} minibatches per epoch".format(len(dataloader)))
        num_steps = num_epochs * len(dataloader)
        optim = ClippedAdam({"lr": learning_rate,
                             "weight_decay": weight_decay,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        svi = SVI(self.model, self.guide, optim, TraceCausalEffect_ELBO())
        losses = []
        for i in tqdm(range(num_epochs), desc=f'train'):
            for x, t, y in dataloader:
                # x = self.whiten(x)
                loss = svi.step(x, t, y, size=len(dataset)) / len(dataset)
                # print(loss)
                logging.debug("step {: >5d} loss = {:0.6g}".format(len(losses), loss))
                assert not torch_isnan(loss)
                losses.append(loss)
        return losses

    @torch.no_grad()
    def ite(self, x, ym, ys, num_samples=None, batch_size=None):
        r"""
        Computes Individual Treatment Effect for a batch of data ``x``.

        .. math::

            ITE(x) = \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=1) \bigr]
                   - \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=0) \bigr]

        This has complexity ``O(len(x) * num_samples ** 2)``.

        :param ~torch.Tensor x: A batch of data.
        :param int num_samples: The number of monte carlo samples.
            Defaults to ``self.num_samples`` which defaults to ``100``.
        :param int batch_size: Batch size. Defaults to ``len(x)``.
        :return: A ``len(x)``-sized tensor of estimated effects.
        :rtype: ~torch.Tensor
        """
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        logging.info("Evaluating {} minibatches".format(len(dataloader)))
        result = []
        for x in dataloader:
            # x = self.whiten(x)
            with pyro.plate("num_particles", num_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0 = poutine.replay(self.model.y_mean, tr.trace)(x) * ys + ym
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1 = poutine.replay(self.model.y_mean, tr.trace)(x) * ys + ym
            ite = (y1 - y0).mean(0)
            if not torch._C._get_tracing_state():
                logging.debug("batch ate = {:0.6g}".format(ite.mean()))
            result.append(ite)
        return torch.cat(result)

    def to_script_module(self):
        """
        Compile this module using :func:`torch.jit.trace_module` ,
        assuming self has already been fit to data.

        :return: A traced version of self with an :meth:`ite` method.
        :rtype: torch.jit.ScriptModule
        """
        self.train(False)
        fake_x = torch.randn(2, self.feature_dim)
        with pyro.validation_enabled(False):
            # Disable check_trace due to nondeterministic nodes.
            result = torch.jit.trace_module(self, {"ite": (fake_x,)}, check_trace=False)
        return result
