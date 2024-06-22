import torch
from pyro.infer import Trace_ELBO
from pyro.infer.util import torch_item


class TraceCausalEffect_ELBO(Trace_ELBO):
    """
    Loss function for training a :class:`TEDVAE`.
    From [1], the CEVAE objective (to maximize) is::

        -loss = ELBO + log q(t|z,zt) + log q(y|t,z,zy)
    """
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        # Construct -ELBO part.
        blocked_names = [name for name, site in guide_trace.nodes.items()
                         if site["type"] == "sample" and site["is_observed"]]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        loss, surrogate_loss = super()._differentiable_loss_particle(
            model_trace, blocked_guide_trace)

        # Add log q terms.
        for name in blocked_names:
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            loss = loss - 100* torch_item(log_q)
            surrogate_loss = surrogate_loss - 100* log_q

        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))

