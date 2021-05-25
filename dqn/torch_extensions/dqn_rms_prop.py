import torch
from torch.optim.optimizer import Optimizer


class DQNRMSprop(Optimizer):
    r"""Implements RMSprop algorithm used by the authors of "Human-level control through deep reinforcemnet learning"""

    def __init__(self, params, lr=0.00025, g_momentum=0.95, sg_momentum=0.95, eps=0.1, weight_decay=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= g_momentum <= 1.0:
            raise ValueError("Invalid momentum value: {}".format(g_momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= sg_momentum <= 1.0:
            raise ValueError("Invalid alpha value: {}".format(g_momentum))

        defaults = dict(lr=lr, g_momentum=g_momentum, sg_momentum=sg_momentum, eps=eps, centered=centered,
                        weight_decay=weight_decay)
        super(DQNRMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DQNRMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('DQNRMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['g'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['g2'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                g = state['g']
                g2 = state['g2']
                sg_momentum = group['sg_momentum']
                g_momentum = group['g_momentum']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                g.mul_(g_momentum).add_(grad, alpha=1 - g_momentum)
                g2.mul_(sg_momentum).addcmul_(grad, grad, value=1 - sg_momentum)

                avg = g2.addcmul_(-g, g, value=1).add_(group['eps']).sqrt()

                p.addcdiv_(grad, avg, value=-group['lr'])

        return loss
