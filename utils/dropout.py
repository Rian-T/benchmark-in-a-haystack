import sys
import torch
import torch.nn as nn
from types import ModuleType


class DropoutContext:
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True

class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx):
        if isinstance(local_ctx, float):
            p = local_ctx
            mask = torch.bernoulli(torch.ones_like(input) * (1 - p))
        else:
            p = local_ctx.dropout
            if local_ctx.reuse_mask and local_ctx.mask is not None:
                mask = local_ctx.mask
            else:
                mask = torch.bernoulli(torch.ones_like(input) * (1 - p))
                if local_ctx.reuse_mask:
                    local_ctx.mask = mask
        ctx.scale = 1 / (1 - p) if not isinstance(local_ctx, float) else 1 / (1 - p)
        return input * mask * ctx.scale
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

class StableDropout(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None
    
    def forward(self, x):
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x
    
    def clear_context(self):
        self.count = 0
        self.context_stack = None
    
    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale
    
    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob

def inject_stabledropout():
    if 'transformers.models.deberta_v2.modeling_deberta_v2' not in sys.modules:
        sys.modules['transformers.models.deberta_v2.modeling_deberta_v2'] = ModuleType('modeling_deberta_v2')
    sys.modules['transformers.models.deberta_v2.modeling_deberta_v2'].StableDropout = StableDropout

