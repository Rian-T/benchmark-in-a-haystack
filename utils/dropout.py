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

def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None
    
    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()
    
    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask
    
    return mask, dropout

class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None

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

class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config
    
    def forward(self, hidden_states):
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        from transformers.activations import ACT2FN
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output
    
    @property
    def output_dim(self):
        return self.config.hidden_size

def inject_stabledropout():
    try:
        import transformers.models.deberta_v2.modeling_deberta_v2 as deberta_module
    except ImportError:
        deberta_module = ModuleType('modeling_deberta_v2')
        sys.modules['transformers.models.deberta_v2.modeling_deberta_v2'] = deberta_module
    
    deberta_module.StableDropout = StableDropout
    deberta_module.DropoutContext = DropoutContext
    deberta_module.XDropout = XDropout
    deberta_module.get_mask = get_mask
    deberta_module.ContextPooler = ContextPooler

