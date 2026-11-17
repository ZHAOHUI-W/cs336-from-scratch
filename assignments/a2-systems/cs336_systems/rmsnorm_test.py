import torch
from torch import nn
# from cs336_basics.rmsnorm import RMSNorm



x = torch.randn((4, 512, 2560), requires_grad=True,device="cuda")

def rmsnorm_fwd(x:torch.Tensor, weight:torch.Tensor, eps: float):
    rms = torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)
    x = x * rms
    return weight * x

rms_fwd_compiled = torch.compile(rmsnorm_fwd,backend="aot_eager")
class RMSNorm(nn.Module):
    def __init__(
    self,
    hidden_size: int,
    eps: float = 1e-5,
    device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps
    def forward(self, x):
        return rms_fwd_compiled(x,self.weight,self.eps)


@torch.compiler.disable    
def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
@torch.compiler.disable
def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


ln = RMSNorm(x.shape[-1],device="cuda")

# ln_compiled = torch.compile(ln)
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()