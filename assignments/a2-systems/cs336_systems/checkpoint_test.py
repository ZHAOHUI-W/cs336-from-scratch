import torch

from cs336_basics.transformer import TransformerBlock
from cs336_basics.rope import RoPE
from cs336_basics.grad_clip import gradient_clipping

from cs336_basics.cross_entropy import cross_entropy
from torch.utils.checkpoint import checkpoint
from cs336_basics.adamw import AdamW
# num_layers for this model is 32
d_model, d_ff, num_heads, context_length = 768, 3072, 12, 256
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,max_seq_len=context_length,theta=10000,device="cuda",dtype=torch.float)

positional_encoder=RoPE(d_k=d_model // num_heads,theta=10000, max_seq_len=context_length,device="cuda")
optimizer = AdamW(block.parameters(),lr=1e-3)
grad_clip = 1
# Fuse as much torch.compile will allow
# block = torch.compile(block, fullgraph=True,backend="aot_eager")
x = torch.randn((4, context_length, d_model), requires_grad=True,device="cuda")

# logits = block(x)
# y = torch.randn((4, context_length, d_model), requires_grad=True,device="cuda")

# loss = cross_entropy(logits,y)
# loss.backward()



# Now logs the number of bytes saved
total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter): # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
def synchronize(device):
    """Synchronize the given device; do nothing on CPU (torch.cuda.synchronize would fail)."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
# Run forward pass, saving for backward
# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
def four_blocks(x):
 x = block(x)
 x = block(x)
 x = block(x)
 x = block(x)
 return x
def four_blocks_checkpoint(x):
 # checkpoint throws out all the saved tensors until the backward pass
 # when getting to the checkpointed block in the backward pass,
 # it reruns a forward pass to produce the saved tensors,
 # then completes normal backward pass.
 x = checkpoint(four_blocks, x, use_reentrant=False)
 x = checkpoint(four_blocks, x, use_reentrant=False)
 return x



x.grad = None
optimizer.zero_grad()
synchronize(device="cuda")
torch.cuda.reset_peak_memory_stats()

y_normal = four_blocks(x)
loss = y_normal.sum()
# y_check = checkpoint(block,four_blocks_checkpoint(x),use_reentrant=False)
# loss_check = y_check.sum()
# optimizer.zero_grad()
loss.backward()
synchronize(device="cuda")
peak_memory = torch.cuda.max_memory_allocated(device="cuda")
# gradient_clipping(parameters=block.parameters(),max_l2_norm=grad_clip)
# optimizer.step()
# x_copy = x.grad.clone()

# y_copy = y_normal.detach().clone()
# para_list = []
# for p in block.parameters():
#     if p.grad is not None:
#         para_list.append(p.grad.clone())
#     else:
#         para_list.append(None)


    
    # y_check = checkpoint(block,x,use_reentrant=False)
    # loss_check = y_check.sum()
    # optimizer.zero_grad()
    # loss_check.backward()

    # print(torch.allclose(x.grad,x_copy))
    # print(torch.allclose(y_copy,y_check))
    # for i,p in zip(block.parameters(),para_list):
    #     if i.grad is not None and p is not None:
    #         print(torch.allclose(i.grad,p))
    #     elif i.grad is None and p is None:
    #         print("True")
    #     else:
    #         print("False")
    
            

print(f"Total size of peak tensors in single TransformerBlock: {peak_memory / (1024**2):.2f} MiB")