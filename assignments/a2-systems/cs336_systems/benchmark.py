import torch
import numpy as np
import timeit
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.load_dataset import data_loading
from cs336_basics.grad_clip import gradient_clipping
import argparse
import torch.cuda.nvtx as nvtx
NUM_LAYERS = 4
BATCH_SIZE = 4

train_dataset = ""
batch_size = BATCH_SIZE
grad_clip = 1
CONFIG = dict(
    vocab_size = 10000,
    context_length = 256,
    d_model = 512,
    num_layers = NUM_LAYERS,
    num_heads = 16,
    d_ff = 1344,
    rope_theta = 10000,
)

# Table 1 in the handout (2.1.2 Model Sizing): --model-size picks one of these presets.
MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def synchronize(device):
    """Synchronize the given device; do nothing on CPU (torch.cuda.synchronize would fail)."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CS336 A2 end-to-end benchmark")
    parser.add_argument("--model-size", choices=tuple(MODEL_CONFIGS), default="small")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--w", type=int, default=10, help="number of warm-up steps (>= 0)")
    parser.add_argument("--n", type=int, default=50, help="number of timed steps (> 0)")
    parser.add_argument("--options", type=int, choices=(0, 1, 2), required=True,
                        help="0: forward only, 1: forward and backward, 2: forward and backward with optimizer step")
    parser.add_argument("--device", default="cuda", help='"cuda", "cuda:1" or "cpu"')
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error(f"--n must be > 0 (got {args.n})")
    if args.w < 0:
        parser.error(f"--w must be >= 0 (got {args.w})")
    if args.batch_size <= 0:
        parser.error(f"--batch-size must be > 0 (got {args.batch_size})")
    if args.context_length <= 0:
        parser.error(f"--context-length must be > 0 (got {args.context_length})")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("--device requests CUDA, but CUDA is not available; use --device cpu")
    return args


def benchmark(w,n,options:int | None = None ,device="cuda", model_config=None, batch_size=None):

    if n <= 0:
        raise ValueError(f"n must be > 0 (got {n})")
    if w < 0:
        raise ValueError(f"w must be >= 0 (got {w})")
    if options not in (0, 1, 2):
        raise ValueError(f"options must be one of 0, 1, 2 (got {options!r})")

    device = torch.device(device)
    config = dict(CONFIG) if model_config is None else dict(model_config)
    bs = BATCH_SIZE if batch_size is None else batch_size


    model = TransformerLM(**config,device=device)
    optimizer = AdamW(model.parameters(),lr=1e-3)


    x =  torch.randint(0,config["vocab_size"],size=(bs,config["context_length"]))
    y =  torch.randint(0,config["vocab_size"],size=(bs,config["context_length"]))
    
    x = x.to(dtype=torch.long,device=device)
    y = y.to(dtype=torch.long, device=device)

    
    # 前向
    if options == 0x00:
        # 预热
        with nvtx.range("warmup_forward"):
            for t in range(w):
                with nvtx.range("warmup_forward_step_{}",t):
                    model.forward(x)
            
            synchronize(device)


        # start_time = timeit.default_timer()
        times = []
        with nvtx.range("measurement_forward"):
            for i in range(n):
                synchronize(device)
                with nvtx.range("measurement_forward_step_{}",i):
                    start_time = timeit.default_timer()
                    with nvtx.range("single_forward"):
                        model.forward(x)
                    synchronize(device)
                    end_time = timeit.default_timer()
                    time = end_time - start_time
                    times.append(time)
            times = np.array(times)
            avg_time = times.mean()
            std_time = times.std()
            print(f"n={n} iterations | Avg: {avg_time:.4f} s | Std: {std_time:.4f} s")
        return avg_time, std_time, times

    
    
    # 前向+反向
    elif options == 0x01:
        with nvtx.range("warmup_forward+backward"):
            for t in range(w):
                with nvtx.range("warmup_forward+backward_step_{}",t):
                    # model.forward(x)
                    with nvtx.range("single_warmup_forward"):
                        logits = model(x)
                    loss = cross_entropy(logits,y)
                    optimizer.zero_grad()
                    with nvtx.range("single_warmup_backward"):
                        loss.backward()
            synchronize(device)
        # torch.cuda.synchronize(device=device)
        # start_time = timeit.default_timer()
        times = []
        with nvtx.range("measurement_forward+backward"):
            for i in range(n):
                synchronize(device)
                with nvtx.range("measurement_forward+backward_step_{}",i):
                    with nvtx.range("single_forward+backward"):
                        start_time = timeit.default_timer()
                        # model.forward(x)
                        optimizer.zero_grad()
                        with nvtx.range("single_forward"):
                            logits = model(x)
                        loss = cross_entropy(logits,y)
                        with nvtx.range("single_backward"):
                            loss.backward()
                        synchronize(device)
                        end_time = timeit.default_timer()
                        time = end_time - start_time
                    times.append(time)
            times = np.array(times)
            avg_time = times.mean()
            std_time = times.std()
        print(f"n={n} iterations | Avg: {avg_time:.4f} s | Std: {std_time:.4f} s")
        return avg_time, std_time, times
   
    # 前向 + 反向 + 优化器
    else:
        with nvtx.range("warmup_full_stage"):
            for t in range(w):
                with nvtx.range("warmup_step_{}",t):
                    # model.forward(x)
                    with nvtx.range("warmup_forward"):    
                        logits = model(x)
                    loss = cross_entropy(logits,y)
                    optimizer.zero_grad()
                    with nvtx.range("warmup_backward"):
                        loss.backward()
                    with nvtx.range("warmup_grad_clip"):
                        gradient_clipping(parameters=model.parameters(),max_l2_norm=grad_clip)
                    with nvtx.range("warmup_optimizer"):
                        optimizer.step()
            synchronize(device)
        # torch.cuda.synchronize(device=device)
        # start_time = timeit.default_timer()
        times = []
        with nvtx.range("measurement_full_stage"):
            for i in range(n):
                synchronize(device)
                with nvtx.range("measurement_step_{}",i):
                    start_time = timeit.default_timer()
                    # model.forward(x)
                    optimizer.zero_grad()
                    with nvtx.range("single_forward"):
                        logits = model(x)
                    loss = cross_entropy(logits,y)
                    with nvtx.range("single_backward"):
                        loss.backward()
                    with nvtx.range("single_grad_clip"):
                        gradient_clipping(parameters=model.parameters(),max_l2_norm=grad_clip)
                    with nvtx.range("optimizer_step"):
                        optimizer.step()
                    synchronize(device)
                    end_time = timeit.default_timer()
                    time = end_time - start_time
                    times.append(time)
            times = np.array(times)    
            avg_time = times.mean()
            std_time = times.std()
        print(f"n={n} iterations | Avg: {avg_time:.4f} s | Std: {std_time:.4f} s")
        print(f"NOTE: full training step timing includes gradient clipping (max_l2_norm={grad_clip}); "
              "options=2 minus options=1 is optimizer + clipping, not optimizer alone.")
        return avg_time, std_time, times
   



    

if __name__ == "__main__":
     args = parse_args()
     model_config = {**CONFIG, **MODEL_CONFIGS[args.model_size], "context_length": args.context_length}
     avg_time,std_time,times_list = benchmark(
         w=args.w,
         n=args.n,
         options=args.options,
         device=args.device,
         model_config=model_config,
         batch_size=args.batch_size,
     )
