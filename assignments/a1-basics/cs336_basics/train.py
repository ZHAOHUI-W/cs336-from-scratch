import torch
import os
import numpy as np
import argparse
from cs336_basics import transformer_lm
from cs336_basics import adamw
from cs336_basics import model_checkpoint
from cs336_basics import lr_schedule
from cs336_basics import load_dataset
from cs336_basics import cross_entropy
from cs336_basics import grad_clip

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",type=str,default="data/tinystories_train.bin")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--grad_clip",type=float,default=1.0)
    parser.add_argument("--valid_path",type=str,default="data/tinystories_valid.bin")
    parser.add_argument("--vocab_size",type=int,default=10000)
    parser.add_argument("--context_length",type=int,default=256)
    parser.add_argument("--d_model",type=int,default=512)
    parser.add_argument("--num_layers",type=int,default=4)
    parser.add_argument("--num_heads",type=int,default=16)
    parser.add_argument("--d_ff",type=int,default=1344)
    parser.add_argument("--rope_theta",type=float,default=10000)
    parser.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_iters",type=int,default=40000)
    parser.add_argument("--checkpoint_path",type=str,default="checkpoint_path/ckpt.pt")
    parser.add_argument("--log_interval",type=int,default=100)
    parser.add_argument("--save_interval",type=int,default=2000)
    parser.add_argument("--valid_interval",type=int,default=1000)
    parser.add_argument("--eval_batches",type=int,default=20)
    parser.add_argument("--lr_max",type=float,default=1e-3)
    parser.add_argument("--lr_min",type=float,default=1e-4)
    parser.add_argument("--t_w",type=int,default=1000)
    parser.add_argument("--t_c",type=int,default=40000)

    


    args = parser.parse_args()

    train_dataset = np.memmap(args.train_path,mode="r",dtype=np.uint16)
    valid_dataset = np.memmap(args.valid_path,mode="r",dtype=np.uint16)

    model = transformer_lm.TransformerLM(vocab_size=args.vocab_size,context_length=args.context_length,d_model=args.d_model,num_layers=args.num_layers,num_heads=args.num_heads,
                                         d_ff=args.d_ff,rope_theta=args.rope_theta,device=args.device).to(args.device)
    optimizer = adamw.AdamW(params=model.parameters(),lr=args.lr_max)

    checkpoint_path = args.checkpoint_path
    if os.path.exists(checkpoint_path):
        iteration = model_checkpoint.load_checkpoint(src=args.checkpoint_path,model=model,optimizer=optimizer)
        start_it = iteration
    else: start_it = 0
    
    for it in range(start_it,args.max_iters):
        lr = lr_schedule.lr_schedule(t=it,t_w=args.t_w,t_c=args.t_c,lr_min=args.lr_min,lr_max=args.lr_max)

        for g in optimizer.param_groups:
            g["lr"] = lr
        x,y = load_dataset.data_loading(dataset=train_dataset,batch_size=args.batch_size,context_length=args.context_length,device=args.device)

        logits = model(x)
        

        loss = cross_entropy.cross_entropy(logits,y)


        optimizer.zero_grad()

        loss.backward()

        grad_clip.gradient_clipping(parameters=model.parameters(),max_l2_norm=args.grad_clip)

        optimizer.step()
        if it % args.log_interval == 0:
            print(f"it: {it}  |  loss: {loss.item():.4f}  | lr:{lr:.2e}")

        if it % args.valid_interval == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(args.eval_batches):
                    vx,vy = load_dataset.data_loading(dataset=valid_dataset,batch_size=args.batch_size,context_length=args.context_length,device=args.device)
                    losses.append(cross_entropy.cross_entropy(model(vx),vy).item())
                val_loss = sum(losses) / len(losses)
            print(f"it: {it} |  val_loss: {val_loss:.4f}")
            model.train()
        if it % args.save_interval == 0:
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            model_checkpoint.save_checkpoint(model=model,optimizer=optimizer,iteration=it+1,out=args.checkpoint_path)


if __name__ == "__main__": 
    train()



