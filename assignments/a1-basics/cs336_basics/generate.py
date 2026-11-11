import torch
import pickle
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.model_checkpoint import load_checkpoint
from cs336_basics.adamw import AdamW

CONFIG = dict(
    vocab_size = 10000,
    context_length = 256,
    d_model = 512,
    num_layers = 4,
    num_heads = 16,
    d_ff = 1344,
    rope_theta = 10000,
)


def load_tokenizer(path="artifacts/tinystories_bpe/bpe.pkl"):
    with open(path,"rb") as f:
        d = pickle.load(f)
    return Tokenizer(d["vocab"],d["merges"],special_tokens=["<|endoftext|>"])


def load_model(path="checkpoint_path/ckpt.pt",device="cuda"):
    model = TransformerLM(**CONFIG,device=device).to(device)
    optimizer = AdamW(model.parameters(),lr=1e-3)
    iteration = load_checkpoint(path,model,optimizer)

    model.eval()
    print("loaded iteration:", iteration)

    return model





@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p=None,device=None):

    ids = tokenizer.encode(prompt)

    if not ids:
        raise ValueError("prompt 编码后为空")

    for _ in range(max_new_tokens):
        window = ids[-CONFIG["context_length"]:]
        x = torch.tensor(window,dtype=torch.long,device=device).unsqueeze(0)

        logits = model(x)[0,-1,:]

        if temperature <=0:
            next_id = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_p and top_p > 0:
                sorted_probs,indices = torch.sort(torch.softmax(logits,dim=-1),descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = (cumsum - sorted_probs) < top_p
                sorted_probs = sorted_probs * mask
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_id = indices[torch.multinomial(sorted_probs,1)]
                next_id = int(next_id)
            else:
                next_id = int(torch.multinomial(torch.softmax(logits,dim=-1),1))

        if int(next_id) == 0:
            break

        ids.append(next_id)

    return tokenizer.decode(ids)







if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    model = load_model(device=device)

    for p in ["A flying bird was suddenly caught by an owl ", "The little girl was", "<|endoftext|>"]:
        print("=" * 70)
        print("[greedy]", repr(p))
        print(generate(model, tok, p, max_new_tokens=120, temperature=0.0, device=device))
        print("[t=0.8, top_p=0.9]", repr(p))
        print(generate(model, tok, p, max_new_tokens=120, temperature=0.8, top_p=0.9, device=device))






    


