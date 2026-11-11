# 1. **加载你训练好的 TinyStories BPE**，也就是 vocab 和 merges，再加上特殊 token `<|endoftext|>`，构造出 `Tokenizer`。
# 2. **流式编码。** 打开 txt，用 `tokenizer.encode_iterable(f)` 一边读一边产出 token id，不要一次性把整个文件读进内存。训练集有 2GB 左右，一次读进来会很慢，也很占内存。
# 3. **存成 uint16。** 把 id 收集起来转成 `np.array(ids, dtype=np.uint16)`，再 `.tofile("data/tinystories_train.bin")`。之所以能用 uint16，是因为 vocab 是 10000，小于 65536。
# 4. **验证集也照做一遍**，存成 `tinystories_valid.bin`。

# 存完后检查两件事：一是 `np.memmap` 读回来的长度，二是解码前几十个 token，看还原出来的文字对不对。


import os
import pickle, numpy as np
from cs336_basics.tokenizer import Tokenizer

def encode_file(tokenizer, in_path, out_path, chunk=1_000_000):
    with open(in_path, encoding="utf-8") as fin, open(out_path, "wb") as fout:
        buf = []
        total = 0
        for tid in tokenizer.encode_iterable(fin):                  # 流式产出 id
            buf.append(tid)
            total += 1
            if len(buf) >= chunk:
                buf_list = np.array(buf,dtype=np.uint16)
                fout.write(buf_list.tobytes())
                buf = []
        buf_list = np.array(buf,dtype=np.uint16)                                # 攒满 chunk 个就写一次 uint16
        fout.write(buf_list.tobytes())                              # 写剩下的
    return total
if __name__ == "__main__":
    with open("artifacts/tinystories_bpe/bpe.pkl", "rb") as f:
        d = pickle.load(f)
    tok = Tokenizer(d["vocab"], d["merges"], special_tokens=["<|endoftext|>"])
    total = encode_file(tok, "data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_train.bin")


    print("cache_size=%d  hits=%d  misses=%d  hit_rate=%.2f%%"
      % (len(tok._cache), tok._cache_hits, tok._cache_misses,
         100.0 * tok._cache_hits / max(total, 1)))
    print(f"total: {total} ")
    assert os.path.getsize("data/tinystories_train.bin") == total * 2
    assert np.memmap("data/tinystories_train.bin", dtype=np.uint16, mode="r").shape[0] == total
    # 验证集跑通、检查无误后，再跑训练集