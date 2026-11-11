import pickle, time, os
from cs336_basics.tokenizer import Tokenizer

with open("artifacts/tinystories_bpe/bpe.pkl", "rb") as f:
    d = pickle.load(f)
tok = Tokenizer(d["vocab"], d["merges"], special_tokens=["<|endoftext|>"])

path = "data/TinyStoriesV2-GPT4-valid.txt"
with open(path, encoding="utf-8") as f:
    text = "".join([next(f) for _ in range(1000)])

# 第一遍：填缓存（模拟真实编码前 1000 行，大部分 miss）
t0 = time.perf_counter()
ids1 = tok.encode(text)
t1 = time.perf_counter()

# 第二遍：缓存已热，测稳定吞吐
t2 = time.perf_counter()
ids2 = tok.encode(text)
t3 = time.perf_counter()

assert ids1 == ids2
npre = len(tok.pre_token_re.findall(text))
hits, misses = tok._cache_hits, tok._cache_misses
print("chars=%d pretokens=%d ids=%d" % (len(text), npre, len(ids1)))
print("first_pass=%.2fs  second_pass=%.2fs" % (t1 - t0, t3 - t2))
print("steady ids/s=%.0f pretokens/s=%.0f" % (len(ids2) / (t3 - t2), npre / (t3 - t2)))
print("hit_rate=%.2f%% cache_size=%d" % (100 * hits / max(hits + misses, 1), len(tok._cache)))

# 用实测 ids/字符 外推全量时间
valid_bytes = os.path.getsize(path)
train_bytes = os.path.getsize("data/TinyStoriesV2-GPT4-train.txt")
ids_per_char = len(ids1) / len(text)
rate = len(ids2) / (t3 - t2)
for name, nbytes in [("valid", valid_bytes), ("train", train_bytes)]:
    est_ids = nbytes * ids_per_char
    print("%s: bytes=%d est_ids=%.2fM est_time=%.1f min" % (name, nbytes, est_ids / 1e6, est_ids / rate / 60))