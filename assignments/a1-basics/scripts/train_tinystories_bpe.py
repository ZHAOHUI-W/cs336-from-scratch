#记 wall time + 内存（psutil / Windows 下也可用），训完写出 vocab/merges，并打印 max(vocab.values(), key=len)。
#先在 valid（22 MB）上试跑通序列化与计时，再上 train


#调用你已有的 train_bpe（或薄封装脚本）
# 把 vocab、merges 序列化到磁盘（例如 pickle / 自定义格式，路径自定

from cs336_basics.train_bpe_multiprocessing_buckets import train_bpe
import multiprocessing
import time
import psutil
import os
from memory_monitor import MemoryMonitor

if __name__ == "__main__":
    # 创建输出目录
    output_dir = "artifacts/tinystories_bpe"
    os.makedirs(output_dir, exist_ok=True)


    monitor = MemoryMonitor(interval=0.2)
    monitor.start()


    # 计时，+ 内存采样
    start_time = time.time()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024 * 1024)  # MB
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_workers=4,
    )
    monitor.stop()
    end_time = time.time() - start_time
    mem_after = proc.memory_info().rss / (1024 * 1024)
    mem_used = mem_after - mem_before
    print(f"Training time: {end_time:.2f} seconds")
    # print(f"Memory used: {mem_used:.2f} MB")
    # 打印最长token bytes
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token bytes: {longest_token} (length: {len(longest_token)})")

    info = proc.memory_info()
    peak_mb = info.peak_wset / (1024 * 1024)  # Windows
    
    # Linux 可看 info.rss 采样，或 resource.getrusage
    # print(f"Peak WSET: {peak_mb:.2f} MB")

    print(f"Peak total memory (including children): {monitor.peak_total_mb:.2f} MB")
    


    # 将输出的merges存成一个文件到当前目录下，文件名为merges.txt，每行是一个pair，格式为<token1> <token2>，其中<token1>和<token2>是bytes类型的字符串，需要decode成utf-8
    with open(os.path.join(output_dir, "merges_train_process4_buckets.txt"), "w", encoding="utf-8") as f:
        for pair in merges:
            b1, b2 = pair
            # CS336要求：bytes用latin-1解码，不要utf-8（很多单字节不是合法utf8）
            s1 = b1.decode("latin-1")
            s2 = b2.decode("latin-1")
            f.write(f"{s1} {s2}\n")

    # 将输出的vocab存成一个文件到当前目录下，文件名为vocab.txt，每行是一个token，格式为<token> <count>，其中<token>是bytes类型的字符串，需要decode成utf-8
    with open(os.path.join(output_dir, "vocab_train_process4_buckets.txt"), "w", encoding="utf-8") as f:
        for tid, token_bytes in vocab.items():
            s = token_bytes.decode("latin-1")
            f.write(f"{tid}\t{s}\n")  # 或 json/pickle；关键是保存 id↔bytes