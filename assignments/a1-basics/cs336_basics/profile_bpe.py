"""
profile_bpe.py：用 cProfile（Python 自带的性能分析器）分析分桶版 train_bpe 在 TinyStories 上的耗时。

运行后：结果保存到当前目录的 profile_bpe_buckets.prof，并按累计耗时（cumtime）打印前 50 个函数。
"""
# from pathlib import Path
# from tests.adapters import run_train_bpe

# if __name__ == "__main__":
#     run_train_bpe(
#         Path("data/TinyStoriesV2-GPT4-train.txt"),
#         10000,
#         ["<|endoftext|>"],
#         num_workers=4
#     )



import cProfile
import pstats
from pathlib import Path

from cs336_basics.train_bpe_multiprocessing_buckets import train_bpe


def main():
    """在 TinyStories 训练集上训练 vocab_size=10000 的 BPE（4 个进程）。"""
    train_bpe(
        input_path=Path("data/TinyStoriesV2-GPT4-train.txt"),
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_workers=4,
    )


if __name__ == "__main__":
    profiler = cProfile.Profile()

    profiler.enable()
    main()
    profiler.disable()

    profiler.dump_stats("profile_bpe_buckets.prof")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(50)