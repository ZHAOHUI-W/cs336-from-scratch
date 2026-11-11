"""
train_bpe_multiprocessing_buckets.py：多进程预分词 + 「频次分桶」选最优 pair 的 BPE 训练（目前最快的版本）。

和 train_bpe_multiprocessing.py 的区别：
      不再每轮对整个 pair_freq 做 max（O(pair 数)），而是用 PairFrequencyBuckets 按频次分桶：
      buckets[频次] = 该频次的 pair 集合，并记住当前最大频次 max_count，
      选最优 pair 时只需在最大频次那一桶里取字典序最大的 pair。
流程：切块 → 子进程预分词计数 → 合并词频 → 初始化 pair 计数与反向索引 → merge 主循环。
"""
import multiprocessing as mp
import os
from collections import Counter, defaultdict
import cProfile
import regex as re
import pickle

from cs336_basics.pretokenization_example import find_chunk_boundaries


PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class PairFrequencyBuckets:
    """
    维护 pair 频率，并支持 O(1) 平均复杂度更新。
    （中文补充）pair 频次表 + 按频次分桶的索引。

    属性：
        counts: Counter，pair -> 当前频次
        buckets: dict[int, set[pair]]，频次 -> 该频次的所有 pair
        max_count: int，当前可能的最大频次（只增不主动减，best_pair 时再向下找非空桶）
    """

    def __init__(self) -> None:
        self.counts: Counter[tuple[bytes, bytes]] = Counter()
        self.buckets: dict[
            int,
            set[tuple[bytes, bytes]],
        ] = defaultdict(set)
        self.max_count = 0

    def update(
        self,
        pair: tuple[bytes, bytes],
        delta: int,
    ) -> None:
        """
        把 pair 的频次加上 delta（可正可负），同时把它从旧频次桶挪到新频次桶。
        新频次 ≤ 0 时直接从 counts 中删除。
        """
        old_count = self.counts.get(pair, 0)
        new_count = old_count + delta

        if old_count > 0:
            old_bucket = self.buckets[old_count]
            old_bucket.discard(pair)

            if not old_bucket:
                del self.buckets[old_count]

        if new_count > 0:
            self.counts[pair] = new_count
            self.buckets[new_count].add(pair)

            if new_count > self.max_count:
                self.max_count = new_count
        else:
            self.counts.pop(pair, None)

    def best_pair(self) -> tuple[bytes, bytes] | None:
        """
        返回频次最高的 pair；同频次时取字典序最大的 pair。没有 pair 时返回 None。
        从 max_count 开始向下找第一个非空桶。
        """
        while self.max_count > 0:
            bucket = self.buckets.get(self.max_count)

            if bucket:
                # 同频率时选择字典序最大的 pair
                return max(bucket)

            self.max_count -= 1  # 这一桶空了，最大频次下降 1 继续找

        return None

    def __bool__(self) -> bool:
        """让 `while ... and pair_index:` 这种写法可用：还有 pair 时为 True。"""
        return bool(self.counts)
    
def _profiled_count_chunk(args):
    """
    带性能分析（cProfile）的子进程包装：调用 _count_chunk，并把该进程的分析结果写到 worker_<pid>.prof。
    注意: 每次训练都会在当前工作目录为每个子进程生成一个 worker_<pid>.prof 文件（副作用）。
    """
    profiler = cProfile.Profile()
    profiler.enable()

    result = _count_chunk(args)

    profiler.disable()
    profiler.dump_stats(
        f"worker_{os.getpid()}.prof"
    )

    return result

def _count_chunk(
    args: tuple[
        str,
        int,
        int,
        list[str],
    ],
) -> Counter[tuple[bytes, ...]]:
    """
    子进程任务：统计文件 [start, end) 字节区间内的 pre-token 词频。

    输入：args = (input_path, start, end, special_tokens)
    输出：Counter[tuple[bytes, ...]]，单字节元组 -> 出现次数
    """
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start)

    special_tokens_bytes = sorted(
        (
            token.encode("utf-8")
            for token in special_tokens
        ),
        key=len,
        reverse=True,
    )

    if special_tokens_bytes:
        special_pattern = b"|".join(
            re.escape(token)
            for token in special_tokens_bytes
        )
        segments = re.split(special_pattern, chunk)  # 按 special 切开，special 本身被丢弃
    else:
        segments = [chunk]

    counts: Counter[tuple[bytes, ...]] = Counter()

    for segment in segments:
        text = segment.decode("utf-8", errors="strict")

        for match in re.finditer(PAT, text):
            token_bytes = match.group(0).encode("utf-8")
            token = tuple(bytes([value]) for value in token_bytes)
            counts[token] += 1

    return counts


def _build_frequency_table(
    input_path: str,
    special_tokens: list[str],
    num_workers: int,
) -> Counter[tuple[bytes, ...]]:
    """
    并行构建全文件词频表。
    输入：input_path、special_tokens、num_workers
    输出：Counter[tuple[bytes, ...]]
    注意: 切块只用 special_tokens[0] 作为切点；没有 special token 时用 b"<|endoftext|>" 兜底。
    """
    if special_tokens:
        split_token = special_tokens[0].encode("utf-8")
    else:
        split_token = b"<|endoftext|>"

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            num_workers,
            split_token,
        )

    tasks = [
        (
            input_path,
            start,
            end,
            special_tokens,
        )
        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
        )
    ]

    with mp.Pool(processes=num_workers) as pool:
        local_counters = pool.map(_profiled_count_chunk, tasks)  # 每块一个进程（带 profile 包装）

    frequency: Counter[tuple[bytes, ...]] = Counter()

    for local_counter in local_counters:
        frequency.update(local_counter)

    return frequency


def _merge_word(
    word: tuple[bytes, ...],
    left: bytes,
    right: bytes,
    merged: bytes,
) -> tuple[bytes, ...]:
    """
    把词 word 里所有相邻的 (left, right) 从左到右、不重叠地替换为 merged。

    输入：word (tuple[bytes, ...])；left、right (bytes)；merged = left + right
    输出：新的 tuple[bytes, ...]
    例：(b"a", b"b", b"c") 合并 (b"a", b"b") -> (b"ab", b"c")
    """
    result: list[bytes] = []
    index = 0

    while index < len(word):
        if (
            index + 1 < len(word)
            and word[index] == left
            and word[index + 1] == right
        ):
            result.append(merged)
            index += 2
        else:
            result.append(word[index])
            index += 1

    return tuple(result)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """
    分桶版 BPE 训练。

    输入：
        input_path (str): 训练语料路径
        vocab_size (int): 目标词表大小
        special_tokens (list[str]): 特殊 token 列表
        num_workers (int | None): 进程数，默认 min(CPU 核数, 4)
    输出：
        vocab: dict[int, bytes]；merges: list[tuple[bytes, bytes]]（按学习顺序）
    """
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 4)

    frequency = _build_frequency_table(
        input_path,
        special_tokens,
        num_workers,
    )

    vocab: dict[int, bytes] = {
        index: token.encode("utf-8")
        for index, token in enumerate(special_tokens)
    }

    for byte_value in range(256):
        vocab[len(vocab)] = bytes([byte_value])

    merges: list[tuple[bytes, bytes]] = []

    pair_index = PairFrequencyBuckets()
    pair_to_words: dict[
        tuple[bytes, bytes],
        set[tuple[bytes, ...]],
    ] = defaultdict(set)

    # 初始化 pair 频率和反向索引
    for word, count in frequency.items():
        for index in range(len(word) - 1):
            pair = (word[index], word[index + 1])

            pair_index.update(pair, count)
            pair_to_words[pair].add(word)

    # BPE merge 主循环
    while len(vocab) < vocab_size and pair_index:
        best_pair = pair_index.best_pair()  # 从最高频次桶中取字典序最大的 pair

        if best_pair is None:
            break

        left, right = best_pair
        merged = left + right
        merges.append(best_pair)

        affected_words = pair_to_words.pop(best_pair, set())  # 只处理包含 best_pair 的词

        for old_word in list(affected_words):
            if old_word not in frequency:
                continue

            count = frequency.pop(old_word)

            # 删除旧 word 对所有 pair 的贡献
            for index in range(len(old_word) - 1):
                pair = (old_word[index], old_word[index + 1])
                pair_index.update(pair, -count)

                words = pair_to_words.get(pair)

                if words is not None:
                    words.discard(old_word)

                    if not words:
                        del pair_to_words[pair]

            new_word = _merge_word(
                old_word,
                left,
                right,
                merged,
            )

            # 合并相同的新 word 的频率
            frequency[new_word] += count

            # 添加新 word 对所有 pair 的贡献
            for index in range(len(new_word) - 1):
                pair = (new_word[index], new_word[index + 1])

                pair_index.update(pair, count)
                pair_to_words[pair].add(new_word)

        vocab[len(vocab)] = merged  # 新 token 的 id 依次递增

    return vocab, merges

if __name__ == "__main__":
    path = r"E:\work\code\grok\cs336-from-scratch\assignments\a1-basics\data\TinyStoriesV2-GPT4-train.txt"

    vocab, merges = train_bpe(path, 10000, ["<|endoftext|>"])
    with open("artifacts/tinystories_bpe/bpe.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
