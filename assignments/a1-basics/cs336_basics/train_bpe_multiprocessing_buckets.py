import multiprocessing as mp
import os
from collections import Counter, defaultdict
import cProfile
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries


PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class PairFrequencyBuckets:
    """维护 pair 频率，并支持 O(1) 平均复杂度更新。"""

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
        while self.max_count > 0:
            bucket = self.buckets.get(self.max_count)

            if bucket:
                # 同频率时选择字典序最大的 pair
                return max(bucket)

            self.max_count -= 1

        return None

    def __bool__(self) -> bool:
        return bool(self.counts)
    
def _profiled_count_chunk(args):
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
        segments = re.split(special_pattern, chunk)
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
        local_counters = pool.map(_profiled_count_chunk, tasks)

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
        best_pair = pair_index.best_pair()

        if best_pair is None:
            break

        left, right = best_pair
        merged = left + right
        merges.append(best_pair)

        affected_words = pair_to_words.pop(best_pair, set())

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

        vocab[len(vocab)] = merged

    return vocab, merges


