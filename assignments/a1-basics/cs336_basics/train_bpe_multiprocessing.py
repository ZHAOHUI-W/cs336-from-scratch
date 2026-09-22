#*- implementation for BPE (Byte Pair Encoding) training *#

# 实现 train_bpe(...)
# 输入: input_path, vocab_size, special_tokens
# 输出:
#   vocab:  dict[int, bytes]           # id → token bytes 比如 {257: b'<S>', 258: b'</S>'}
#   merges: list[tuple[bytes, bytes]]  # 按学习顺序
# 流程: # 1. 读文件成一个大字符串
        # 2. 用 special_tokens 切开（跨边界不 merge；special 本身不进 pair 统计）
        # 3. 对每个片段用 GPT-2 PAT 做 pre-tokenize（regex.finditer）PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 4. 每个 pre-token → UTF-8 bytes，做成
        #       词频表: dict[tuple[bytes, ...], int]
        # 5. 初始化 vocab = specials + 256 个单 byte
        #    然后循环 merge，直到 len(vocab) == vocab_size:
        #       a. 统计（或维护）所有相邻 pair 的频率
        #       b. 选频率最高的 pair；并列取字典序更大
        #       c. merges.append(pair)
        #       d. 把词频表里所有「相邻这对」替换成合并后的新 bytes
        #       e. vocab 里加入这个新 token
        # 6. return vocab, merges
import multiprocessing as mp
import os
import regex as re
from collections import defaultdict, Counter
from pathlib import Path
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT= r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def _count_chunk(args):
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start)  
    special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
    special_tokens_bytes = sorted(special_tokens_bytes, key=len, reverse=True)


    if special_tokens_bytes:
        special_tokens_pattern = b'|'.join(re.escape(token) for token in special_tokens_bytes)
        segments = re.split(special_tokens_pattern, chunk)
    else:
        segments = [chunk]
    counts = Counter()

    for segment in segments:
        
        text = segment.decode('utf-8', errors='strict')

        for match in re.finditer(PAT, text):
            pre_token_bytes = match.group(0).encode('utf-8')
            token_tuple = tuple(bytes([b]) for b in pre_token_bytes)
            counts[token_tuple] += 1

    return counts

def _build_frequency_table(
    input_path: str,
    special_tokens: list[str],
    num_workers: int,
) -> Counter:
    split_token = (
        special_tokens[0].encode("utf-8")
        if special_tokens
        else b"<|endoftext|>"
    )

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
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with mp.Pool(processes=num_workers) as pool:
        local_counters = pool.map(_count_chunk, tasks)

    frequency = Counter()

    for local_counter in local_counters:
        frequency.update(local_counter)

    return frequency

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_workers: int | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 4)

    freq = _build_frequency_table(input_path, special_tokens, num_workers)

    vocab = {
        index: token.encode("utf-8")
        for index, token in enumerate(special_tokens)
    }
    for i in range(256):
            vocab[len(vocab)] = bytes([i])
    
    merges = []
    # 维护一个索引：`pair_to_words: Dict[Tuple[bytes,bytes], Set[tuple[bytes,...]]]`
    # 含义：**key = pair，value = 所有包含这个 pair 的 token_tuple 集合**。> > 核心思想：> 每次选出 best_pair=(A,B) 之后，**不再遍历整个 freq 字典**，只遍历`pair_to_words[(A,B)]`里面存的那些 word；其余 word 完全跳过。
    pair_freq = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)    # ========== 初始化 pair_freq + pair_to_words 索引 ==========
    
    for word, cnt in freq.items():
        for j in range(len(word) - 1):
            p = (word[j], word[j+1])
            pair_freq[p] += cnt
            pair_to_words[p].add(word)

    # ========== BPE主循环 ==========
    while len(vocab) < vocab_size and pair_freq:
        
        # 选最优pair：频次降序，同频次取pair字典序更大
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
        A, B = best_pair
        merges.append(best_pair)
        C = A + B

       
        # 取出所有包含best_pair的word，这是本轮唯一需要处理的word
        affected_words = pair_to_words.pop(best_pair)
        # new_freq = dict(freq)  # 先把所有不受影响的word原样复制进new_freq

        # 先把所有不受影响的word原样复制进new_freq

        # 只处理受影响的word
        for old_word in list(affected_words):
        # ========== 1. 先从 freq 中移除旧词 ==========
            if old_word not in freq:
                continue  # 如果旧词已经被处理过，跳过
            cnt = freq.pop(old_word)

        # ========== 2. 彻底清理旧词在所有 pair 中的索引 ==========
            for j in range(len(old_word) - 1):
                p = (old_word[j], old_word[j+1])
                # 更新 pair 计数
                pair_freq[p] = pair_freq.get(p, 0) - cnt
                if pair_freq[p] <= 0:
                    del pair_freq[p]
                # 从反向索引中移除旧词
                if p in pair_to_words:
                    pair_to_words[p].discard(old_word)
                    if len(pair_to_words[p]) == 0:
                        del pair_to_words[p]

            # ========== 3. 生成合并后的新词 ==========
            new_word_list = []
            i = 0
            while i < len(old_word):
                if i < len(old_word)-1 and old_word[i]==A and old_word[i+1]==B:
                    new_word_list.append(C)
                    i += 2
                else:
                    new_word_list.append(old_word[i])
                    i += 1
            new_word = tuple(new_word_list)

            # ========== 4. 先把新词写入 freq ==========
            freq[new_word] = freq.get(new_word, 0) + cnt

            # ========== 5. 再把新词加入所有相关 pair 的索引 ==========
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j+1])
                pair_freq[p] = pair_freq.get(p, 0) + cnt
                pair_to_words[p].add(new_word)
        vocab[len(vocab)] = C
        
           
    
    return vocab, merges


































