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

import regex as re
from collections import defaultdict
def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
        
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 读文件成一个大字符串
    with open(input_path, "rb") as f:
        text = f.read()

    # 2. 用 special_tokens 切开（跨边界不 merge；special 本身不进 pair 统计）注意：special_tokens不进入pre-tokenize
    # 这里可以用正则切分，或者用 str.split
    # 这里假设 special_tokens 是 bytes 类型的列表
    # 将 special_tokens 转换为 bytes 类型
    special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
    # 构建正则模式，匹配任意一个 special token 先长后短
    special_tokens_bytes = sorted(special_tokens_bytes, key=len, reverse=True)
    if len(special_tokens_bytes) == 0:
        segments = [text]
    else:
        special_tokens_pattern = b'|'.join(re.escape(token) for token in special_tokens_bytes)
    # 使用正则切分文本
        segments = re.split(special_tokens_pattern, text)

    # 3. 对每个片段用 GPT-2 PAT 做 pre-tokenize（regex.finditer）PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    PAT= r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = []
    for segment in segments:
        pre_tokens.extend([m.group(0).encode('utf-8') for m in re.finditer(PAT, segment.decode('utf-8', errors='strict'))])
        

    # 4. 每个 pre-token → UTF-8 bytes，做成
    #       词频表: dict[tuple[bytes, ...], int]
    
    freq = defaultdict(int)
    for pt in pre_tokens:
        # 将bytes拆成单个字节 包装成tuple
        token_tuple = tuple(bytes([b]) for b in pt)
        freq[token_tuple] += 1

    # 优化后：维护 pair → count，merge (A,B)→C 时只改邻域上的计数，不要每轮从头数。
    vocab = {i: token.encode('utf-8') for i, token in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    
    merges = []
    # 维护一个索引：`pair_to_words: Dict[Tuple[bytes,bytes], Set[tuple[bytes,...]]]`
    # 含义：**key = pair，value = 所有包含这个 pair 的 token_tuple 集合**。> > 核心思想：> 每次选出 best_pair=(A,B) 之后，**不再遍历整个 freq 字典**，只遍历`pair_to_words[(A,B)]`里面存的那些 word；其余 word 完全跳过。
    pair_freq = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)    # ========== 初始化 pair_freq + pair_to_words 索引 ==========
    for word, cnt in freq.items():
        for j in range(len(word) - 1):
            p = (word[j], word[j+1])
            pair_freq[p] += cnt
            pair_to_words[p].add(word)

    # ========== BPE主循环 ==========
    while len(vocab) < vocab_size:
        if not pair_freq:
            break
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
        
           
        

    # 6. return vocab, merges
    # 将输出的merges存成一个文件到当前目录下，文件名为merges.txt，每行是一个pair，格式为<token1> <token2>，其中<token1>和<token2>是bytes类型的字符串，需要decode成utf-8
    with open("merges.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            b1, b2 = pair
            # CS336要求：bytes用latin-1解码，不要utf-8（很多单字节不是合法utf8）
            s1 = b1.decode("latin-1")
            s2 = b2.decode("latin-1")
            f.write(f"{s1} {s2}\n")
    return vocab, merges


































