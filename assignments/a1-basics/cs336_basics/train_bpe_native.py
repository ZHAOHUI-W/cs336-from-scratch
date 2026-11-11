"""
train_bpe_native.py：最朴素的单进程 BPE 训练（未优化版本，用来和优化版对照）。

和 train_bpe.py 的区别：每一轮 merge 都重新扫描全部词条来统计 pair 频次，
再扫描全部词条做替换，复杂度约为「词条数 × merge 次数」，很慢但最直观。
其余流程（按 special 切分、GPT-2 预分词、vocab 初始化、并列取字典序最大的 pair、写 merges.txt）与 train_bpe.py 相同。
"""
#*- implementation for BPE (Byte Pair Encoding) training *#

# 实现 train_bpe(...)
# 输入: input_path, vocab_size, special_tokens
# 输出:
#   vocab:  dict[int, bytes]           # id → token 字节串
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
    """
    朴素版 BPE 训练。

    输入：
        input_path (str): 训练语料路径
        vocab_size (int): 目标词表大小
        special_tokens (list[str]): 特殊 token 列表
    输出：
        vocab: dict[int, bytes]；merges: list[tuple[bytes, bytes]]（按学习顺序）
    副作用：在当前工作目录写出 merges.txt。
    """
    # 1. 读文件成一个大字符串
    with open(input_path, "rb") as f:
        text = f.read()

    # 2. 用 special_tokens 切开（跨边界不 merge；special 本身不进 pair 统计）注意：special_tokens不进入pre-tokenize
    # 这里可以用正则切分，或者用 str.split
    # 这里假设 special_tokens 是 bytes 类型的列表
    # 将 special_tokens 转换为 bytes 类型
    # 注意: 实际传入的 special_tokens 是 str 列表，上一行才把它们编码成 bytes
    special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
    # 构建正则模式，匹配任意一个 special token 先长后短
    special_tokens_bytes = sorted(special_tokens_bytes, key=len, reverse=True)
    if len(special_tokens_bytes) == 0:
        segments = [text]
    else:
        special_tokens_pattern = b'|'.join(re.escape(token) for token in special_tokens_bytes)
    # 使用正则切分文本
        segments = re.split(special_tokens_pattern, text)  # re.split 会丢掉 special token 本身

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

    # 优化前：每轮：扫全部词条 → 建 pair_freq → max → 再扫全部词条改写
    # 所以复杂度大约是「词条规模 × merge 次数」。profile 里海量的 len / append 就是这两次全表扫描的痕迹。
    
    # 5. 初始化 vocab = specials + 256 个单 byte
    #    然后循环 merge，直到 len(vocab) == vocab_size:
    #       a. 统计（或维护）所有相邻 pair 的频率
    #       b. 选频率最高的 pair；并列取字典序更大
    #       c. merges.append(pair)
    #       d. 把词频表里所有「相邻这对」替换成合并后的新 bytes
    #       e. vocab 里加入这个新 token
    
    # 先放special 再加入256 byte 保证id连续
    vocab = {i: token.encode('utf-8') for i, token in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    
    merges = []
    while len(vocab) < vocab_size:
        # a.
        pair_freq = defaultdict(int)
        for token_tuple, count in freq.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                pair_freq[pair] += count

        # b.
        if not pair_freq:
            break
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]  # 频次最高；同频次取字典序更大的 pair

        # c.
        merges.append(best_pair)

        # d.
        new_token = best_pair[0] + best_pair[1]
        new_freq = defaultdict(int)
        for token_tuple, count in freq.items():
            new_token_tuple = []
            i = 0
            while i < len(token_tuple):
                if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == best_pair:
                    new_token_tuple.append(new_token)
                    i += 2
                else:
                    new_token_tuple.append(token_tuple[i])
                    i += 1
            new_freq[tuple(new_token_tuple)] += count  # 不同旧词合并后可能变成同一个新词，所以累加
        freq = new_freq
        # e.
        vocab[len(vocab)] = new_token

    
        
           
        

    # 6. return vocab, merges
    # 将输出的merges存成一个文件到当前目录下，文件名为merges.txt，每行是一个pair，格式为<token1> <token2>，其中<token1>和<token2>是bytes类型的字符串，需要decode成utf-8
    # 注意: 这里会在「当前工作目录」覆盖写出 merges.txt（副作用）
    with open("merges.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            b1, b2 = pair
            # CS336要求：bytes用latin-1解码，不要utf-8（很多单字节不是合法utf8）
            s1 = b1.decode("latin-1")
            s2 = b2.decode("latin-1")
            f.write(f"{s1} {s2}\n")
    return vocab, merges


































