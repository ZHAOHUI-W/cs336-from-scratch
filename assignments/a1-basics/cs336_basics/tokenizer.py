"""
tokenizer.py：字节级 BPE（Byte Pair Encoding，字节对编码）分词器 Tokenizer。

encode（编码）流程：
      1. 用 special token 正则把文本切成「特殊 token 段」和「普通文本段」（长的 special 优先匹配）
      2. 特殊 token 整段直接查表得到 id，不进入 BPE
      3. 普通文本用 GPT-2 预分词正则（pre-tokenization）切成词级片段
      4. 每个片段转成 UTF-8 字节，拆成单字节序列
      5. 按 merges 的训练顺序，逐条应用合并规则
      6. 合并后的每个字节片段查 bytes_to_id 得到 id
decode（解码）流程：
      id -> vocab 里的 bytes -> 全部拼接 -> 按 UTF-8 解码（非法字节用替换符 U+FFFD 兜底）
"""
# 先实现「无 merge、只按单字节查 vocab」的 encode/decode，冲 empty / single character
# 注意: 上面这行是最初版本的说明；现在的实现已经包含完整的 BPE merge 和 special token 处理。
from typing import Iterator, Dict, Optional, List, Tuple
import regex 
# GPT-2 官方 pre-tokenization 正则，与 train_bpe 完全一致
GPT2_PRETOKEN_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """
    字节级 BPE 分词器。

    属性：
        vocab: dict[int, bytes]，id -> 字节串
        bytes_to_id: dict[bytes, int]，字节串 -> id（vocab 的反向表）
        merges: list[tuple[bytes, bytes]]，按训练顺序排列的合并规则
        special_tokens: list[str]，特殊 token（例如 "<|endoftext|>"）
        special_pattern: 编译好的特殊 token 正则（没有特殊 token 时为 None）
        pre_token_re: 编译好的 GPT-2 预分词正则
    """
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        构造分词器。

        :param vocab: dict[int, bytes]，id -> bytes，decode 直接使用
        :param merges: list[tuple[bytes, bytes]]，按训练顺序排列的 BPE 合并规则，逐条按序应用
        :param special_tokens: 特殊 token 字符串列表（可为 None），长优先匹配，整段查表不进 BPE
        注意: special token 必须已经在 vocab 里，否则 encode 时 bytes_to_id 查表会 KeyError。
        """
        self.vocab = vocab  # id → bytes（decode 直接读）
        self.bytes_to_id: Dict[bytes, int] = {b: tid for tid, b in vocab.items()}  # bytes → id（encode 查表）
        self.merges = merges  # 严格按列表顺序应用，不做任何重排序
        self.special_tokens: List[str] = special_tokens or []
        
        self.use_cache:bool=True
        self._cache: Dict[str, Tuple[int, ...]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        # 1. 编译 special token 匹配正则：长 token 优先，避免短串抢占
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [regex.escape(s) for s in sorted_specials]
            self.special_pattern = regex.compile("|".join(escaped))
        else:
            self.special_pattern = None

        # 2. 编译 pre-tokenization 正则（GPT-2 官方规则）
        self.pre_token_re = regex.compile(GPT2_PRETOKEN_PAT)

    def encode(self, text: str) -> List[int]:
        """
        字符串 -> token id 列表。

        输入：text (str)
        输出：list[int]，空字符串返回 []
        """
        if not text:
            return []

        ids: List[int] = []
        if self.special_pattern is None:
            self._encode_normal_text(text, ids)
            return ids

        # 交替处理：特殊 token 段 + 普通文本段
        last_pos = 0
        for match in self.special_pattern.finditer(text):
            start, end = match.span()
            # 先处理两段 special 之间的普通文本
            if start > last_pos:
                self._encode_normal_text(text[last_pos:start], ids)
            # special 整段直接查表，不进 BPE
            special_bytes = match.group().encode("utf-8")
            ids.append(self.bytes_to_id[special_bytes])
            last_pos = end

        # 处理末尾剩余普通文本
        if last_pos < len(text):
            self._encode_normal_text(text[last_pos:], ids)

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        token id 列表 -> 字符串。

        输入：ids (list[int])
        输出：str；先把各 id 对应的 bytes 拼起来再整体解码，
              所以一个多字节字符被拆在多个 token 里也能正确还原；坏字节用替换符兜底。
        """
        byte_buffer = b"".join(self.vocab[tid] for tid in ids)  # 拼接所有 token 的字节
        return byte_buffer.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable) -> Iterator[int]:
        """
        流式编码：对可迭代的字符串块（如逐行读文件）逐块 encode，逐个产出 token id，节省内存。

        输入：iterable，产出 str 的可迭代对象（如文件句柄）
        输出：Iterator[int]
        注意: 每个块独立 encode，块与块之间不会合并（按行读文件时每行末尾带换行，一般没问题）。
        """
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("iterable must yield str chunks")
            for tid in self.encode(chunk):
                yield tid

    # ------------------------------ 内部辅助 ------------------------------
    def _encode_normal_text(self, text: str, ids: List[int]) -> None:
        """
        普通文本（不含 special token）：pre-tokenize → 逐词 BPE 合并 → 查表追加 id。

        输入：text (str)；ids (list[int])，结果直接 append 到这个列表里（原地修改）
        输出：None
        注意: 对每个 pre-token 都完整遍历一遍全部 merges（复杂度约为 片段数 × merges 数），结果正确但较慢，也没有缓存。
        """
        # 1. 按 GPT-2 正则切分为 pre-token（词级边界）
        pre_tokens = self.pre_token_re.findall(text)

        for token in pre_tokens:
            # 1.先查缓存，命中就直接拼进结果 跳过bpe合并
            if self.use_cache:
                cached = self._cache.get(token)
                if cached is not None:
                    self._cache_hits += 1
                    ids.extend(cached)
                    continue
            # 2. 转 UTF-8 字节，拆为单字节序列（BPE 初始状态）
                self._cache_misses += 1
            token_bytes = token.encode("utf-8")
            byte_seq: List[bytes] = [bytes([b]) for b in token_bytes]  # 例如 b"hi" -> [b"h", b"i"]

            # 3. 按 merges 顺序逐条应用合并规则
            for merge_pair in self.merges:
                byte_seq = self._apply_single_merge(byte_seq, merge_pair)

            # 4. 合并完成后，每个片段查表转 id
            token_ids = tuple(self.bytes_to_id[b] for b in byte_seq)

            # 3. 结果存成不可变 tuple 再缓存，避免共享可变对象被误改
            if self.use_cache:
                self._cache[token] = token_ids
            ids.extend(token_ids)

    @staticmethod
    def _apply_single_merge(byte_seq: List[bytes], merge_pair: Tuple[bytes, bytes]) -> List[bytes]:
        """
        应用单条 merge 规则：左到右扫描，匹配则合并，不重叠；一条规则可在当前序列中多次生效。

        输入：byte_seq (list[bytes])，当前片段序列；merge_pair (a, b)
        输出：新的 list[bytes]，所有相邻的 (a, b) 被替换为 a+b
        例：[b"a", b"a", b"a"] 应用 (b"a", b"a") -> [b"aa", b"a"]
        """
        a, b = merge_pair
        merged = []
        i = 0
        n = len(byte_seq)
        while i < n:
            if i < n - 1 and byte_seq[i] == a and byte_seq[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(byte_seq[i])
                i += 1
        return merged
        
        
   