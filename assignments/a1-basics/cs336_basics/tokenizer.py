# 先实现「无 merge、只按单字节查 vocab」的 encode/decode，冲 empty / single character
from typing import Iterator, Dict, Optional, List, Tuple
import regex 
# GPT-2 官方 pre-tokenization 正则，与 train_bpe 完全一致
GPT2_PRETOKEN_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        :param vocab: id -> bytes，decode 直接使用
        :param merges: 按训练顺序排列的 BPE 合并规则，逐条按序应用
        :param special_tokens: 特殊 token 字符串列表，长优先匹配，整段查表不进 BPE
        """
        self.vocab = vocab  # id → bytes（decode 直接读）
        self.bytes_to_id: Dict[bytes, int] = {b: tid for tid, b in vocab.items()}  # bytes → id（encode 查表）
        self.merges = merges  # 严格按列表顺序应用，不做任何重排序
        self.special_tokens: List[str] = special_tokens or []

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
        """字符串 → token id 列表"""
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
        """token id 列表 → 字符串，坏字节用替换符兜底"""
        byte_buffer = b"".join(self.vocab[tid] for tid in ids)
        return byte_buffer.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable) -> Iterator[int]:
        """流式迭代输入，逐个产出 token id"""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("iterable must yield str chunks")
            for tid in self.encode(chunk):
                yield tid

    # ------------------------------ 内部辅助 ------------------------------
    def _encode_normal_text(self, text: str, ids: List[int]) -> None:
        """普通文本：pre-tokenize → 逐词 BPE 合并 → 查表追加 id"""
        # 1. 按 GPT-2 正则切分为 pre-token（词级边界）
        pre_tokens = self.pre_token_re.findall(text)

        for token in pre_tokens:
            # 2. 转 UTF-8 字节，拆为单字节序列（BPE 初始状态）
            token_bytes = token.encode("utf-8")
            byte_seq: List[bytes] = [bytes([b]) for b in token_bytes]

            # 3. 按 merges 顺序逐条应用合并规则
            for merge_pair in self.merges:
                byte_seq = self._apply_single_merge(byte_seq, merge_pair)

            # 4. 合并完成后，每个片段查表转 id
            for b in byte_seq:
                ids.append(self.bytes_to_id[b])

    @staticmethod
    def _apply_single_merge(byte_seq: List[bytes], merge_pair: Tuple[bytes, bytes]) -> List[bytes]:
        """
        应用单条 merge 规则：左到右扫描，匹配则合并，不重叠
        一条规则可在当前序列中多次生效
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
        
        
   