"""
pretokenization_example.py：课程提供的示例代码，把大文件切成可以独立并行预分词的块。

思路：先按文件大小均匀猜测切点，再从每个切点往后找最近的 special token（如 b"<|endoftext|>"），
      把切点挪到 special token 开头，保证一个文档不会被切成两半，也不会切断 UTF-8 多字节字符。
"""
import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    中文说明：
    输入：
        file (BinaryIO): 以 "rb" 打开的文件对象
        desired_num_chunks (int): 期望的块数（通常等于进程数）
        split_special_token (bytes): 用作切点的特殊 token 字节串
    输出：list[int]，排好序、去重后的切点字节偏移，首个为 0、最后一个为文件大小；
          相邻两个切点 [start, end) 就是一块。如果多个切点挪到同一位置，块数会少于期望值。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)  # 在 4KB 小块中找 special token
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at  # 切点挪到 special token 开头
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

if __name__ == "__main__":

## Usage
    with open(..., "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
