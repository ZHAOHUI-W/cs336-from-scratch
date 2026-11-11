"""cs336_basics 包入口：只负责读取已安装包的版本号，没有其他逻辑。"""
import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
