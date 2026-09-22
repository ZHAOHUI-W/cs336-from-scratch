import threading
import time
import psutil


class MemoryMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.process = psutil.Process()
        self.peak_total = 0
        self.peak_by_process = {}
        self.running = False
        self.thread = None

    def _sample(self):
        processes = [self.process]

        try:
            processes.extend(self.process.children(recursive=True))
        except psutil.Error:
            pass

        total_rss = 0

        for process in processes:
            try:
                rss = process.memory_info().rss
                total_rss += rss

                self.peak_by_process[process.pid] = max(
                    self.peak_by_process.get(process.pid, 0),
                    rss,
                )
            except psutil.Error:
                # 子进程可能刚好退出
                continue

        self.peak_total = max(self.peak_total, total_rss)

    def _run(self):
        while self.running:
            self._sample()
            time.sleep(self.interval)

        self._sample()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    @property
    def peak_total_mb(self):
        return self.peak_total / 1024 / 1024