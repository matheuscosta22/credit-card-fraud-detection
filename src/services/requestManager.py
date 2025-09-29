import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil


# isso deve ser gerenciado de acordo com  da maquina que esta sendo usada
MAX_CONCURRENT_REQUESTS = 3
MAX_WORKERS = 3


class RequestManager:
    _instance = None
    _initialized = False
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self):
        if not RequestManager._initialized:
            self.threadPool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
            self.maxConcurrentRequests = MAX_CONCURRENT_REQUESTS
            self.currentRequests = 0
            RequestManager._initialized = True


    async def acquire_slot(self) -> bool:
        async with self._lock:
            if self.currentRequests < self.maxConcurrentRequests:
                self.currentRequests += 1
                return True

            return False


    async def release_slot(self):
        async with self._lock:
            if self.currentRequests > 0:
                self.currentRequests -= 1


    def too_many_processes(self) -> bool:
        lowMemory = psutil.virtual_memory().available <= 2 * 1024**3

        return self.threadPool._work_queue.qsize() >= 10 or lowMemory