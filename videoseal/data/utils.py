
import threading
from collections import OrderedDict


class LRUDict(OrderedDict):
    def __init__(self, maxsize=10):
        super().__init__()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            super().__setitem__(key, value)

        if len(self) >= self.maxsize:
            # Clear 10% of the max size
            num_to_clear = int(self.maxsize * 0.1)
            keys_to_remove = list(self.keys())[:num_to_clear]
            for key in keys_to_remove:
                del self[key]

    def __getitem__(self, key):
        with self.lock:
            return super().__getitem__(key)

    def __delitem__(self, key):
        with self.lock:
            return super().__delitem__(key)
