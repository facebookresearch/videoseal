
import threading
from collections import OrderedDict


class LRUDict(OrderedDict):
    def __init__(self, maxsize=10):
        super().__init__()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            if key in self:
                del self[key]
            elif len(self) >= self.maxsize:
                self.popitem(last=False)
            super().__setitem__(key, value)

    def __getitem__(self, key):
        with self.lock:
            return super().__getitem__(key)

    def __delitem__(self, key):
        with self.lock:
            return super().__delitem__(key)
