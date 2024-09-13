
from collections import OrderedDict


class LRUDict(OrderedDict):
    def __init__(self, maxsize=10):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        elif len(self) >= self.maxsize:
            self.popitem(last=False)
        super().__setitem__(key, value)
