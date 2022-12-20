import collections

import numpy as np
import torch


class SharedMemoryDict(collections.UserDict):
    ''' Transparently store numpy arrays as torch tensors
        which utilize shared memory.
    '''

    def __setitem__(self, key, item) -> None:
        if isinstance(item, dict):
            return super().__setitem__(key, SharedMemoryDict(item))
        elif isinstance(item, np.ndarray):
            tensor = torch.from_numpy(item)
            #tensor.share_memory_()
            return super().__setitem__(key, tensor)
        else:
            return super().__setitem__(key, item)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(item, torch.Tensor):
            return item.numpy()
        else:
            return item
