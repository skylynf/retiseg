class DataContainer:
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only


def collate(*args, **kwargs):
    raise NotImplementedError


def scatter(*args, **kwargs):
    raise NotImplementedError


class MMDataParallel:
    pass


class MMDistributedDataParallel:
    pass
