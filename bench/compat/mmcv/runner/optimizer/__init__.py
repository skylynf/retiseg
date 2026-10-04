class DefaultOptimizerConstructor:
    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = paramwise_cfg
