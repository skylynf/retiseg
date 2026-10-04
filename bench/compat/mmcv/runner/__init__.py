class Hook:
    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False


def auto_fp16(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def force_fp32(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """Load a local checkpoint. URLs stay refused; callers pass a downloaded file."""
    if not isinstance(filename, str) or filename.startswith(("http://", "https://", "open-mmlab://")):
        raise RuntimeError("pretrained URLs are not loaded; pass a local file")
    import torch

    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    incompatible = model.load_state_dict(cleaned, strict=strict)
    return incompatible


def get_dist_info():
    return 0, 1


def build_optimizer(*args, **kwargs):
    raise NotImplementedError


def build_runner(*args, **kwargs):
    raise NotImplementedError
