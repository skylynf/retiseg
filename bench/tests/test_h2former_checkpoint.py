import torch

from bench.models.h2former import _checkpoint_swin_blocks, _load_res34_swin_MS, _rgb_stem


def _small_net(seed):
    torch.manual_seed(seed)
    net = _load_res34_swin_MS()(64, 5)
    _rgb_stem(net)
    return net.train()


def test_checkpointed_swin_blocks_give_the_same_loss_and_gradients():
    plain = _small_net(0)
    checkpointed = _small_net(0)
    checkpointed.load_state_dict(plain.state_dict())
    _checkpoint_swin_blocks(checkpointed)
    image = torch.randn(2, 3, 64, 64)
    grads = []
    for net in (plain, checkpointed):
        torch.manual_seed(1)
        loss = net(image).square().mean()
        loss.backward()
        grads.append({name: param.grad.clone() for name, param in net.named_parameters() if param.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        assert torch.allclose(grads[0][name], grads[1][name], rtol=1e-4, atol=1e-6), name
    assert all(layer.use_checkpoint for layer in checkpointed.swin_layers)
