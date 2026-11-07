"""Fast shape/gradient tests for every head x label-mode x encoder family (CPU, random weights).
python tests/test_models.py"""
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frpv.config import load_config  # noqa: E402
from frpv.models.losses import compute_loss  # noqa: E402
from frpv.models.network import FibrosisNet, rater_wall_fib, wall_fib_probs  # noqa: E402
from frpv.stain.physics import REFERENCE_VECTORS  # noqa: E402


def run(encoder, head, mode, branch="conc", size=128):
    cfg = load_config("configs/base.yaml", overrides=[
        f"model.encoder={encoder}", "model.pretrained=false", f"model.head={head}",
        f"model.label_mode={mode}", "model.decoder_dim=32", "model.detail_channels=[8,8,8,8]",
        f"stain.branch_input={branch}", "model.fm_input_mpp=0.252"])
    net = FibrosisNet(cfg, REFERENCE_VECTORS)
    x = torch.rand(2, 3, size, size + 32)
    y = torch.randint(0, 3, (2, 2, size, size + 32))
    roi = torch.ones(2, size, size + 32)
    out = net(x)
    loss, _ = compute_loss(out, y, roi, cfg, net)
    loss.backward()
    pw, pf = wall_fib_probs(out["z"], head)
    assert pw.shape == (2, size, size + 32)
    if head != "multilabel":
        assert bool((pf <= pw + 1e-6).all()), "soft containment violated"
    rp = rater_wall_fib(out, head, mode)
    if mode in ("offset", "cm_pixel"):
        assert rp is not None and len(rp) == 2
    g = [p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
    assert len(g) > 0
    lora_b = [p for n, p in net.named_parameters() if "lora_a" in n]
    if lora_b:   # LoRA A gets gradient only via B (zero-init) -> check B instead
        lb = [p for n, p in net.named_parameters() if "lora_b" in n]
        assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in lb), "LoRA grads missing"
    return float(loss.detach())


if __name__ == "__main__":
    n = 0
    for head, mode in itertools.product(["nested", "flat", "multilabel"],
                                        ["e1", "e2", "mean_soft", "dual_loss", "cm_pixel", "offset"]):
        if head == "multilabel" and mode == "cm_pixel":
            continue
        l = run("cnn:resnet18", head, mode)
        n += 1
        print(f"ok cnn {head:10s} {mode:10s} loss={l:.3f}")
    for br in ["rgb", "rgb+conc", "none"]:
        run("cnn:resnet18", "nested", "offset", br)
        print("ok branch", br)
    for ft in ["lora"]:
        l = run("fm:dinov2_s", "nested", "offset", size=140)
        print(f"ok fm dinov2_s nested offset (LoRA) loss={l:.3f}")
    print(f"all {n + 4} tests passed")


def test_checkpoint_roundtrip():
    """Slim checkpoint (trainable + buffers) must restore the exact outputs of a LoRA model."""
    from frpv.train import load_trainable, trainable_state
    cfg = load_config("configs/base.yaml", overrides=[
        "model.encoder=fm:dinov2_s", "model.pretrained=false", "model.decoder_dim=32",
        "model.detail_channels=[8,8,8,8]", "model.fm_input_mpp=0.252"])
    torch.manual_seed(0)
    a = FibrosisNet(cfg, REFERENCE_VECTORS)
    for n, p in a.named_parameters():          # perturb trainable weights
        if p.requires_grad:
            p.data.add_(0.01 * torch.randn_like(p))
    sd = trainable_state(a)
    n_all = sum(v.numel() for v in a.state_dict().values())
    n_saved = sum(v.numel() for v in sd.values())
    torch.manual_seed(0)
    b = FibrosisNet(cfg, REFERENCE_VECTORS)    # same frozen init (stands in for the hub weights)
    load_trainable(b, sd)
    a.eval(), b.eval()
    x = torch.rand(1, 3, 140, 140)
    with torch.no_grad():
        assert torch.allclose(a(x)["z"], b(x)["z"], atol=1e-5)
    print(f"ok checkpoint round-trip (saved {n_saved/n_all:.1%} of the tensors' elements)")


if __name__ == "__main__":
    test_checkpoint_roundtrip()
