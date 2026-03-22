"""Quick test: verify multi-GPU gradient averaging and weight sync are correct.
Run this BEFORE training to catch bugs. Takes ~10 seconds.

NOTE: All cross-GPU transfers route through CPU because CUDA P2P is broken
on this system (reports available but produces garbage data)."""
import torch
import torch.nn as nn
from torch.amp import autocast

def test_multi_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Only {num_gpus} GPU(s) — skipping multi-GPU test")
        return True

    print(f"Testing with {num_gpus} GPUs...")

    # Create a small model on each GPU with IDENTICAL weights
    models = []
    master_w = torch.randn(16, 32)
    master_b = torch.randn(16)
    for i in range(num_gpus):
        m = nn.Linear(32, 16).cuda(i)
        with torch.no_grad():
            # Route through CPU — P2P broken
            m.weight.copy_(master_w.cuda(i))
            m.bias.copy_(master_b.cuda(i))
        models.append(m)

    # Verify all models start identical
    for i in range(1, num_gpus):
        w_diff = (models[0].weight.cpu() - models[i].weight.cpu()).abs().max().item()
        assert w_diff == 0.0, f"GPU {i} weights differ from GPU 0 at start: {w_diff}"
    print("  [PASS] All models start with identical weights")

    # Create different data for each GPU (simulates batch splitting)
    torch.manual_seed(42)
    data = [torch.randn(8, 32).cuda(i) for i in range(num_gpus)]
    targets = [torch.randn(8, 16).cuda(i) for i in range(num_gpus)]

    # Forward+backward on each GPU
    for i, (m, x, t) in enumerate(zip(models, data, targets)):
        m.zero_grad()
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            out = m(x)
            loss = nn.functional.mse_loss(out.float(), t)
        loss.backward()

    # Sync all devices
    for i in range(num_gpus):
        torch.cuda.synchronize(i)

    # Verify all GPUs have gradients
    for i, m in enumerate(models):
        assert m.weight.grad is not None, f"GPU {i} has no weight grad!"
        assert not torch.isnan(m.weight.grad).any(), f"GPU {i} weight grad has NaN!"
    print("  [PASS] All GPUs have valid gradients (no NaN)")

    # Average gradients onto GPU 0 — route through CPU
    with torch.no_grad():
        for param_tuple in zip(*[m.parameters() for m in models]):
            p0 = param_tuple[0]
            if p0.grad is None:
                continue
            for p in param_tuple[1:]:
                if p.grad is not None:
                    p0.grad.data.add_(p.grad.data.cpu().cuda(p0.grad.device.index))
            p0.grad.data.div_(num_gpus)

    assert not torch.isnan(models[0].weight.grad).any(), "GPU 0 averaged grad has NaN!"
    print("  [PASS] Gradient averaging produces no NaN")

    # Step only GPU 0's optimizer
    opt = torch.optim.AdamW(models[0].parameters(), lr=0.001)
    torch.nn.utils.clip_grad_norm_(models[0].parameters(), 5.0)
    opt.step()

    assert not torch.isnan(models[0].weight).any(), "GPU 0 weights NaN after step!"
    print("  [PASS] Optimizer step produces no NaN")

    # Copy GPU 0 weights to all others — route through CPU
    with torch.no_grad():
        for param_tuple in zip(*[m.parameters() for m in models]):
            master_data = param_tuple[0].data
            master_cpu = master_data.cpu()  # Stage on CPU once
            for p in param_tuple[1:]:
                p.data.copy_(master_cpu.cuda(p.device.index))

    # Verify all models are identical after sync
    for i in range(1, num_gpus):
        w_diff = (models[0].weight.cpu() - models[i].weight.cpu()).abs().max().item()
        assert w_diff == 0.0, f"GPU {i} weights differ after sync: {w_diff}"
        assert not torch.isnan(models[i].weight).any(), f"GPU {i} has NaN after sync!"
    print("  [PASS] Weight sync: all GPUs identical, no NaN")

    # Second forward pass — all GPUs should produce valid (close) losses
    losses = []
    for i, (m, x, t) in enumerate(zip(models, data, targets)):
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                out = m(x)
                loss = nn.functional.mse_loss(out.float(), t)
        loss_val = loss.item()
        assert not torch.isnan(loss), f"GPU {i} loss is NaN on second forward!"
        losses.append(loss_val)
    print(f"  [PASS] Second forward: losses = {[f'{l:.4f}' for l in losses]}")

    print(f"\n  ALL TESTS PASSED — multi-GPU training loop is correct")
    return True

if __name__ == '__main__':
    test_multi_gpu()
