# Chess_Brain_mp_spawn_4_12_26.py

DDP (DistributedDataParallel) version of the chess training script. All GPUs train simultaneously via NCCL instead of sequentially.

## Usage

```bash
python Chess_Brain_mp_spawn_4_12_26.py
```

Same interactive prompts as before: load/create model, select GPUs, pick data file, set hyperparameters. No torchrun or special launcher needed.

## What changed from Chess_Brain_3_21_26.py

The old multi-GPU code ran one GPU at a time in a loop — forward on GPU 0, wait, forward on GPU 1, wait, etc. Each GPU was idle 75% of the time. This version uses PyTorch DDP so all GPUs work in parallel.

| | Old (sequential) | New (DDP) |
|---|---|---|
| GPU utilization | ~25% each (one active at a time) | ~100% all simultaneously |
| Gradient sync | Manual copy through CPU | NCCL GPU-to-GPU |
| Wall-clock speedup | None vs single GPU | ~3-4x with 4 GPUs |
| Launch command | `python script.py` | `python script.py` (same) |
| Requires torchrun | No | No |

## Checkpoint compatibility

- **Old checkpoints load into this script.** Optimizer state lists (from the old multi-GPU format) are handled automatically.
- **New checkpoints work with Chess_Inference.py and Chess_4_8_26.py.** Same `model_state_dict`, same `hyperparameters` dict, same `tokenizer` field.
- **No GPU count restriction.** Train on 4 GPUs, resume on 1 (or vice versa). The old script required matching GPU counts because it saved per-GPU optimizer states. This version saves a single optimizer state.

## How DDP works here

1. `mp.spawn` launches one worker process per GPU (no torchrun needed).
2. Each worker creates its own model replica on its assigned GPU.
3. `DistributedDataParallel` wraps each model — NCCL synchronizes gradients automatically after every `backward()`.
4. `DistributedSampler` gives each GPU a different shard of the dataset (no duplicate work).
5. Only rank 0 handles: printing, Ctrl+C menu, checkpoint saving, sample move generation.
6. Ctrl+C decisions are broadcast from rank 0 to all workers via `dist.broadcast`.

For single GPU, DDP is skipped entirely — no spawn, no overhead.

## Token modes

Both tokenization modes are fully supported:

- **Classic mode** (1 token per move, ~20K vocab, 64x63x5 move tokens) — single `lm_head` output, weight-tied to embeddings.
- **4-token mode** (4 tokens per ply, 140 vocab: COLOR/FROM/TO/PROMO) — four role-specific output heads with FROM-conditioned TO prediction.

Mode is selected at training startup (new model) or auto-detected from checkpoint. DDP is transparent to both — it synchronizes gradients after `backward()` regardless of how the loss was computed internally.

## Key differences for users

- **Batch size** is the total across all GPUs. With batch_size=512 on 4 GPUs, each GPU processes 128 samples per step.
- **Loss values** reported are from rank 0's shard only (representative, not averaged across ranks).
- **Ctrl+C** in multi-GPU mode: pause, change LR, or quit. Loading new data requires quit and restart (can't synchronize new data across DDP workers mid-training). Single-GPU mode supports all options including mid-training data change.
- **torch.compile** is applied in single-GPU mode only (DDP handles optimization differently).

## Files

| File | Role |
|---|---|
| `Chess_Brain_mp_spawn_4_12_26.py` | Training (this version, DDP multi-GPU) |
| `Chess_Brain_3_21_26.py` | Training (original, sequential multi-GPU) |
| `Chess_Inference.py` | Inference engine (works with checkpoints from either) |
| `Chess_4_8_26.py` | Pygame GUI (works with checkpoints from either) |

---

## Pitfalls when adding mp.spawn DDP to existing training code

Reference for next time. These are the bugs we hit and fixed, in order.

### 1. CUDA tensors can't be pickled across processes

**Symptom:** `RuntimeError: pidfd_getfd: Operation not permitted` at spawn time.

**Cause:** `mp.spawn` serializes (pickles) all arguments to send to child processes. If any argument contains CUDA tensors (model state dict, optimizer state dict loaded onto GPU), pickling fails.

**Fix:** Move everything to CPU before spawning. Use a recursive helper:
```python
def _to_cpu(obj):
    if isinstance(obj, torch.Tensor): return obj.cpu()
    elif isinstance(obj, dict): return {k: _to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_cpu(v) for v in obj]
    return obj
```
Apply to model state dict and optimizer state dict before putting them in `train_args`.

### 2. Do NOT set mp.set_start_method('spawn') globally

**Symptom:** Flood of "Using CUDA with optimized settings" messages. Dozens of 444 MiB CUDA contexts created on GPU 0. OOM crash.

**Cause:** `mp.set_start_method('spawn', force=True)` changes the default for ALL multiprocessing — including DataLoader workers. Each DataLoader worker (8 per GPU x 4 GPUs = 32 processes) re-imports the module via spawn, running module-level CUDA initialization code. Each creates a ~444 MiB CUDA context on GPU 0.

**Fix:** Remove the global `set_start_method` call. `mp.spawn` already uses spawn internally for DDP workers. DataLoader workers will use fork (Linux default) — they share parent memory without re-importing the module.

```python
# BAD — affects DataLoader workers too
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # DELETE THIS

# GOOD — mp.spawn already defaults to spawn start_method internally
```

### 3. Guard module-level CUDA init from running in DDP workers

**Symptom:** Every spawned DDP worker creates a CUDA context on GPU 0 (the default device) during module import, before `torch.cuda.set_device(rank)` is called. 4 workers x 444 MiB = 1.7 GB wasted on GPU 0.

**Cause:** Module-level code like `torch.cuda.empty_cache()`, `torch.cuda.set_stream(...)`, or `torch.cuda.set_per_process_memory_fraction(...)` initializes a CUDA context on the default device (GPU 0). When mp.spawn creates child processes (spawn start method), they re-import the module and execute this code before your worker function runs.

**Fix:** Set an environment variable before `mp.spawn`. Check it at module level to skip CUDA init in workers:

```python
# Module level:
if torch.cuda.is_available() and not os.environ.get('_MY_DDP_WORKER'):
    torch.cuda.empty_cache()  # Only main process
    torch.cuda.set_per_process_memory_fraction(0.80)
elif torch.cuda.is_available():
    device = torch.device('cuda')  # Minimal setup for workers

# Before mp.spawn:
os.environ['_MY_DDP_WORKER'] = '1'
mp.spawn(worker_fn, ...)
os.environ.pop('_MY_DDP_WORKER', None)

# In worker, set per-rank CUDA config:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### 4. Tokenize data ONCE before spawn, not in each worker

**Symptom:** Tokenization prints appear N times (once per GPU). N copies of the full token tensor in RAM. Startup takes N times longer.

**Cause:** If you pass raw text to workers and each one tokenizes independently, you get N copies of the work and N copies of the data.

**Fix:** Tokenize once in the main process. Put tensors in shared memory. Pass the tensors (not text) to workers:
```python
tokens_tensor = tokenize(text)
tokens_tensor.share_memory_()  # All workers access same memory, zero copies
# Pass tokens_tensor in train_args instead of text
```
Workers create a lightweight Dataset wrapper around the shared tensors.

### 5. All ranks must handle SIGINT (Ctrl+C)

**Symptom:** Ctrl+C crashes ranks 1-3 with `KeyboardInterrupt`. Only rank 0 catches it gracefully.

**Cause:** SIGINT is sent to the entire process group. If only rank 0 has a signal handler, other ranks get the default handler which raises `KeyboardInterrupt` and kills the process.

**Fix:**
```python
if rank == 0:
    signal.signal(signal.SIGINT, my_handler)  # Catch and set flag
else:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore silently
```
Then rank 0 broadcasts the interrupt decision to all ranks via `dist.broadcast`.

### 6. Wrap the entire worker in try/finally

**Symptom:** `destroy_process_group() was not called` warnings. TCPStore connection errors at exit.

**Cause:** If a worker crashes (OOM, bug, etc.), `_ddp_cleanup()` is never called. Other ranks hang waiting for the dead rank, then NCCL timeout kills them with ugly errors.

**Fix:**
```python
def worker(rank, ...):
    try:
        _ddp_setup(rank, world_size)
        # ... all training code ...
    finally:
        _ddp_cleanup()
```

### 7. Use PYTORCH_ALLOC_CONF, not PYTORCH_CUDA_ALLOC_CONF

**Symptom:** Flood of `PYTORCH_CUDA_ALLOC_CONF is deprecated` warnings from every process.

**Fix:** PyTorch 2.9+ renamed the env var. Use `PYTORCH_ALLOC_CONF`.

### 8. Fresh training: seed random init so all ranks match

**Symptom:** DDP errors about parameter mismatch across ranks on fresh (non-checkpoint) training.

**Cause:** Each rank initializes model weights with different random seeds. DDP requires identical starting weights.

**Fix:** Set a fixed seed before model creation in each worker:
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = MyModel(...)  # All ranks get identical random weights
```
For checkpoint resume this is harmless (load_state_dict overwrites the random init).

### 9. Spawned workers have stdin replaced with /dev/null

**Symptom:** `input()` in rank 0 worker immediately returns EOFError. Interactive Ctrl+C menu is unusable — training quits or continues without waiting for user input.

**Cause:** Python's `multiprocessing.util._close_stdin()` is called during child process bootstrap. It closes the real stdin and replaces `sys.stdin` with `/dev/null`. This is by design to prevent multiple processes from competing for terminal input. But it breaks any interactive prompt in the worker.

**Fix:** Rank 0 reopens stdin from the terminal at worker startup:
```python
if rank == 0:
    try:
        sys.stdin = open('/dev/tty', 'r')
    except OSError:
        pass  # No terminal available (headless/script mode)
```
Only rank 0 needs this (other ranks never read input). Also make `input()` failure safe — default to "continue training" not "quit":
```python
try:
    choice = input("Choice: ").strip()
except (KeyboardInterrupt, EOFError):
    choice = ''  # continue, don't quit
```

### 10. Main process must ignore SIGINT during mp.spawn

**Symptom:** Ctrl+C kills main process (traceback in `mp.spawn.join()`), which orphans workers or breaks their stdin.

**Cause:** Ctrl+C sends SIGINT to the entire process group. Workers have handlers, but the main process (sitting in `mp.spawn.join()`) has the default handler which raises `KeyboardInterrupt`.

**Fix:**
```python
old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
try:
    mp.spawn(worker_fn, ...)
finally:
    signal.signal(signal.SIGINT, old_handler)
```

### Summary checklist for adding DDP to any training script

- [ ] All data passed to `mp.spawn` must be CPU tensors (no CUDA tensors)
- [ ] Do NOT call `mp.set_start_method('spawn')` globally
- [ ] Guard module-level CUDA init with env var to skip in workers
- [ ] Tokenize/preprocess data ONCE before spawn, use `share_memory_()`
- [ ] All ranks must handle SIGINT (rank 0 catches, others ignore)
- [ ] Main process must ignore SIGINT during mp.spawn
- [ ] Rank 0 must reopen stdin from `/dev/tty` for interactive input
- [ ] `input()` failure must default to continue, not quit
- [ ] Wrap worker in `try/finally` with cleanup
- [ ] Use `PYTORCH_ALLOC_CONF` not `PYTORCH_CUDA_ALLOC_CONF`
- [ ] Set fixed random seed before model init for fresh training
- [ ] Only rank 0 does: printing, saving, file dialogs, interactive input
- [ ] All collective ops (`dist.broadcast`) must be called by ALL ranks
