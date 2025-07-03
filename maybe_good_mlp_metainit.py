"""
pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
torchrun --nproc_per_node=gpu main.py
"""

import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distr

import simple_fsdp


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, 4 * dim)
        self.l2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        y = self.l1(x)
        y = F.gelu(y, approximate="tanh")
        y = self.l2(y)
        return x + y

    def init_weights(self):
        nn.init.trunc_normal_(self.l1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.l2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)


class MyMLP(nn.Module):
    def __init__(self, vocab=32_768, dim=4096, depth=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        x = self.emb(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        x = self.head(x)
        return x

    def init_weights(self):
        self.emb.reset_parameters()  # TODO: Better init
        self.ln.reset_parameters()  # TODO: better parametrization

        nn.init.zeros_(self.head.weight)

        for block in self.blocks:
            block.init_weights()


def printmem(name, rank=0):
    torch.cuda.synchronize()
    distr.barrier()
    if rank == 0:
        print(f"Mem {name}: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB", flush=True)


def main(rank, local_rank, world_size):
    # In theory we only need `init_device_mesh`, but in practice, we need this
    # whole verbose `init_process_group` or else the `barrier` will throw a warning.
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    distr.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    mesh = distr.device_mesh.init_device_mesh(
        "cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",)  # Add "tp" for 2d parallel
    )
    printmem("at init", rank)

    with torch.device("meta"):
        model = MyMLP(depth=1)
    printmem("model(device=meta)", rank)

    # (no real tensor here, so can't print any stats ofc)

    model = simple_fsdp.data_parallel(
        model, mesh,
        mode="fully_shard",
        ac_mode="full",  # Really just "none" (default) or not "none".
        mp_policy=simple_fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
    )
    printmem("data_parallel(model)", rank)

    # Just checkin bruv
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        assert tensor.device == torch.device("meta")

    # Allocate buffers and sharded parameters on GPU
    model.to_empty(device=torch.cuda.current_device())
    printmem("model.to_empty(cuda)", rank)

    # Should all just be garbage memory (likely but not certainly zero)
    for name, param in model.named_parameters():
        if "bias" in name: continue  # Just for brevity, skip.
        print(f"[STATS {rank}] {name}: {param.norm().full_tensor().item():.5f}")

    # Run user-defined initializers
    with torch.no_grad():
      with simple_fsdp.disable_data_parallel():  # super important, otherwise nothing happens.
        model.init_weights() # or `model.apply(init_weights)`
    printmem("model.init_weights", rank)

    for name, param in model.named_parameters():
        if "bias" in name: continue  # Just for brevity, skip.
        print(f"[STATS {rank}] {name}: {param.norm().full_tensor().item():.5f}")

    # NOTE: I've seen this in torchtitan, but it doesn't have any effect for me yet
    torch._inductor.config.reorder_for_peak_memory = False  # Seen https://github.com/pytorch/torchtitan/blob/d9cc6b4df341eec27768b5ab9cead87ef595dbc2/torchtitan/experiments/simple_fsdp/parallelize.py#L96

    # NOTE: Optimizer doesn't alloc here, only allocs on `.step()`.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    printmem("optimizer(params)", rank)
    vocab_size = model.emb.num_embeddings
    seq_len = 4096

    # NB: This is actually both faster and more memory efficient than compile(model),
    #     see README timing tables!
    @torch.compile
    def _fwd_bwd(batch):
        logits = model(batch)
        loss = F.cross_entropy(logits.view(-1, vocab_size), data.view(-1))
        loss.backward()
        optimizer.step()
        return loss

    # Let's overfit on a single batch, and also exclude this from timing.
    data = torch.randint(0, vocab_size, (8, seq_len), device="cuda")
    printmem("data", rank)

    peak_mems = []
    step_times = []

    for step in range(8):
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        model.zero_grad(set_to_none=True)
        loss = _fwd_bwd(data)
        print(f"[{rank}] step {step}: loss {loss.item():.2f}")  # Causes a transfer.

        for name, param in model.named_parameters():
            print(f"[STATS {rank}] {name}: {param.norm().full_tensor().item():.5f} {param.grad.norm().full_tensor().item():.5f}")

        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)  # ms
        peak_mems.append(torch.cuda.max_memory_allocated() / 1024**2)  # MiB

        distr.barrier()  # Just for simplicity for now.

    print(f"[{rank}] Peak mems (med: {np.median(peak_mems):.1f}MiB): {' '.join(f'{t:.0f}' for t in peak_mems)}")
    print(f"[{rank}] Step times (med: {np.median(step_times):.1f}ms): {' '.join(f'{t:.0f}' for t in step_times)}")
    torch._dynamo.reset()  # Avoid hang: https://x.com/main_horse/status/1937900381574717940
    distr.destroy_process_group()
    print(f"[{rank}] Destroyed group")


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank=rank, local_rank=local_rank, world_size=world_size)
