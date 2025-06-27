"""
pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
torchrun --nproc_per_node=gpu main.py
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distr

import simple_fsdp


class Block(nn.Module):
    def __init__(self, dim, device="meta"):
        super().__init__()
        self.l1 = nn.Linear(dim, 4 * dim, device=device)
        self.l2 = nn.Linear(4 * dim, dim, device=device)

    def forward(self, x):
        y = self.l1(x)
        y = F.gelu(y, approximate="tanh")
        y = self.l2(y)
        return x + y


class MyMLP(nn.Module):
    def __init__(self, vocab=32_768, dim=4096, depth=4, device="meta"):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, device=device)
        self.blocks = nn.ModuleList([Block(dim, device=device) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim, device=device)
        self.head = nn.Linear(dim, vocab, bias=False, device=device)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        x = self.head(x)
        return x


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    def printmem(name):
        torch.cuda.synchronize()
        if rank == 0:
            print(f"Mem {name}: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB", flush=True)

    # In theory we only need `init_device_mesh`, but in practice, we need this
    # whole verbose `init_process_group` or else the `barrier` will throw a warning.
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    distr.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    mesh = distr.device_mesh.init_device_mesh(
        "cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",)  # Add "tp" for 2d parallel
    )
    printmem("at init")

    # Later I want to do the "meta" device thing to never materialize full params.
    # model = MyMLP(device="meta")
    model = MyMLP(device="cuda")
    printmem("model(device=cuda)")

    # And I will also want to run at the very least the compute in bf16.
    # Not yet sure if this is the way, will play around.
    policy = simple_fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        # param_dtype=torch.bfloat16, reduce_dtype=None
    )

    # I'll also want to do activation checkpointing, of course.
    model = simple_fsdp.data_parallel(
        model, mesh, mode="fully_shard",
        ac_mode="full",  # Really just "none" (default) or not "none".
        mp_policy=policy,
    )
    printmem("data_parallel(model)")

    # Of course, SimpleFSDP supposedly does nice overlap etc only when compiled.

    # NOTE: I've seen this in torchtitan, but it doesn't have any effect yet:
    torch._inductor.config.reorder_for_peak_memory = False  # Seen https://github.com/pytorch/torchtitan/blob/d9cc6b4df341eec27768b5ab9cead87ef595dbc2/torchtitan/experiments/simple_fsdp/parallelize.py#L96

    # NOTE: if I don't specify reduce-overhead, peak mem is high! But it's not specified in torchtitan?
    # https://github.com/pytorch/torchtitan/blob/d9cc6b4df341eec27768b5ab9cead87ef595dbc2/torchtitan/experiments/simple_fsdp/parallelize.py#L97
    # model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    # Fun fact: with mode="reduce-overhead", the destroy_process_group hangs forever!

    # NOTE: Optimizer doesn't alloc here, only allocs on `.step()`.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    printmem("optimizer(params)")
    vocab_size = model.embed.num_embeddings
    seq_len = 4096

    compile_kw = {
        # "max_autotune": True,
        # "epilogue_fusion": True,
        "triton.cudagraphs": True,  # This one and only this one HALVES MEMORY!
        # "shape_padding": True,
    }

    model = torch.compile(model, options=compile_kw)

    # @torch.compile(fullgraph=True, mode="reduce-overhead")
    # @torch.compile(options=compile_kw)
    def _fwd(batch):
      # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch)
        loss = F.cross_entropy(logits.view(-1, vocab_size), data.view(-1))
        return loss

    # @torch.compile(options=compile_kw)
    def _bwd(loss):
        loss.backward()
        optimizer.step()

    # Let's overfit on a single batch, and also exclude this from timing.
    data = torch.randint(0, vocab_size, (8, seq_len), device="cuda")
    printmem("data")

    peak_mems = []
    step_times = []

    for step in range(16):
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        model.zero_grad(set_to_none=True)
        # logits = model(data)
        # loss = F.cross_entropy(logits.view(-1, vocab_size), data.view(-1))
        # loss.backward()
        # optimizer.step()
        # loss = _step(data)
        loss = _fwd(data)
        _bwd(loss)
        print(f"[{rank}] step {step}: loss {loss.item():.2f}")  # Causes a transfer.

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
    # I don't need all these, but o3 told me this for sanity checks
    print(torch._dynamo.list_backends()[:3])
    os.environ["TORCH_COMPILE_DEBUG"] = "1"        # rich compiler logs :contentReference[oaicite:7]{index=7}
    os.environ["TORCH_LOGS"] = "dynamo,inductor"   # print break / recompile info
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = False          # crash on silent fallbacks

    main()
