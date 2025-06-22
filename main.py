"""
pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
torchrun --nproc_per_node=gpu train.py
"""

import os
import time

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
        y = F.gelu(y)
        y = self.l2(y)
        return x + y


class MyMLP(nn.Module):
    def __init__(self, vocab=32000, dim=1024, depth=1, device="meta"):
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

    # In theory we only need `init_device_mesh`, but in practice, we need this
    # whole verbose `init_process_group` or else the `barrier` will throw a warning.
    distr.init_process_group("nccl", rank=rank, world_size=world_size, device_id=local_rank)
    torch.cuda.set_device(local_rank)
    mesh = distr.device_mesh.init_device_mesh(
        "cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",)  # Add "tp" for 2d parallel
    )

    # Later I want to do the "meta" device thing to never materialize full params.
    # model = MyMLP(device="meta")
    model = MyMLP(device="cuda")

    # And I will also want to run at the very least the compute in bf16.
    # Not yet sure if this is the way, will play around.
    # policy = simple_fsdp.MixedPrecisionPolicy(
    #     param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    # )

    # I'll also want to do activation checkpointing, of course.
    model = simple_fsdp.data_parallel(
        model, mesh, mode="fully_shard", # ac_mode="full", mp_policy=policy
    )

    # Whoops!? What happened, why is this now a single row only???
    print("First linear weights: ", model.blocks[0].l1.weight)
    # Curiously, the exact same thing happens with mode="replicate" so idk man...

    # Of course, SimpleFSDP supposedly does nice overlap etc only when compiled.
    # model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    vocab_size = model.embed.num_embeddings
    seq_len = 128

    # Let's overfit on a single batch, and also exclude this from timing.
    data = torch.randint(0, vocab_size, (8, seq_len), device="cuda")

    for step in range(16):
        if step == 1:  # Don't measure compilation time
            t0 = time.perf_counter()

        model.zero_grad(set_to_none=True)
        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, vocab_size), data.view(-1))
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"step {step}: loss {loss.item()}")  # Causes a transfer.
        distr.barrier()  # Just for simplicity for now.

    t1 = time.perf_counter()
    print(f"Took {t1 - t0:.1f}s so {(t1 - t0) / step * 1000:.1f}ms/step")
    distr.destroy_process_group()


if __name__ == "__main__":
    # I don't need all these, but o3 told me this for sanity checks
    print(torch._dynamo.list_backends()[:3])
    os.environ["TORCH_COMPILE_DEBUG"] = "1"        # rich compiler logs :contentReference[oaicite:7]{index=7}
    os.environ["TORCH_LOGS"] = "dynamo,inductor"   # print break / recompile info
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = False          # crash on silent fallbacks

    main()
