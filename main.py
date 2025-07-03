"""
pip install -U -r requirements.txt
torchrun --nproc_per_node=gpu main.py
"""

import argparse
from functools import partial
import itertools
import os
import sys
import time

from einops import rearrange
import numpy as np
import torch
import torch.distributed as distr
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.profiler import profile, record_function, ProfilerActivity

import torch.cuda.nvtx as nvtx

import simple_fsdp


# start from the beginning to track every gpu memory allocation
# otherwise we lost cpp tracestack for model initialization
torch.cuda.memory._record_memory_history(max_entries=10000000)


class Attention(nn.Module):
    def __init__(self, dim, head_dim, kv_reduce=1):
        super().__init__()
        assert dim % head_dim == 0, f"Bad {dim=}/{head_dim=}"
        assert (dim // head_dim) % kv_reduce == 0, f"Bad {kv_reduce=} for {dim=} and {head_dim=}"
        self.dim = dim
        self.head_dim = head_dim
        self.n_q_heads = dim // head_dim
        self.n_kv_heads = self.n_q_heads // kv_reduce
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    @record_function("Attention")
    def forward(self, x, mask):
        B, T, D = x.size()
        q = rearrange(self.q(x), "B T (Q D) -> B Q T D", Q=self.n_q_heads)
        k = rearrange(self.k(x), "B T (K D) -> B K T D", K=self.n_kv_heads)
        v = rearrange(self.v(x), "B T (V D) -> B V T D", V=self.n_kv_heads)
        # NOTE: I tried compiling only flex_attention with max-autotune-no-cudagraphs, but no difference.
        x = flex_attention(q, k, v, block_mask=mask,  # NB: scaled by 1/sqrt if scale=None, the default
                           enable_gqa=self.n_q_heads != self.n_kv_heads)
        o = self.o(rearrange(x, "B Q T D -> B T (Q D)"))
        return o

    def init_weights(self):
        # TODO: more careful
        nn.init.trunc_normal_(self.q.weight, mean=0.0, std=0.02)  # TODO: qk such that dot-var is 1
        nn.init.trunc_normal_(self.k.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.v.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.o.weight, mean=0.0, std=1/np.sqrt(self.dim))  # TODO: carefully


class MLP(nn.Module):
    def __init__(self, dim, grow=4):
        super().__init__()
        self.dim = dim
        self.grow = grow
        self.l1 = nn.Linear(dim, int(grow * dim))
        self.l2 = nn.Linear(int(grow * dim), dim)

    @record_function("MLP")
    def forward(self, x):
        x = self.l1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.l2(x)
        return x

    def init_weights(self):
        nn.init.trunc_normal_(self.l1.weight, mean=0.0, std=1/np.sqrt(self.dim * self.grow / 2))
        nn.init.trunc_normal_(self.l2.weight, mean=0.0, std=1/np.sqrt(self.dim * self.grow / 2))
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)


class Block(nn.Module):
    def __init__(self, dim, head_dim=128, grow=4, kv_reduce=4):
        super().__init__()
        self.att_ln = nn.LayerNorm(dim)  # TODO: better ln parametrization
        self.mlp_ln = nn.LayerNorm(dim)
        self.att = Attention(dim, head_dim, kv_reduce)
        self.mlp = MLP(dim, grow)

    @record_function("Block")
    def forward(self, x, mask):
        x = x + self.att(self.att_ln(x), mask)
        x = x + self.mlp(self.mlp_ln(x))
        return x

    def init_weights(self):
        self.att.init_weights()
        self.mlp.init_weights()
        self.att_ln.reset_parameters()
        self.mlp_ln.reset_parameters()


class SinCosPosEmb(nn.Module):
    def forward(self, x, pos=None) -> torch.Tensor:
        if pos is None:
            pos = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)

        # NOTE: could register_buffer this one, but how do I handle devices??
        #       Looks not worth it yet, it's 2ms out of 1s.
        d_model = x.shape[-1]
        ifreqs = torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) / d_model  # d_model/2
        freqs = 10000.0 ** (-ifreqs)

        x[..., 0::2] += torch.sin(pos[:, None] * freqs[None, :])
        x[..., 1::2] += torch.cos(pos[:, None] * freqs[None, :])

        return x


class SimpleTransformer(nn.Module):
    def __init__(self, dim, depth, vocab=32_768, head_dim=128, grow=4, kv_reduce=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.posemb = SinCosPosEmb()
        self.blocks = nn.ModuleList([Block(dim, head_dim, grow, kv_reduce) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

        self.dim = dim

    @record_function("Transformer")
    def forward(self, x, mask, positions=None):
        x = self.emb(x)
        x = self.posemb(x, positions)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln(x)
        x = self.head(x)
        return x

    def init_weights(self):
        nn.init.trunc_normal_(self.emb.weight, 0, 1/self.dim)
        self.ln.reset_parameters()  # TODO: better parametrization

        nn.init.zeros_(self.head.weight)

        for block in self.blocks:
            block.init_weights()


def print_mem(name, verbosity=1, rank=0):
    if verbosity < 1: return

    torch.cuda.synchronize()
    if rank == 0:
        print(f"Mem {name}: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB", flush=True)


def global_normf(x):
    if x is None:
        return None

    n = x.norm()

    # For DTensor, this returns a _NormPartial object which only holds the local "norm piece".
    # Only calling `full_tensor` on it syncs the output and gives each rank the same whole norm!
    if hasattr(n, 'full_tensor'):
        n = n.full_tensor()

    return n.item()


def global_norms(x):
    n = global_normf(x)
    return f"{n:.5f}" if n is not None else "None"


def print_pg(model, verbosity=2, rank=0, file=sys.stderr):
    if verbosity < 2: return

    # Print all param norms and gradnorms to stderr
    torch.cuda.synchronize()
    distr.barrier()
    for name, param in model.named_parameters():
        print(f"[{rank}] {name:>22s}: ‖p‖={global_norms(param)} ‖∇‖={global_norms(param.grad)}", file=file)


def main(args, rank, local_rank, world_size):
    printmem = partial(print_mem, verbosity=args.verbose, rank=rank)
    printpg = partial(print_pg, verbosity=args.verbose, rank=rank)

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
    with torch.device("meta"):
        model = SimpleTransformer(dim=4096, depth=4, kv_reduce=4)
    printmem("model(device=meta)")

    model = simple_fsdp.data_parallel(
        model, mesh,
        mode="fully_shard",
        ac_mode="full",  # Really just "none" (default) or not "none".
        mp_policy=simple_fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
    )
    printmem("data_parallel(model)")

    # Just checkin bruv
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        assert tensor.device == torch.device("meta")

    # Allocate buffers and sharded parameters on GPU
    model.to_empty(device=torch.cuda.current_device())
    printmem("model.to_empty(cuda)")

    # Run user-defined initializers
    with torch.no_grad():
      with simple_fsdp.disable_data_parallel():  # super important, otherwise nothing happens.
        model.init_weights() # or `model.apply(init_weights)`
    printmem("model.init_weights")
    printpg(model)

    # NOTE: I've seen this in torchtitan, but it doesn't have any effect for me yet
    torch._inductor.config.reorder_for_peak_memory = False  # Seen https://github.com/pytorch/torchtitan/blob/d9cc6b4df341eec27768b5ab9cead87ef595dbc2/torchtitan/experiments/simple_fsdp/parallelize.py#L96

    # NOTE: Optimizer doesn't alloc here, only allocs on `.step()`.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    printmem("optimizer(params)")
    vocab_size = model.emb.num_embeddings
    seq_len = 4096

    @record_function("fwd")
    @torch.compile
    def _fwd(batch, mask):
        logits = model(batch, mask)
        return F.cross_entropy(logits.view(-1, vocab_size), data.view(-1))

    @record_function("bwd")
    @torch.compile
    def _bwd(loss):
        loss.backward()
        optimizer.step()
        return loss

    # Let's overfit on a single batch, and also exclude this from timing.
    data = torch.randint(0, vocab_size, (8, seq_len), device="cuda")
    printmem("data")

    def causal_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    mask = create_block_mask(causal_mask_mod, data.size(0), None, seq_len, seq_len)
    printmem("mask")

    peak_mems = []
    step_times = []
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )

    for step in range(16):
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        if step > 1:
            nvtx.range_push("step")
        if step == 2:
            torch.cuda.cudart().cudaProfilerStart()
            prof.start()

        model.zero_grad(set_to_none=True)
        loss = _fwd(data, mask)
        _bwd(loss)
        print(f"[{rank}] step {step}: loss {loss.item():.2f}")  # Causes a transfer. TODO: avg globally.

        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)  # ms
        peak_mems.append(torch.cuda.max_memory_allocated() / 1024**2)  # MiB

        if step > 1:
            nvtx.range_pop()
        if step == 2:
            # dumping first 3 iterations from init are enough
            # to include optimizer states. otherwise the .pkl becomes too big 
            # and freezes chrome
            # drag .pkl file to https://docs.pytorch.org/memory_viz
            torch.cuda.memory._dump_snapshot(f"prof_memsnap_r{rank}.pkl")
        if step == 6:
            torch.cuda.cudart().cudaProfilerStop()
            prof.stop()
            prof.export_chrome_trace(f"prof_trace_r{rank}.json.gz")  # TODO: speedup gz
            prof.export_stacks(f"prof_stacks_cpu_r{rank}.txt")
        distr.barrier()  # Just for simplicity for now.
        printpg(model)

    print(f"[{rank}] Peak mems (med: {np.median(peak_mems):.1f}MiB): {' '.join(f'{t:.0f}' for t in peak_mems)}")
    print(f"[{rank}] Step times (med: {np.median(step_times):.1f}ms): {' '.join(f'{t:.0f}' for t in step_times)}")
    torch._dynamo.reset()  # Avoid hang: https://x.com/main_horse/status/1937900381574717940
    distr.destroy_process_group()
    print(f"[{rank}] Destroyed group")


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    parser = argparse.ArgumentParser(description="iykyk")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: print memory and timings. -vv: Also print expensive details like gradnorms.")
    args = parser.parse_args()

    main(args, rank=rank, local_rank=local_rank, world_size=world_size)
