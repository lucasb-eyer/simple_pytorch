"""
pip install -U -r requirements.txt
torchrun --nproc_per_node=gpu longwajong.py

NOTE: I tried to make it run on non-cuda/cpu with gloo etc.
      However, I failed: a lot of DTensor stuff seems to break then.
"""

import argparse
from datetime import datetime
from functools import cache, partial
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
from torch.utils.checkpoint import checkpoint

import simple_fsdp


# Allow using the (lower-precision) tensorcores for all fp32 matmuls.
# See https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
# torch.backends.fp32_precision = "tf32"
# SOMETHING in torch compile uses the old API, forcing us to use that too:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# But on a quick try it did make no diff?

# Reduce limit to make the issue appear faster
# torch._dynamo.config.recompile_limit = 1


# 1: Compiled flex_attention is necessary to checkpoint the attention block, see:
# https://github.com/pytorch/pytorch/issues/147879#issuecomment-3041193259
# 2: max-autotune-no-cudagraphs takes a long time, but did 1039ms->1028ms on 4k seqlen.
# cflex_attention = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
cflex_attention = torch.compile(flex_attention)


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
        T, D = x.size()
        q = rearrange(self.q(x), "T (Q D) -> 1 Q T D", Q=self.n_q_heads)
        k = rearrange(self.k(x), "T (K D) -> 1 K T D", K=self.n_kv_heads)
        v = rearrange(self.v(x), "T (V D) -> 1 V T D", V=self.n_kv_heads)
        x = cflex_attention(q, k, v, block_mask=mask,  # NB: scaled by 1/sqrt if scale=None, the default
                            enable_gqa=self.n_q_heads != self.n_kv_heads)
        # x = F.scaled_dot_product_attention(  # NB: scaled by 1/sqrt by default
        #     q, k, v, is_causal=True, enable_gqa=self.n_q_heads != self.n_kv_heads)
        o = self.o(rearrange(x, "1 Q T D -> T (Q D)"))
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
    def __init__(self, dim, head_dim=128, grow=4, kv_reduce=4, remat=True):
        super().__init__()
        self.att_ln = nn.LayerNorm(dim)  # TODO: better ln parametrization
        self.mlp_ln = nn.LayerNorm(dim)
        self.att = Attention(dim, head_dim, kv_reduce)
        self.mlp = MLP(dim, grow)
        self.remat = remat

    @record_function("Block")
    def forward(self, x, mask):
        if self.remat:
            x = x + checkpoint(lambda y, mask: self.att(self.att_ln(y), mask), x, mask, use_reentrant=False)
            x = x + checkpoint(lambda y: self.mlp(self.mlp_ln(y)), x, use_reentrant=False)
        else:
            x = x + self.att(self.att_ln(x), mask)
            x = x + self.mlp(self.mlp_ln(x))
        return x

    def init_weights(self):
        self.att.init_weights()
        self.mlp.init_weights()
        self.att_ln.reset_parameters()
        self.mlp_ln.reset_parameters()


class SinCosPosEmb(nn.Module):
    def forward(self, x, pos) -> torch.Tensor:
        # NOTE: could register_buffer this one, but how do I handle devices??
        #       Looks not worth it yet, it's 2ms out of 1s.
        d_model = x.shape[-1]
        ifreqs = torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) / d_model  # d_model/2
        freqs = 10000.0 ** (-ifreqs)

        x[..., 0::2] += torch.sin(pos[:, None] * freqs[None, :])
        x[..., 1::2] += torch.cos(pos[:, None] * freqs[None, :])

        return x


class SimpleTransformer(nn.Module):
    def __init__(self, dim, depth, vocab=32_768, **block_kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.posemb = SinCosPosEmb()
        self.blocks = nn.ModuleList([Block(dim, **block_kw) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

        self.dim = dim

    @record_function("Transformer")
    def forward(self, x, mask, positions):
        x = self.emb(x)
        x = self.posemb(x, positions)
        for blk in self.blocks:
            x = blk(x, mask)
            # x = checkpoint(blk, x, mask, use_reentrant=False)
        x = self.ln(x)
        x = self.head(x)
        return x

    def init_weights(self):
        nn.init.trunc_normal_(self.emb.weight, 0, 1/self.dim)
        self.ln.reset_parameters()  # TODO: better parametrization

        nn.init.zeros_(self.head.weight)

        for block in self.blocks:
            block.init_weights()

def main(args, rank, local_rank, world_size):
    printpg = partial(print_pg, verbosity=args.verbose, rank=rank)
    prints = partial(print_stamped, rank=rank)
    prints(f"Running with arguments: {args}")

    # start from the beginning to track every gpu memory allocation
    # otherwise we lost cpp tracestack for model initialization
    torch.cuda.memory._record_memory_history(max_entries=10000000)

    # In theory we only need `init_device_mesh`, but in practice, we need this
    # whole verbose `init_process_group` or else the `barrier` will throw a warning.
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    distr.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    mesh = distr.device_mesh.init_device_mesh(
        "cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",)  # Add "tp" for 2d parallel
    )

    # Create the model on "meta" device, this avoids materializing param buffers.
    with torch.device("meta"):
        model = SimpleTransformer(dim=args.width, depth=args.depth, kv_reduce=args.grouping, remat=args.remat)

    # This wraps all param properties with a shard/gather code.
    model = simple_fsdp.data_parallel(
        model, mesh,
        mode="fully_shard",
        need_ac=not args.remat,  # Only do FSDP's "remat" (of params/comms) if we don't already cover that ourselves.
        mp_policy=simple_fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
        min_bytes=1024*1024,  # Don't shard params that are less than 1MiB
    )

    # Allocate buffers and sharded parameters on GPU
    model.to_empty(device=torch.cuda.current_device())

    # And then run initializers on them, one by one.
    with torch.no_grad():
      with simple_fsdp.disable_data_parallel():  # super important, or nothing happens.
        model.init_weights()
    printpg(model)
    if rank == 0:
        summary_table(model, stats=False)

    # NOTE: Optimizer doesn't alloc here, only allocs on `.step()`.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    vocab_size = model.emb.num_embeddings

    # I actually checked, this double-compile is both faster and more memory-efficient
    # than the usual way of `model = compile(model)`.
    @record_function("fwd")
    @torch.compile
    def _fwd(batch, mask, positions):
        logits = model(batch, mask, positions)
        return F.cross_entropy(logits, data)

    @record_function("bwd")
    @torch.compile
    def _bwd(loss):
        loss.backward()
        optimizer.step()
        return loss

    # If we keep a fixed batch composition forever, we can precompute all this once.
    # This is not realistic, but useful for speed/mem comparison with main.py
    ntoks = args.batch * args.seqlen
    cbm = torch.compile(create_block_mask)
    if not args.randomize:
        @cache
        def get_data_and_mask():
            data = torch.randint(0, vocab_size, (ntoks,), device="cuda")

            # Make [0, 1, 2, 3, ..., 0, 1, 2, 3, ...]
            positions = torch.arange(args.seqlen, dtype=torch.float32, device="cuda")
            positions = torch.tile(positions, [args.batch])

            # Make [0, 0, 0, ..., 1, 1, 1, ...]
            document_id = torch.repeat_interleave(torch.arange(args.batch, device="cuda"), args.seqlen)

            def mask_mod(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                document = document_id[q_idx] == document_id[kv_idx]
                return causal & document
            mask = cbm(mask_mod, B=None, H=None, Q_LEN=ntoks, KV_LEN=ntoks, device=device)
            return data, positions, mask
    else:
        def get_data_and_mask():
            data = torch.randint(0, vocab_size, (ntoks,), device="cuda")
            positions, document_id = random_pack_range(ntoks, min_len=32, max_len=args.seqlen, device="cuda")
            print(f"Created seq with {positions.max().item()} maxlen and {document_id.max().item()} documents")

            # TODO: Probably create a custom constructor instead to avoid OOM here:
            # https://pytorch.org/blog/flexattention/#q-how-can-we-compute-blockmask-quicker
            def mask_mod(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                document = document_id[q_idx] == document_id[kv_idx]
                return causal & document
            mask = cbm(mask_mod, B=None, H=None, Q_LEN=ntoks, KV_LEN=ntoks, device=device)
            return data, positions, mask

    peak_mems = []
    step_times = []
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,  # Done with torch.cuda functions instead.
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )

    for step in range(16):
        data, positions, mask = get_data_and_mask()

        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        if step == 3:
            torch.cuda.cudart().cudaProfilerStart()
            prof.start()

        model.zero_grad(set_to_none=True)
        loss = _fwd(data, mask, positions)
        _bwd(loss)
        prints(f"step {step}: loss {loss.item():.2f}")  # Causes a transfer.

        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)  # ms
        peak_mems.append(torch.cuda.max_memory_allocated() / 1024**2)  # MiB

        if step == 2:
            # dumping first 3 iterations from init are enough to include optimizer states.
            # Otherwise the .pkl becomes too big and freezes chrome.
            # Drag .pkl file to https://docs.pytorch.org/memory_viz
            torch.cuda.memory._dump_snapshot(f"prof_memsnap_r{rank}.pkl")
        if step == 6:
            torch.cuda.cudart().cudaProfilerStop()
            prof.stop()
            prof.export_chrome_trace(f"prof_trace_r{rank}.json.gz")  # TODO: speedup gz
            prof.export_stacks(f"prof_stacks_cpu_r{rank}.txt")

        distr.barrier()  # Just for simplicity for now.
        printpg(model)

    prints(f"Peak mems (med: {np.median(peak_mems):.1f}MiB): {' '.join(f'{t:.0f}' for t in peak_mems)}")
    prints(f"Step times (med: {np.median(step_times):.1f}ms): {' '.join(f'{t:.0f}' for t in step_times)}")
    torch._dynamo.reset()  # Avoid hang: https://x.com/main_horse/status/1937900381574717940
    distr.destroy_process_group()
    prints(f"Destroyed group")


###############
# UTILS BELOW #
###############
def print_stamped(s, *a, rank, **kw):
    t = datetime.now().time().isoformat(timespec='milliseconds')
    print(f"[{rank} {t}] {s}", *a, **kw)


def global_reduce(x, method):
    """Call `method` such as norm, mean, std, ... and return global scalar."""
    if x is None:
        return None

    n = getattr(x, method)()

    # For DTensor, this returns a _NormPartial object which only holds the local "norm piece".
    # Only calling `full_tensor` on it syncs the output and gives each rank the same whole norm!
    if hasattr(n, 'full_tensor'):
        n = n.full_tensor()

    return n.item()


def global_norms(x):
    n = global_reduce(x, "norm")
    return f"{n:.5f}" if n is not None else "None"


def print_pg(model, verbosity=2, rank=0, file=sys.stderr):
    if verbosity < 2: return

    # Print all param norms and gradnorms to stderr
    torch.cuda.synchronize()
    distr.barrier()
    for name, param in model.named_parameters():
        print_stamped(f"{name:>22s}: ‖p‖={global_norms(param)} ‖∇‖={global_norms(param.grad)}", rank=rank, file=file)


def swissnum(x):
    return f"{x:_}".replace("_", "'")


def summary_table(model, stats=True):
    import rich
    from rich.table import Table

    tbl = Table(show_header=True, header_style="bold magenta",
                show_footer=True, footer_style="bold magenta",
                box=rich.box.HORIZONTALS)
    tbl.add_column("name", justify="left")
    tbl.add_column("shape", justify="right")
    tbl.add_column("dtype", justify="right")
    tbl.add_column("params", justify="right")
    tbl.add_column("placement", justify="right")
    tbl.add_column("local shape", justify="right")
    if stats:
        tbl.add_column("mean", justify="right")
        tbl.add_column("std", justify="right")

    total_num, total_bytes, local_bytes = 0, 0, 0
    for name, x in itertools.chain(model.named_parameters(), model.named_buffers()):
        total_num += x.numel()
        total_bytes += x.nbytes
        cols = [name]
        cols += [str(tuple(x.shape))]
        cols += [str(x.dtype)[len("torch."):]]
        cols += [swissnum(x.numel())]
        if hasattr(x, "placements"):
            cols += [str(x.placements), str(tuple(x.to_local().shape))]
            local_bytes += x.to_local().nbytes
        else:
            cols += ["-", "shape"]
        if stats:
            cols += [global_reduce(x, "mean"), global_reduce(x, "std")]
        tbl.add_row(*cols)

    tbl.columns[0].footer = f"Total: {swissnum(total_num)}"
    tbl.columns[1].footer = f"({total_bytes/1024/1024:.0f}MiB)"
    tbl.columns[2].footer = f"Local: {local_bytes/1024/1024:.0f}MiB"
    rich.print(tbl)


def random_pack_range(total, min_len, max_len, *, device=None, generator=None):
    """
    Randomly chop `total` tokens into variable-length contiguous segments,
    each length in [min_len, max_len] (inclusive).

    Returns
    -------
    positions     : torch.long  [total]
        0 … (len_i-1) inside every segment.
    document_id   : torch.long  [total]
        0,0,…,1,1,…,2,2,… with one contiguous block per segment.
    """
    # --- sanity checks ------------------------------------------------------
    if not (1 <= min_len <= max_len):
        raise ValueError("need 1 ≤ min_len ≤ max_len")
    # smallest / largest feasible number of segments
    min_B = (total + max_len - 1) // max_len          # ceil(total / max_len)
    max_B = total // min_len                          # floor(total / min_len)
    if min_B > max_B:
        raise ValueError("Impossible combination of total / min_len / max_len")

    g   = generator
    dev = device

    # --- choose a random segment count B ------------------------------------
    B = torch.randint(min_B, max_B + 1, (1,),
                      generator=g, device=dev).item()

    # start with every segment at min_len
    lengths  = torch.full((B,), min_len, device=dev, dtype=torch.long)
    extras   = torch.zeros_like(lengths)              # how many tokens to add
    spare    = total - B * min_len                    # tokens left to distribute
    extra_cap = max_len - min_len                     # per-segment head-room

    # sprinkle the spare tokens randomly without exceeding max_len
    while spare > 0:
        idx = torch.randperm(B, generator=g, device=dev)
        for i in idx:
            if spare == 0:
                break
            if extras[i] < extra_cap:
                extras[i] += 1
                spare -= 1
    lengths += extras                                 # final segment lengths

    # --- build document_id and positions ------------------------------------
    document_id = torch.repeat_interleave(
        torch.arange(B, device=dev, dtype=torch.long), lengths)

    starts    = torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths,
                                        lengths)
    positions = (torch.arange(total, device=dev) - starts).long()

    return positions, document_id


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    parser = argparse.ArgumentParser(description="iykyk")
    parser.add_argument("-b", "--batch", type=int, default=8,
                        help="Per-device batch-size.")
    parser.add_argument("-s", "--seqlen", type=int, default=4096,
                        help="Sequence length in tokens.")
    parser.add_argument("-d", "--depth", type=int, default=4,
                        help="Depth of the model, in blocks.")
    parser.add_argument("-w", "--width", type=int, default=4096,
                        help="Width of the model, aka d_model.")
    parser.add_argument("-g", "--grouping", type=int, default=4,
                        help="Reduction factor for GQA (1=MHA)")
    parser.add_argument("--no-remat", dest="remat", action="store_false",
                        help="Disable rematerialization aka selective activation checkpointing.")
    parser.set_defaults(remat=True)
    parser.add_argument("-r", "--randomize", action="store_true", default=False,
                        help="Whether to use fixed batch structure, or randomize at each step.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: print memory and timings. -vv: Also print expensive details like gradnorms.")
    args = parser.parse_args()

    main(args, rank=rank, local_rank=local_rank, world_size=world_size)
